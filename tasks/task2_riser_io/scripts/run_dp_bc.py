"""Synchronized three-camera recordings of contact-driven assembly attempts."""
from pathlib import Path
import argparse,sys,json,csv,traceback,time
import numpy as np
from scipy.spatial.transform import Rotation
R=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io')
COLLECT=R/'sim'/'collect'
ARTIFACTS=Path('/mnt/data2/wjy_SERL/Embodied_Safety/outputs/task2_riser_io')
sys.path.insert(0,str(COLLECT))
from acceptance import ACCEPTANCE_VERSION,task_assembly_hold,hold_passed,authored_assembly_geometry,ring_on_seat
def assembly_hold(rows,hz,target):
 return task_assembly_hold(rows,hz,target,assembly_geometry)
from motion_timing import motion_steps as quintic_motion_steps
from ik_jacobian import pose_residual_jacobian
motion_config=json.loads((R/'configs/motion.json').read_text())
runtime_config=json.loads((R/'configs/runtime.json').read_text())
physics_device=runtime_config.get('physics_device','cuda:0')
if physics_device not in ('cpu','cuda:0'):raise ValueError('Unsupported physics device: '+physics_device)
shared_app=globals().get('_simulation_app')
scene_session=globals().get('_scene_session',{})
progress_callback=globals().get('_progress_callback')
pa=argparse.ArgumentParser()
pa.add_argument('--seed',type=int,default=0);pa.add_argument('--record',action='store_true');pa.add_argument('--probe-wire',action='store_true')
pa.add_argument('--physics-hz',type=int,choices=[120,240,480],default=120)
pa.add_argument('--collection-config',type=Path)
args=pa.parse_args()
collection_config=globals().get("_collection_config",json.loads((args.collection_config or (COLLECT/"feedback_dart.json")).read_text()))
if collection_config and collection_config["method"] not in ("expert","dart","dagger"):raise ValueError("Unknown collection method")
reset_spec=None;target=np.array([-.092,-.071,.818]);board_rotation=np.eye(3)
expert=None;router=None;policy_client=None;expert_label=None;action_source="hold"
out=Path(globals().get('_episode_root',ARTIFACTS/'data'/'raw'))/('episode_'+str(args.seed).zfill(4))
if out.exists():raise RuntimeError('episode already exists')
import io
records=[];frames=[];events=[];image_frames=[];writer=None;app=None;phase='initializing';stepnum=0
rng=np.random.default_rng(args.seed)
variation={'carry_height_m':float(rng.uniform(.865,.885)),'waypoint_x_m':float(rng.uniform(-.008,.008)),'waypoint_y_m':float(rng.uniform(-.008,.008)),'duration_scale':float(rng.uniform(.95,1.15))}
report={'scene':'riser_io_success_v014.usda','full_task_completed':False,'safety_verdict':'UNASSESSED','ring_geometry_unchanged':True,'task_body_pose_writes_after_reset':False,'seed':args.seed,'variation':variation}
report['acceptance_version']=ACCEPTANCE_VERSION
report['collection_method']=collection_config.get("method","expert")
report['collection_config']=collection_config
report['motion_config']=motion_config
report['runtime_config']=runtime_config
report['ik_jacobian']='analytic_pose_log_rotation'
report['runtime_metadata']=globals().get('_runtime_metadata',{})
motion_segments=[]
report['clamp_force_role']='diagnostic only; no minimum force magnitude'
contact_log=io.StringIO()
camera_writers={}
performance_totals={}
wall_started=time.perf_counter()
def measured(name):
 def decorate(fn):
  def call(*a,**kw):
   start=time.perf_counter()
   try:return fn(*a,**kw)
   finally:
    entry=performance_totals.setdefault(name,{'seconds':0.,'calls':0})
    entry['seconds']+=time.perf_counter()-start;entry['calls']+=1
  return call
 return decorate
def motion_steps(config, phase_name, nominal_seconds, distance_m, angle_rad, duration_variation):
 limits={**config["default"],**config["phases"].get(phase_name,{})}
 if not limits.get("cruise_rotation",False) or abs(angle_rad)<=.5:
  return quintic_motion_steps(config,phase_name,nominal_seconds,distance_m,angle_rad,duration_variation)
 ramp=limits["ramp_seconds"]
 seconds=max(nominal_seconds*limits["duration_scale"]*duration_variation,
             distance_m/(limits["peak_linear_mm_s"]/1000.)+ramp,
             abs(angle_rad)/limits["peak_angular_rad_s"]+ramp,
             1.5*distance_m/(ramp*(limits["peak_linear_acceleration_mm_s2"]/1000.))+ramp,2*ramp)
 return int(np.ceil(seconds*60))

def motion_fraction(t, steps, angle_rad):
 limits=motion_config["phases"].get(phase,{})
 if not limits.get("cruise_rotation",False) or abs(angle_rad)<=.5:
  return t*t*t*(10+t*(-15+6*t))
 duration=steps/60.;ramp=limits["ramp_seconds"];elapsed=t*duration
 def ramp_distance(x):return (x**3/ramp**2-.5*x**4/ramp**3)/(duration-ramp)
 if elapsed<ramp:return ramp_distance(elapsed)
 if elapsed>duration-ramp:return 1.-ramp_distance(duration-elapsed)
 return (elapsed-.5*ramp)/(duration-ramp)

def performance_snapshot():
 return {'wall_seconds':time.perf_counter()-wall_started,'simulation_seconds':stepnum/args.physics_hz,'components':performance_totals}


try:
 sys.argv=[sys.argv[0]]
 from isaacsim import SimulationApp
 app=shared_app if shared_app is not None else SimulationApp({'headless':True,'limit_cpu_threads':4,'width':128,'height':128,'sync_loads':True,'extra_args':['--/log/file='+str(Path(globals().get('_kit_log',ARTIFACTS/'logs'/'current_kit.log')))]})
 import omni.usd,omni.physics.tensors as tensors,torch
 torch.set_num_threads(1)
 from pxr import UsdGeom,UsdPhysics,UsdLux,Gf
 from isaacsim.core.api import SimulationContext
 from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver,interface_config_loader
 if not scene_session:
  omni.usd.get_context().open_stage(str(R/'scene/riser_io_success_v014.usda'))
  for _ in range(8):app.update()
 stage=omni.usd.get_context().get_stage();stage.SetEditTarget(stage.GetSessionLayer())
 from randomization import capture_initial_transforms,restore_initial_transforms
 if scene_session:
  restore_initial_transforms(stage,scene_session["initial_transforms"])
  initial_transforms=scene_session["initial_transforms"]
 else:initial_transforms=capture_initial_transforms(stage)
 assembly_geometry=scene_session.get("assembly_geometry") or authored_assembly_geometry(stage)
 report["assembly_geometry"]=assembly_geometry
 report["acceptance_policy"]="released functional assembly; numerical limits recorded only"
 report["scene_reused"]=bool(scene_session)
 if collection_config.get("randomize"):
  from randomization import sample_reset,apply_reset,route_waypoint,rotation as reset_rotation
  reset_spec=globals().get("_initial_conditions") or sample_reset(args.seed)
  target,board_rotation=apply_reset(stage,reset_spec)
  report["initial_conditions"]=reset_spec;report["route_mode"]=reset_spec["route_mode"]
 report["target_position_m"]=target.tolist()
 UsdLux.DomeLight.Define(stage,'/World/DemoLight').CreateIntensityAttr(1500.)
 from pxr import PhysxSchema
 for p in stage.Traverse():
  if p.HasAPI(UsdPhysics.RigidBodyAPI):
   if str(p.GetPath())=='/ScrewTray/Nest_1/Screw_1':
    PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr(0.)
   elif p.HasAPI(PhysxSchema.PhysxContactReportAPI):
    p.RemoveAPI(PhysxSchema.PhysxContactReportAPI)
   body_api=PhysxSchema.PhysxRigidBodyAPI.Apply(p)
   body_api.CreateSolverPositionIterationCountAttr(runtime_config['solver_position_iterations'])
   body_api.CreateSolverVelocityIterationCountAttr(runtime_config['solver_velocity_iterations'])
  if p.HasAPI(UsdPhysics.ArticulationRootAPI):
   articulation_api=PhysxSchema.PhysxArticulationAPI.Apply(p)
   articulation_api.CreateSolverPositionIterationCountAttr(runtime_config['articulation_position_iterations'])
   articulation_api.CreateSolverVelocityIterationCountAttr(runtime_config['solver_velocity_iterations'])
 for p in stage.Traverse():
  if p.IsA(UsdPhysics.Scene):
   physics=PhysxSchema.PhysxSceneAPI.Apply(p)
   physics.CreateEnableGPUDynamicsAttr(physics_device!='cpu')
   physics.CreateBroadphaseTypeAttr('MBP' if physics_device=='cpu' else 'GPU')
 report['physics_backend']='PhysX CPU' if physics_device=='cpu' else 'PhysX GPU'
 report['contact_reporting']='screw contact pairs used by assembly controller and clamp verification'
 for path in ['/World/Robot','/World/CableSensor']:
  PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(path)).CreateSolverPositionIterationCountAttr(runtime_config['articulation_position_iterations'])
 report['solver_position_iterations']=runtime_config['solver_position_iterations']
 report['solver_velocity_iterations']=runtime_config['solver_velocity_iterations']
 physics_hz=args.physics_hz
 report['physics_rate_Hz']=physics_hz;report['measurement_rate_Hz']=120 if args.record else 60
 dt=1/physics_hz
 if scene_session and str(scene_session["sim"].device)==physics_device:
  sim=scene_session["sim"]
  sim.reset()
 else:
  if scene_session:SimulationContext.clear_instance()
  sim=SimulationContext(physics_dt=dt,rendering_dt=1/60,backend='torch',device=physics_device)
  sim.get_physics_context().enable_fabric(runtime_config['fabric'])
  sim.initialize_physics();sim.play()
 report['fabric_enabled']=runtime_config['fabric']
 for _ in range(4):sim.step(render=True)
 view=tensors.create_simulation_view('torch');view.set_subspace_roots('/')
 class HostObservations:
  """Expose GPU physics state as NumPy observations to the controller."""
  def __init__(self, native):self.native=native
  def __getattr__(self, name):
   member=getattr(self.native,name)
   if not callable(member):return member
   def call(*args,**kwargs):
    def upload(x):
     if isinstance(x,np.ndarray):
      if x.dtype==np.uint32:x=x.astype(np.int32)
      return torch.as_tensor(x,device='cpu' if name in ['set_dof_max_forces','set_dof_stiffnesses','set_dof_dampings'] else physics_device).contiguous()
     return x
    def download(x):
     if isinstance(x,torch.Tensor):return x.detach().cpu().numpy()
     if isinstance(x,(tuple,list)):return type(x)(download(v) for v in x)
     return x
    value=member(*(upload(v) for v in args),**kwargs)
    if name in ['get_contact_data','get_friction_data']:
     counts,starts=value[-2:]
     used=int(torch.where(counts>0,starts+counts,0).max().item())
     value=tuple(v[:used] for v in value[:-2])+tuple(value[-2:])
    return download(value)
   return call
 robot=HostObservations(view.create_articulation_view('/World/Robot'))
 sensor=HostObservations(view.create_articulation_view('/World/CableSensor'))
 seg=HostObservations(view.create_rigid_body_view('/wire1/Segment_*'))
 rv=HostObservations(view.create_rigid_body_view('/wire1/_0347300_ring_terminal'))
 sv=HostObservations(view.create_rigid_body_view('/ScrewTray/Nest_1/Screw_1'))
 assert robot.count==1 and sensor.count==1 and seg.count==24 and rv.count==1
 names=list(robot.shared_metatype.dof_names);ln=list(sensor.shared_metatype.link_names);si=ln.index('Segment_00')
 ci=next(i for i,n in enumerate(ln) if 'TE_L26620011' in n)
 report['sensor_links']=ln
 report['robot_dof_names']=names
 report['action_representation']='joint position targets at 60 Hz; radians for arm joints and meters for fingers'
 report['clamp_measurement']='normal force under the screw head, evaluated after releasing the gripper'
 solver=LulaKinematicsSolver(**interface_config_loader.load_supported_lula_kinematics_solver_config('Franka'))
 ai=np.array([names.index(n) for n in solver.get_joint_names()]);fi=np.array([names.index('panda_finger_joint1'),names.index('panda_finger_joint2')]);ix=np.array([0],np.uint32)
 base=robot.get_root_transforms()[0];solver.set_robot_base_pose(base[:3],np.r_[base[6],base[3:6]])
 @measured("inverse_kinematics")
 def solve(p,rr,seed):
  joints,ok=(solver.compute_inverse_kinematics('panda_hand',p,np.r_[rr[3],rr[:3]],warm_start=seed,position_tolerance=.00002,orientation_tolerance=.0002) if phase=='initializing' else (None,False))
  if ok and (phase=='initializing' or np.max(np.abs(np.asarray(joints)-seed))<.03):return np.asarray(joints)
  from scipy.optimize import least_squares
  goal=Rotation.from_quat(rr)
  def residual(qs):
   pp,rot=solver.compute_forward_kinematics('panda_hand',qs)
   return np.r_[(np.asarray(pp)-p)*10,(Rotation.from_matrix(rot)*goal.inv()).as_rotvec()]
  lo=np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-.0175,-2.8973]);hi=np.array([2.8973,1.7628,2.8973,-.0698,2.8973,3.7525,2.8973])
  if phase!='initializing':
   bounds=(np.maximum(lo+1e-7,seed-.03),np.minimum(hi-1e-7,seed+.03))
   def local_residual(x):return np.r_[residual(x),.01*(x-seed)]
   def local_jacobian(x):return pose_residual_jacobian(solver._kinematics,x,solver._robot_rot,solver._meters_per_unit,residual(x)[3:])
   r=least_squares(local_residual,np.clip(seed,*bounds),jac=local_jacobian,bounds=bounds,max_nfev=60,ftol=1e-10,xtol=1e-10,gtol=1e-10)
   err=residual(r.x)
   if np.linalg.norm(err[:3])<.0002 and np.linalg.norm(err[3:])<.0002:return r.x
   report["largest_ik_residual"]=max(report.get("largest_ik_residual",0.),float(np.linalg.norm(err)))
   if np.isfinite(r.x).all():return r.x
   raise RuntimeError('IK produced nonfinite command')
  best=None
  for start in [seed,[0,-.8,0,-2.6,0,1.8,.7854],[.5,.2,-.5,-2.,0,2.3,0],[-.5,.2,.5,-2.,0,2.3,1.5]]:
   r=least_squares(residual,np.clip(start,lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=160,ftol=1e-10,xtol=1e-10,gtol=1e-10)
   if best is None or np.linalg.norm(r.fun)<np.linalg.norm(best.fun):best=r
   if np.linalg.norm(r.fun[:3])<.005 and np.linalg.norm(r.fun[3:])<.01:return r.x
  raise RuntimeError('IK unreachable: '+str(p)+' residual '+str(best.fun))
 Rdown=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]])
 q=robot.get_dof_positions()[0].copy();q[ai]=[0,-.5,0,-2.2,0,1.7,.7854];q[fi]=.04
 initial_hand=np.array([-.285,.02,1.0434])
 initial_rotation=Rdown
 if reset_spec is not None:
  initial_hand+=reset_spec["robot_xyz_m"]
  initial_rotation=Rotation.from_euler('xyz',reset_spec["robot_rpy_deg"],degrees=True).as_matrix()@Rdown
 q[ai]=solve(initial_hand,Rotation.from_matrix(initial_rotation).as_quat(),q[ai])
 robot.set_dof_positions(q[None].astype(np.float32),ix);robot.set_dof_velocities(np.zeros_like(q)[None].astype(np.float32),ix);robot.set_dof_position_targets(q[None].astype(np.float32),ix)
 forces=robot.get_dof_max_forces().copy();forces[0,fi]=15.;robot.set_dof_max_forces(forces,ix)
 stiffness=robot.get_dof_stiffnesses().copy();stiffness[0,fi]=400.;robot.set_dof_stiffnesses(stiffness,ix)
 damping=robot.get_dof_dampings().copy();damping[0,fi]=15.;robot.set_dof_dampings(damping,ix)
 report['finger_control']={'stiffness_N_m':400.,'damping_Ns_m':15.,'effort_cap_N':15.}
 if collection_config:
  from feedback import JointCommandLimits,FeedbackExpert,RecoveryRouter
  from dagger_router import DaggerRouter
  from interface import observation_state
  if not np.array_equal(ai,np.arange(7)):raise RuntimeError("Task2 joint contract order changed")
  expert=FeedbackExpert(solve,lambda arm:solver.compute_forward_kinematics('panda_hand',arm),JointCommandLimits(q))
  if collection_config["method"]=="dagger":
   from client import PolicyClient
   policy_client=PolicyClient(collection_config["checkpoint"],device=collection_config.get("policy_device","cpu"),n_action_steps=collection_config["n_action_steps"],fixed_noise=collection_config["fixed_noise"],seed=args.seed,backend=collection_config.get("policy_backend","legacy"))
  router=(DaggerRouter if collection_config["method"]=="dagger" else RecoveryRouter)(expert,collection_config,args.seed,policy_client)
  expert_label=q.copy()
 jp0=np.array([stage.GetPrimAtPath(f'/wire1/Bend_{i:02d}').GetAttribute('physics:localPos0').Get() for i in range(23)])
 jp1=np.array([stage.GetPrimAtPath(f'/wire1/Bend_{i:02d}').GetAttribute('physics:localPos1').Get() for i in range(23)])
 screw_contacts=HostObservations(view.create_rigid_contact_view('/ScrewTray/Nest_1/Screw_1',['/NutPlate_Coarse/Nut_1/IndependentNut','/wire1/_0347300_ring_terminal','/PCB_4_Terminal/SingleSeat'],2048))
 step_physics=measured('physics_step')(sim.step)
 latched=False
 latest={}
 @measured("state_contact_readback")
 def capture_states():
  normal_data=screw_contacts.native.get_contact_data(dt)
  # PhysX reuses the pair index buffers for friction data on the same contact view.
  normal_data=tuple(normal_data[:-2])+tuple(x.clone() for x in normal_data[-2:])
  requests={
   'segments':seg.native.get_transforms(),'ring':rv.native.get_transforms(),
   'screw':sv.native.get_transforms(),'ring_velocity':rv.native.get_velocities(),
   'joints':robot.native.get_dof_positions(),'joint_velocities':robot.native.get_dof_velocities(),
   'screw_contacts':normal_data,'screw_friction':screw_contacts.native.get_friction_data(dt)}
  chunks=[];layout=[];offset=0
  for name,value in requests.items():
   values=value if isinstance(value,(tuple,list)) else [value]
   entries=[]
   for v in values:
    size=v.numel();entries.append((offset,size,tuple(v.shape)));offset+=size
    chunks.append(v.flatten().to(torch.float32))
   layout.append((name,entries,isinstance(value,(tuple,list))))
  packed=torch.cat(chunks).detach().cpu().numpy()
  latest.clear()
  for name,entries,islist in layout:
   values=[packed[o:o+n].reshape(shape) for o,n,shape in entries]
   latest[name]=values if islist else values[0]
 def screw_measure():
  sp=latest['screw'][0].copy();force=np.zeros(3);moment=np.zeros(3)
  sf,cp,cn,dep,counts,starts=latest['screw_contacts']
  ff,fp,fc,fs=latest['screw_friction']
  for nn,ss in zip(counts.ravel(),starts.ravel()):
   sl=slice(int(ss),int(ss+nn))
   if nn:
    v=sf[sl]*cn[sl];force+=v.sum(axis=0);moment+=np.cross(cp[sl]-sp[:3],v).sum(axis=0)
  for nn,ss in zip(fc.ravel(),fs.ravel()):
   sl=slice(int(ss),int(ss+nn))
   if nn:force+=ff[sl].sum(axis=0);moment+=np.cross(fp[sl]-sp[:3],ff[sl]).sum(axis=0)
  axis=Rotation.from_quat(sp[3:]).apply([0,0,1]);axial=float(np.dot(force,axis))
  return sp,abs(axial),float(np.linalg.norm(force-axis*axial)),abs(float(np.dot(moment,axis)))

 def handpose():return solver.compute_forward_kinematics('panda_hand',latest['joints'][0,ai] if latest else robot.get_dof_positions()[0,ai])
 def tick(n=1):
  global stepnum,latched
  for _ in range(n):
   step_physics(render=False);stepnum+=1
   capture_states()
   analysis_started=time.perf_counter()
   tr=latest['segments'];xyz=tr[:,:3];ring=latest['ring'][0]
   gaps=np.linalg.norm(xyz[:-1]+Rotation.from_quat(tr[:-1,3:]).apply(jp0)-xyz[1:]-Rotation.from_quat(tr[1:,3:]).apply(jp1),axis=1)
   sp,sax,slat,storque=screw_measure();sq=Rotation.from_quat(sp[3:])
   if not hasattr(tick,'last_screw_rotation'):tick.last_screw_rotation=sq;tick.screw_angle=0.
   tick.screw_angle+=float(np.dot((sq*tick.last_screw_rotation.inv()).as_rotvec(),sq.apply([0,0,1])))
   tick.last_screw_rotation=sq
   cf,cp,cn,cd,cc,cs=latest['screw_contacts']
   start=int(cs.ravel()[1]);count=int(cc.ravel()[1]);part=slice(start,start+count)
   local=sq.inv().apply(cp[part]-sp[:3])
   under_head=(np.linalg.norm(local[:,:2],axis=1)>.0055)&(np.abs(local[:,2]-.018)<.0005)
   head_force=(cf[part]*cn[part])[under_head].sum(axis=0)
   rec={'time_s':stepnum*dt,'phase':phase,'screw_axial_N':sax,'screw_lateral_N':slat,'screw_torque_Nm':storque,
    'screw_tip_z':float(sp[2]),'screw_x_m':float(sp[0]),'screw_y_m':float(sp[1]),
    'screw_tilt_deg':float(np.degrees(np.arccos(np.clip(sq.apply([0,0,1])[2],-1,1)))),
    'screw_rotation_accumulated_deg':float(np.degrees(tick.screw_angle)),
    'clamp_N':float(abs(np.dot(head_force,sq.apply([0,0,1])))),
    'head_contact_count':int(np.count_nonzero(under_head)),
    'nut_contact_count':int(cc.ravel()[0]),
    'head_contact_signed_sum_N':float(cf[part].reshape(-1)[under_head].sum()),
    'ring_x':float(ring[0]),'ring_y':float(ring[1]),'ring_z':float(ring[2]),
    'ring_speed_m_s':float(np.linalg.norm(latest['ring_velocity'][0,:3])),
    'ring_tilt_deg':float(np.degrees(np.arccos(np.clip(Rotation.from_quat(ring[3:]).as_matrix()[2,2],-1,1)))),
    'robot_joint_positions':latest['joints'][0].tolist(),
    'robot_joint_velocities':latest['joint_velocities'][0].tolist(),'robot_joint_targets':q.tolist(),
    'joint_gap_m':float(gaps.max()),'contact_separation_m':0.,
    'expert_joint_targets':(q if expert_label is None else expert_label).tolist(),
    'action_source':action_source}
   records.append(rec)
   active=phase not in ['settle','calibration','initializing']
   if stepnum%(physics_hz//60)==0:
    commanded=q[ai].copy()
    if hasattr(tick,'previous_command'):
     command_v=(commanded-tick.previous_command)*60.
     if hasattr(tick,'previous_command_v') and active:
      command_a=(command_v-tick.previous_command_v)*60.
      if np.max(np.abs(command_v))>2. or np.max(np.abs(command_a))>12.:
       report['command_limit_events']=report.get('command_limit_events',0)+1
     tick.previous_command_v=command_v
    tick.previous_command=commanded
   if not np.isfinite(xyz).all():raise RuntimeError('physics produced nonfinite body positions')
   entry=performance_totals.setdefault('state_analysis',{'seconds':0.,'calls':0})
   entry['seconds']+=time.perf_counter()-analysis_started;entry['calls']+=1
   if args.record and stepnum%(physics_hz//10)==0:write_frame(rec,tr,ring)
   if stepnum%(physics_hz*5)==0:print('PERFORMANCE',json.dumps(performance_snapshot()),flush=True)
   if stepnum%physics_hz==0:
    if phase=='thread_turn':
     geometry=report['assembly_geometry']
     separation=float(np.linalg.norm(np.array([rec['ring_x'],rec['ring_y']])-target[:2]))
     if separation>geometry['seat_radius_m']+geometry['ring_radius_m']:
      raise RuntimeError('Assembly incomplete: terminal no longer overlaps the seat during threading.')
    reset_request=Path(globals().get('_reset_request',ARTIFACTS/'reset_requested'))
    if reset_request.exists():
     reason=reset_request.read_text().strip();reset_request.unlink()
     raise RuntimeError('Assembly attempt reset: '+reason)
    print('STATE',json.dumps({k:rec[k] for k in ['time_s','phase','ring_x','ring_y','ring_z','clamp_N']}),flush=True)
    if progress_callback is not None:progress_callback(dict({k:rec[k] for k in ['time_s','phase','ring_x','ring_y','ring_z','clamp_N']}, execution_counts=router.counts if router else {}, phase_execution_counts=getattr(router,'phase_counts',{}), teacher_interventions=router.interventions if router else 0),performance_snapshot())
 def setup_render():
  import omni.replicator.core as rep
  if hasattr(setup_render,'anns'):return
  if scene_session.get("render"):
   for name,value in scene_session["render"].items():setattr(setup_render,name,value)
   return
  external=UsdGeom.Camera.Define(stage,'/World/ExternalCamera')
  external.CreateFocalLengthAttr(23.)
  external.CreateClippingRangeAttr(Gf.Vec2f(.005,10.))
  mat=Gf.Matrix4d().SetLookAt(Gf.Vec3d(.13,-.57,1.35),Gf.Vec3d(-.15,.055,.86),Gf.Vec3d(0,0,1)).GetInverse()
  UsdGeom.Xformable(external).AddTransformOp().Set(mat)
  paths={'external':'/World/ExternalCamera','wrist_left':'/World/Robot/panda_hand/WristCam_L','wrist_right':'/World/Robot/panda_hand/WristCam_R'}
  setup_render.anns={};setup_render.products=[];setup_render.paths=paths;manifest={}
  for name,path in paths.items():
   p=stage.GetPrimAtPath(path)
   if not p or p.GetTypeName()!='Camera':raise RuntimeError('missing camera '+path)
   product=rep.create.render_product(path,(128,128));setup_render.products.append(product)
   ann=rep.AnnotatorRegistry.get_annotator('rgb');ann.attach([product]);setup_render.anns[name]=ann
   c=UsdGeom.Camera(p)
   manifest[name]={'path':path,'resolution':[128,128],'focal_length':c.GetFocalLengthAttr().Get(),'horizontal_aperture':c.GetHorizontalApertureAttr().Get(),'vertical_aperture':c.GetVerticalApertureAttr().Get(),'local_matrix':np.asarray(UsdGeom.Xformable(p).GetLocalTransformation()).tolist()}
  manifest['external_configuration']='fixed collection camera; task2 has no saved external camera calibration'
  manifest['timestamp']='simulation time; all cameras rendered without intervening physics steps'
  setup_render.manifest=manifest
  for attempt in range(60):
   sim.render()
   if all(a.get_data() is not None and a.get_data().size for a in setup_render.anns.values()):break
  else:raise RuntimeError('three-camera render initialization failed')
 @measured("camera_render_readback")
 def write_frame(rec,tr,ring):
  setup_render();sim.render();images={}
  for name,ann in setup_render.anns.items():
   rgb=ann.get_data()
   if rgb is None or not rgb.size:raise RuntimeError('empty camera '+name)
   images[name]=np.ascontiguousarray(rgb[:,:,:3]).copy()
  image_frames.append(images)
  frames.append({'frame':len(frames),'time_s':rec['time_s'],'phase':phase,
   'ring_pose':ring.tolist(),'screw_pose':latest['screw'][0].tolist(),
   'robot_joints':latest['joints'][0].tolist(),'physics_step':stepnum})
 def event(name):
  row={'event':name,'time_s':stepnum*dt,'ring_pose':rv.get_transforms()[0].tolist()};events.append(row);print('EVENT',json.dumps(row),flush=True)
 intended_grip_width=float(q[fi].sum())
 def dispatch_feedback(force_expert=False):
  global q,expert_label,action_source
  limits={**motion_config["default"],**motion_config["phases"].get(phase,{})}
  expert.set_motion_limits(limits["peak_linear_mm_s"]/1000.,limits["peak_angular_rad_s"])
  j=pose_residual_jacobian(solver._kinematics,latest["joints"][0,ai],solver._robot_rot,solver._meters_per_unit,np.zeros(3))[:6]
  j[:3]/=10.
  images=image_frames[-1] if image_frames else None
  state=observation_state(latest["joints"][0],latest["joint_velocities"][0],q)
  expert_label,q,action_source=router.command(latest,phase,j,images,state,len(frames),force_expert=force_expert)
  robot.set_dof_position_targets(q[None].astype(np.float32),ix)
  tick(physics_hz//60)
 def control_schedule(steps,settle_steps=0):
  if expert is None:
   yield from range(steps+settle_steps);return
  from motion_progress import MotionProgress,MotionTimeout
  completion=.006 if phase=="thread_turn" else .002
  tracker=MotionProgress(steps,settle_steps,completion_error=completion)
  original_override=router.teacher_override
  original_reason=getattr(router,"teacher_override_reason","task_recovery")
  recovery_extended=False
  try:
   while True:
    if collection_config["method"]=="dagger" and tracker.cursor>=tracker.steps:
     router.teacher_override=True
     router.teacher_override_reason="phase_completion"
    yield tracker.index
    try:
     if tracker.update(expert.tracking_error(latest),float(np.max(np.abs(expert.limits.velocity)))):return
    except MotionTimeout as exc:
     if collection_config["method"]=="dagger" and not recovery_extended:
      recovery_extended=True
      router.teacher_override=True
      router.teacher_override_reason="motion_timeout_recovery"
      tracker.budget=tracker.elapsed+1800
      report.setdefault("teacher_timeout_recoveries",[]).append(dict(phase=phase,time_s=stepnum/physics_hz,error_m=expert.tracking_error(latest)))
      continue
     report.setdefault("incomplete_reference_tracking_phases",[]).append(phase)
     raise RuntimeError("Expert could not complete "+phase+": "+str(exc)) from exc
  finally:
   router.teacher_override=original_override
   router.teacher_override_reason=original_reason
 def hold_feedback(seconds):
  global expert_label,action_source
  # Preserve physical verification time; no artificial nonzero action.
  if expert is not None:
   for _ in range(round(seconds*60)):
    hold_target=q.copy();hold_target[fi]=intended_grip_width/2
    qnext=expert.limits.commit(hold_target)
    q[:]=qnext;expert_label=q.copy();action_source="hold"
    router.counts["hold"]+=1
    robot.set_dof_position_targets(q[None].astype(np.float32),ix);tick(physics_hz//60)
  else:tick(round(seconds*physics_hz))
  if router is not None:router.clear_policy_history()
 def move(tcp,seconds=3.,width=None,rotation=None,stop_force=None):
  global q,intended_grip_width
  desired=Rdown if rotation is None else rotation
  if width is not None:intended_grip_width=float(width)
  desired_width=intended_grip_width
  start,rot=solver.compute_forward_kinematics('panda_hand',q[ai]);end=np.asarray(tcp)-desired@np.array([0,0,.1034])
  r0=Rotation.from_matrix(rot);r1=Rotation.from_matrix(desired);dr=(r1*r0.inv()).as_rotvec()
  steps=motion_steps(motion_config,phase,seconds,float(np.linalg.norm(end-start)),float(np.linalg.norm(dr)),variation['duration_scale']);seed=q[ai].copy()
  motion_segments.append({'phase':phase,'duration_s':steps/60.,'distance_m':float(np.linalg.norm(end-start)),'angle_rad':float(np.linalg.norm(dr))})
  for k in control_schedule(steps):
   t=min(1.,(k+1)/steps);u=motion_fraction(t,steps,float(np.linalg.norm(dr)))
   p=start+(end-start)*u;rr=(Rotation.from_rotvec(dr*u)*r0).as_quat()
   if expert is not None:
    expert.set_reference(p,Rotation.from_quat(rr).as_matrix(),desired_width)
    dispatch_feedback();seed=q[ai].copy();continue
   nxt=solve(p,rr,seed)
   if np.max(np.abs(nxt-seed))>.08:report['large_ik_step_count']=report.get('large_ik_step_count',0)+1
   seed=nxt;q[ai]=seed
   if width is not None:q[fi]=width/2
   robot.set_dof_position_targets(q[None].astype(np.float32),ix);tick(physics_hz//60)
  hold_feedback(motion_config['phases'].get(phase,{}).get('settle_s',.25))
  hp,_=handpose()
  report.setdefault('hand_tracking_errors_m',[]).append(float(np.linalg.norm(np.asarray(hp)-end)))
 def body_move(body,goal,seconds=2.,rotation=None,force_target=None):
  global q
  held_width=intended_grip_width
  initial=body.get_transforms()[0].copy();r0=Rotation.from_quat(initial[3:])
  hp,hr=solver.compute_forward_kinematics('panda_hand',q[ai]);grasp_offset=r0.inv().apply(np.asarray(hp)-initial[:3]);grasp_rotation=r0.inv().as_matrix()@hr
  r1=r0 if rotation is None else Rotation.from_matrix(rotation);dr=(r1*r0.inv()).as_rotvec()
  bias=np.zeros(3);angle_bias=np.zeros(3);admittance=0.
  steps=motion_steps(motion_config,phase,seconds,float(np.linalg.norm(np.asarray(goal)-initial[:3])),float(np.linalg.norm(dr)),variation['duration_scale']);seed=q[ai].copy()
  motion_segments.append({'phase':phase,'duration_s':steps/60.,'distance_m':float(np.linalg.norm(np.asarray(goal)-initial[:3])),'angle_rad':float(np.linalg.norm(dr))})
  for k in control_schedule(steps,round(60*motion_config['phases'].get(phase,{}).get('settle_s',.5))):
   t=min(1.,(k+1)/steps);u=motion_fraction(t,steps,float(np.linalg.norm(dr)))
   desired=initial[:3]+(np.asarray(goal)-initial[:3])*u
   if force_target is not None:
    admittance=float(np.clip(admittance+.000001*(records[-1]['screw_axial_N']-force_target),-.001,.002))
    desired[2]+=admittance
   drt=Rotation.from_rotvec(dr*u)*r0
   if expert is not None:
    expert.set_reference(desired,drt.as_matrix(),held_width,"ring" if body is rv else "screw",grasp_offset,grasp_rotation)
    dispatch_feedback();seed=q[ai].copy();continue
   actual=(latest['ring'] if body is rv else latest['screw'])[0].copy();ar=Rotation.from_quat(actual[3:])
   error=desired-actual[:3];bias=np.clip(bias+.025*error,-.002,.002)
   if force_target is not None:bias[2]=0.
   angle_bias=np.clip(angle_bias+.015*(drt*ar.inv()).as_rotvec(),-.03,.03)
   rd=(Rotation.from_rotvec(angle_bias)*drt).as_matrix()
   hp=desired+bias+rd@grasp_offset;hr=rd@grasp_rotation
   nxt=solve(hp,Rotation.from_matrix(hr).as_quat(),seed)
   if np.max(np.abs(nxt-seed))>.08:report['large_ik_step_count']=report.get('large_ik_step_count',0)+1
   seed=nxt;q[ai]=seed;robot.set_dof_position_targets(q[None].astype(np.float32),ix);tick(physics_hz//60)
 def grip(width,seconds=1.):
  global q,intended_grip_width
  intended_grip_width=float(width)
  start=float(q[fi].sum())
  hp,hr=solver.compute_forward_kinematics('panda_hand',q[ai])
  steps=max(1,round(seconds*60*motion_config['phases'].get(phase,{}).get('grip_duration_scale',1.)))
  for k in range(steps):
   t=(k+1)/steps;fraction=t*t*t*(10+t*(-15+6*t))
   desired_width=start+(width-start)*fraction
   if expert is not None:
    expert.set_reference(hp,hr,desired_width);dispatch_feedback()
   else:
    q[fi]=desired_width/2;robot.set_dof_position_targets(q[None].astype(np.float32),ix);tick(physics_hz//60)
  if expert is not None:
   expert.set_reference(hp,hr,width)
   for _ in range(60):
    dispatch_feedback(force_expert=True)
    if abs(q[fi].sum()-width)<1e-5 and expert.tracking_error(latest)<.002:break
   else:raise RuntimeError("Expert could not establish the gripper reference in "+phase)
 def pick_body(body,offset,rotation,open_width,lift_height,prefix=""):
  global phase
  override=router.teacher_override if router is not None else False
  try:
   for attempt in range(2):
    if attempt and router is not None:router.teacher_override=True
    initial=body.get_transforms()[0].copy()
    grasp=initial[:3]+Rotation.from_quat(initial[3:]).apply(offset)
    phase=prefix+"approach";move(grasp+[0,0,.09],2.,open_width,rotation=rotation)
    phase=prefix+"descend";move(grasp,2.,open_width,rotation=rotation)
    phase=prefix+"grasp";grip(0.,1.)
    phase=prefix+"lift";move(grasp+[0,0,lift_height],2.5,rotation=rotation)
    height=float(body.get_transforms()[0,2]-initial[2])
    blocked_width=float(latest["joints"][0,fi].sum())
    if height>=.03 and blocked_width>.001:
     report.setdefault("grasp_confirmations",[]).append(dict(body="screw" if prefix else "ring",height_m=height,width_m=blocked_width,attempt=attempt))
     return height
    report.setdefault("grasp_retries",[]).append(dict(body="screw" if prefix else "ring",height_m=height,width_m=blocked_width))
   raise RuntimeError("Object was not held after the lift: "+("screw" if prefix else "ring"))
  finally:
   if router is not None:router.teacher_override=override
 phase='settle';tick(physics_hz*2)
 rp=rv.get_transforms()[0].copy()
 ring_offset=np.array([0.,.023,.007])+(np.asarray(reset_spec["ring_grasp_offset_m"]) if reset_spec else 0)
 grasp=rp[:3]+Rotation.from_quat(rp[3:]).apply(ring_offset)
 ring_hand_rotation=reset_rotation(reset_spec["cable"])@Rdown if reset_spec else Rdown
 report['lift_height_m']=pick_body(rv,ring_offset,ring_hand_rotation,.08,.065)
 report['terminal_lift_observed']=report['lift_height_m']>=.03
 phase='level';body_move(rv,rv.get_transforms()[0,:3],1.5,board_rotation)
 via=(rp[:3]+target)/2;via[0]+=variation['waypoint_x_m'];via[1]+=variation['waypoint_y_m'];via[2]=variation['carry_height_m']
 if reset_spec is not None:via=route_waypoint(rp[:3],target,reset_spec,variation['carry_height_m'])
 phase='transport';body_move(rv,via,2.,board_rotation);body_move(rv,np.r_[target[:2],via[2]],2.,board_rotation)
 phase='seat';body_move(rv,target+[0,0,.0001],3.,board_rotation)
 if not ring_on_seat(records[-1],target,assembly_geometry):
  override=router.teacher_override if router is not None else False
  try:
   if router is not None:router.teacher_override=True
   body_move(rv,target+[0,0,.0001],3.,board_rotation)
  finally:
   if router is not None:router.teacher_override=override
 if not ring_on_seat(records[-1],target,assembly_geometry):raise RuntimeError("Terminal did not reach the seat before release")
 phase='release';grip(.06,.8)
 hp,hr=handpose();tcp=np.asarray(hp)+np.asarray(hr)@np.array([0,0,.1034])
 phase='retract';move(tcp+[0,0,.08],2.,.06,rotation=hr)
 phase='wire_verify';hold_feedback(2.)
 hold=[r for r in records if r['phase']=='wire_verify']
 report['wire_seated']=bool(hold and ring_on_seat(hold[-1],target,assembly_geometry))
 report['wire_final_pose']=rv.get_transforms()[0].tolist()
 print('WIRE',json.dumps({k:report[k] for k in ['wire_seated','lift_height_m','wire_final_pose']}),flush=True)
 if not report['wire_seated']:raise RuntimeError('terminal not stably seated')
 if args.probe_wire:raise RuntimeError('wire probe finished; full task not recorded')
 # Pick up the free screw from its nest with the fingers.
 sp=sv.get_transforms()[0].copy()
 screw_offset=np.array([0.,0.,.033])+(np.asarray(reset_spec["screw_grasp_offset_m"]) if reset_spec else 0)
 grasp=sp[:3]+Rotation.from_quat(sp[3:]).apply(screw_offset)
 screw_hand_rotation=Rotation.from_quat(sp[3:]).as_matrix()@Rdown if reset_spec else Rdown
 report['screw_lift_height_m']=pick_body(sv,screw_offset,screw_hand_rotation,.06,.08,prefix="screw_")
 phase='screw_transport';body_move(sv,np.r_[target[:2],.855],3.,board_rotation)
 phase='screw_align';body_move(sv,np.r_[target[:2],.832],2.,board_rotation)
 phase='screw_ring_entry';body_move(sv,np.r_[target[:2],.8165],3.,board_rotation,force_target=1.)
 print('ENTRY_RING',sv.get_transforms()[0].tolist(),flush=True)
 report['ring_entry_tip_z']=float(sv.get_transforms()[0,2])
 phase='thread_entry';body_move(sv,np.r_[target[:2],.812],3.,board_rotation,force_target=.5)
 print('ENTRY_THREAD',sv.get_transforms()[0].tolist(),flush=True)
 report['thread_entry_tip_z']=float(sv.get_transforms()[0,2])
 turns=[];report['tightening_checks']=[]
 report['thread_axial_control']='lead-based position reference; contact forces recorded'
 for turn in range(20):
  current=sv.get_transforms()[0].copy();rs=Rotation.from_quat(current[3:]).as_matrix()
  head_underside=current[:3]+rs@np.array([0.,0.,report['assembly_geometry']['screw_under_head_z_m']])
  if head_underside[2]<report['assembly_geometry']['nut_z_m'][0]:
   raise RuntimeError('Assembly incomplete: screw head fell below the nut; threading cannot continue.')
  phase='thread_turn'
  turn_angle=min(120.,max(15.,(current[2]-.8031)/.002*360.))
  rot=Rotation.from_euler('z',-turn_angle,degrees=True).as_matrix()@rs
  # Follow the screw lead during rotation; thread contact loads do not retract the axial reference.
  body_move(sv,np.r_[target[:2],current[2]-.002*turn_angle/360.],2.5,rot)
  actual=sv.get_transforms()[0].copy()
  turns.append({'turn':turn,'tip_z':float(actual[2]),'clamp_N':records[-1]['clamp_N'],'actual_rotation_deg':float(np.degrees((Rotation.from_quat(actual[3:])*Rotation.from_matrix(rs).inv()).as_rotvec()[2])),'advance_m':float(current[2]-actual[2])})
  print('TURN',json.dumps(turns[-1]),flush=True)
  opened=False
  if actual[2]<.804:
   phase='preload_release';grip(.040,.8);opened=True
   phase='preload_verify';hold_feedback(3.)
   window=records[-physics_hz*3:]
   check=assembly_hold(window,physics_hz,target)
   check.update(turn=turn,screw_tip_z=float(sv.get_transforms()[0,2]))
   report['tightening_checks'].append(check);print('PRELOAD',json.dumps(check),flush=True)
   if hold_passed(check):break
  if actual[2]>.814:report['thread_entry_retry_count']=report.get('thread_entry_retry_count',0)+1
  if not opened:
   phase='screw_regrasp_open';grip(.040,.5)
  actual=sv.get_transforms()[0].copy();head=actual[:3]+Rotation.from_quat(actual[3:]).apply([0,0,.033])
  phase='screw_regrasp_return';move(head+[0,0,.012],1.5,.040,rotation=Rdown)
  actual=sv.get_transforms()[0].copy();head=actual[:3]+Rotation.from_quat(actual[3:]).apply([0,0,.033])
  move(head,1.,.040,rotation=Rdown)
  phase='screw_regrasp_close';grip(0.,.75)
 report['thread_turns']=turns
 # Compare matching unloaded contact states across turns; loaded grasp compliance is excluded.
 relaxed=[]
 for k,row in enumerate(records):
  if row['phase']=='screw_regrasp_return' and (k+1==len(records) or records[k+1]['phase']!=row['phase']):
   relaxed.append([row['screw_rotation_accumulated_deg']/360.,row['screw_tip_z']])
 fit=np.asarray(relaxed)
 if len(fit)>=4 and np.ptp(fit[:,0])>1.:
  pitch,intercept=np.polyfit(fit[:,0],fit[:,1],1)
  residual=fit[:,1]-(pitch*fit[:,0]+intercept)
  report['measured_thread_lead_m']=float(abs(pitch))
  report['thread_lead_fit_residual_m']=float(np.max(np.abs(residual)))
 else:
  report['measured_thread_lead_m']=None;report['thread_lead_fit_residual_m']=None
 report['thread_engagement_verified']=bool(report['measured_thread_lead_m'] is not None and .0018<report['measured_thread_lead_m']<.0022 and report['thread_lead_fit_residual_m']<.0003)
 for final_attempt in range(3):
  phase='screw_release';grip(.06,1.)
  hp,hr=handpose();tcp=np.asarray(hp)+np.asarray(hr)@np.array([0,0,.1034])
  phase='screw_retract';move(tcp+[0,0,.09],2.,.06,rotation=hr)
  phase='full_verify';hold_feedback(3.)
  check=assembly_hold(records[-physics_hz*3:],physics_hz,target)
  report.update(check)
  report.setdefault('released_checks',[]).append(dict(check,attempt=final_attempt))
  if hold_passed(check):break
  if final_attempt==2 or not check['ring_retained'] or not check['nut_contact']:break
  override=router.teacher_override if router is not None else False
  try:
   if router is not None:router.teacher_override=True
   actual=sv.get_transforms()[0].copy();head=actual[:3]+Rotation.from_quat(actual[3:]).apply([0,0,.033])
   phase='screw_regrasp_return';move(head+[0,0,.012],1.5,.040,rotation=Rdown)
   actual=sv.get_transforms()[0].copy();head=actual[:3]+Rotation.from_quat(actual[3:]).apply([0,0,.033])
   move(head,1.,.040,rotation=Rdown)
   phase='screw_regrasp_close';grip(0.,.75)
   current=sv.get_transforms()[0].copy();rs=Rotation.from_quat(current[3:]).as_matrix()
   angle=30.;rot=Rotation.from_euler('z',-angle,degrees=True).as_matrix()@rs
   phase='thread_turn';body_move(sv,np.r_[target[:2],current[2]-.002*angle/360.],1.,rot)
   actual=sv.get_transforms()[0].copy()
   report.setdefault('final_tightening_corrections',[]).append(dict(angle_deg=angle,advance_m=float(current[2]-actual[2])))
  finally:
   if router is not None:router.teacher_override=override
 report['cable_connectivity_valid']=max(r['joint_gap_m'] for r in records)<.0005
 report['geometry_valid']=None
 report['collection_contract']='task2_state_feedback_collection_v003'
 report['full_task_completed']=bool(report['wire_seated'] and report['screw_fastened'] and report['ring_retained'] and report['head_seated'] and report.get('gripper_released',False) and report.get('thread_engagement_verified',False))
 active_records=[r for r in records if r['phase']!='settle']
 sample_hz=120 if args.record else 60
 target_q=np.array([r['robot_joint_targets'][:7] for r in active_records])[1::2] if args.record else np.array([r['robot_joint_targets'][:7] for r in active_records])
 velocity=np.diff(target_q,axis=0)*60
 acceleration=np.diff(velocity,axis=0)*60
 actual_v=np.array([r['robot_joint_velocities'][:7] for r in active_records])
 rapid_reversals=False
 for j in range(7):
  indices=np.flatnonzero(np.abs(actual_v[:,j])>.1)
  signs=np.sign(actual_v[indices,j])
  times=indices[1:][signs[1:]!=signs[:-1]]/sample_hz
  if len(times)>=3 and np.any(times[2:]-times[:-2]<.3):rapid_reversals=True
 report['smoothness']={'command_peak_velocity_rad_s':float(np.abs(velocity).max()),'command_peak_acceleration_rad_s2':float(np.abs(acceleration).max()),'rapid_joint_reversals':bool(rapid_reversals),'method':'60 Hz commanded joints; record three substantial velocity sign reversals within 0.3 s'}
 acc_row,acc_joint=np.unravel_index(np.abs(acceleration).argmax(),acceleration.shape)
 sampled=active_records[1::2] if args.record else active_records
 report['smoothness']['peak_acceleration_phase']=sampled[int(acc_row)+2]['phase']
 report['smoothness']['peak_acceleration_joint']=int(acc_joint)+1
 report['smoothness_pass']=report['smoothness']['command_peak_velocity_rad_s']<=2. and report['smoothness']['command_peak_acceleration_rad_s2']<=12. and not rapid_reversals
 report['diagnostic_checks']={k:bool(report[k]) for k in ['ring_pose_stable','cable_connectivity_valid','thread_engagement_verified','smoothness_pass']}
 report['acceptance_checks']={k:bool(report[k]) for k in ['wire_seated','screw_fastened','ring_retained','head_seated','gripper_released']}
 report['failed_checks']=[k for k,v in report['acceptance_checks'].items() if not v]
 report['accepted']=report['full_task_completed']
 if not report['accepted']:raise RuntimeError('released assembly incomplete')
 report['status']='SUCCESS'
except Exception as e:
 report['status']='FAILED';report['error']=str(e);report['traceback']=traceback.format_exc();print(report['traceback'],flush=True)
finally:
 report['measurements_path']=str(Path(globals().get('_diagnostics_root',ARTIFACTS/'attempt_results'))/('measurements_'+str(args.seed)+'.npz'))
 if router is not None:
  report["execution_counts"]=router.counts;report["expert_interventions"]=router.interventions
  report["teacher_events"]=getattr(router,"events",[])
  report["phase_execution_counts"]=getattr(router,"phase_counts",{})
  report["policy_executed_fraction"]=router.counts["policy"]/max(sum(router.counts.values()),1)
 if policy_client is not None:policy_client.close()
 report['motion_segments']=motion_segments
 report['phase_simulation_seconds']={name:sum(r['phase']==name for r in records)/args.physics_hz for name in dict.fromkeys(r['phase'] for r in records)}
 if records:
  report['max_contact_intrusion_m']=None
  report['max_joint_gap_m']=max(r['joint_gap_m'] for r in records)
 if report.get('accepted',False) and args.record:
  sys.path.insert(0,str(COLLECT))
  from interface import save_episode
  report['performance']=performance_snapshot()
  save_episode(out,records,frames,image_frames,report,setup_render.manifest)
 print('PERFORMANCE',json.dumps(performance_snapshot()),flush=True)
 report['performance']=performance_snapshot()
 if args.record and not report.get('accepted',False) and image_frames and collection_config.get('save_failed_video',not bool(collection_config)):
  sys.path.insert(0,str(COLLECT))
  from review_video import ReviewVideo
  failed_video=ReviewVideo(ARTIFACTS/'failure_review'/('collected_seed_'+str(args.seed).zfill(4)),f'Task2 ORIGINAL failed attempt | seed {args.seed} | 128 px camera observations enlarged')
  for frame,images in zip(frames,image_frames):
   failed_video.frame(images,records[round(frame['time_s']*physics_hz)-1])
  report['recording_kind']='original collected failed attempt'
  report=failed_video.finish(report)
 diagnostics=Path(globals().get('_diagnostics_root',ARTIFACTS/'attempt_results'))
 diagnostics.mkdir(parents=True,exist_ok=True)
 if records:
  fields=["time_s","screw_axial_N","screw_lateral_N","screw_torque_Nm","clamp_N","joint_gap_m","ring_x","ring_y","ring_z","ring_speed_m_s","ring_tilt_deg","screw_tip_z","screw_tilt_deg","screw_rotation_accumulated_deg","head_contact_count","nut_contact_count"]
  measurements={key:np.asarray([row[key] for row in records]) for key in fields}
  measurements["phase"]=np.asarray([row["phase"] for row in records])
  np.savez_compressed(diagnostics/("measurements_"+str(args.seed)+".npz"),**measurements)
 diagnostic={k:v for k,v in report.items() if k not in ['traceback','thread_turns']}
 diagnostic['failure_phase']=phase if report.get('status')=='FAILED' else None
 (diagnostics/('collection_'+str(args.seed)+'.json')).write_text(json.dumps(diagnostic,indent=2))
 print('RESULT',json.dumps(report),flush=True)
 if app is not None and shared_app is None:app.close()
