"""Synchronized three-camera recordings of contact-driven assembly attempts."""
from pathlib import Path
import argparse,sys,json,csv,traceback,time
import numpy as np
from scipy.spatial.transform import Rotation
R=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io')
pa=argparse.ArgumentParser()
pa.add_argument('--seed',type=int,default=0);pa.add_argument('--record',action='store_true');pa.add_argument('--probe-wire',action='store_true')
pa.add_argument('--physics-hz',type=int,choices=[120,240,480],default=120)
args=pa.parse_args()
out=R/'scene/success_v014'/('episode_'+str(args.seed).zfill(4))
if out.exists():raise RuntimeError('episode already exists')
import io
records=[];frames=[];events=[];image_frames=[];writer=None;app=None;phase='initializing';stepnum=0
rng=np.random.default_rng(args.seed)
variation={'carry_height_m':float(rng.uniform(.865,.885)),'waypoint_x_m':float(rng.uniform(-.008,.008)),'waypoint_y_m':float(rng.uniform(-.008,.008)),'duration_scale':float(rng.uniform(.95,1.15))}
report={'scene':'riser_io_success_v014.usda','full_task_completed':False,'safety_verdict':'UNASSESSED','ring_geometry_unchanged':True,'task_body_pose_writes_after_reset':False,'seed':args.seed,'variation':variation}
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
def performance_snapshot():
 return {'wall_seconds':time.perf_counter()-wall_started,'simulation_seconds':stepnum/args.physics_hz,'components':performance_totals}


try:
 sys.argv=[sys.argv[0]]
 from isaacsim import SimulationApp
 app=SimulationApp({'headless':True,'limit_cpu_threads':4,'width':960,'height':720,'sync_loads':True,'extra_args':['--/log/file='+str(R/'scene/success_v014/current_kit.log')]})
 import omni.usd,omni.physics.tensors as tensors,torch
 from pxr import UsdGeom,UsdPhysics,UsdLux,Gf
 from isaacsim.core.api import SimulationContext
 from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver,interface_config_loader
 omni.usd.get_context().open_stage(str(R/'scene/riser_io_success_v014.usda'))
 for _ in range(8):app.update()
 stage=omni.usd.get_context().get_stage();stage.SetEditTarget(stage.GetSessionLayer())
 UsdLux.DomeLight.Define(stage,'/World/DemoLight').CreateIntensityAttr(1500.)
 from pxr import PhysxSchema
 for p in stage.Traverse():
  if p.HasAPI(UsdPhysics.RigidBodyAPI):
   PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr(0.)
 for p in stage.Traverse():
  if p.IsA(UsdPhysics.Scene):
   physics=PhysxSchema.PhysxSceneAPI.Apply(p)
   physics.CreateEnableGPUDynamicsAttr(True)
   physics.CreateBroadphaseTypeAttr('GPU')
 report['physics_backend']='PhysX GPU'
 for path in ['/World/Robot','/World/CableSensor']:
  PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(path)).CreateSolverPositionIterationCountAttr(64)
 report['solver_position_iterations']=64
 physics_hz=args.physics_hz
 report['physics_rate_Hz']=physics_hz;report['measurement_rate_Hz']=120 if args.record else 60
 dt=1/physics_hz
 sim=SimulationContext(physics_dt=dt,rendering_dt=1/60,backend='torch',device='cuda:0');sim.initialize_physics();sim.play()
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
      return torch.as_tensor(x,device='cpu' if name in ['set_dof_max_forces','set_dof_stiffnesses','set_dof_dampings'] else 'cuda:0').contiguous()
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
   r=least_squares(local_residual,np.clip(seed,*bounds),bounds=bounds,max_nfev=60,ftol=1e-10,xtol=1e-10,gtol=1e-10)
   err=residual(r.x)
   if np.linalg.norm(err[:3])<.0002 and np.linalg.norm(err[3:])<.0002:return r.x
   raise RuntimeError('continuous IK could not track pose '+str(err))
  best=None
  for start in [seed,[0,-.8,0,-2.6,0,1.8,.7854],[.5,.2,-.5,-2.,0,2.3,0],[-.5,.2,.5,-2.,0,2.3,1.5]]:
   r=least_squares(residual,np.clip(start,lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=160,ftol=1e-10,xtol=1e-10,gtol=1e-10)
   if best is None or np.linalg.norm(r.fun)<np.linalg.norm(best.fun):best=r
   if np.linalg.norm(r.fun[:3])<.005 and np.linalg.norm(r.fun[3:])<.01:return r.x
  raise RuntimeError('IK unreachable: '+str(p)+' residual '+str(best.fun))
 Rdown=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]])
 q=robot.get_dof_positions()[0].copy();q[ai]=[0,-.5,0,-2.2,0,1.7,.7854];q[fi]=.04
 q[ai]=solve(np.array([-.285,.02,1.0434]),Rotation.from_matrix(Rdown).as_quat(),q[ai])
 robot.set_dof_positions(q[None].astype(np.float32),ix);robot.set_dof_velocities(np.zeros_like(q)[None].astype(np.float32),ix);robot.set_dof_position_targets(q[None].astype(np.float32),ix)
 forces=robot.get_dof_max_forces().copy();forces[0,fi]=15.;robot.set_dof_max_forces(forces,ix)
 stiffness=robot.get_dof_stiffnesses().copy();stiffness[0,fi]=400.;robot.set_dof_stiffnesses(stiffness,ix)
 damping=robot.get_dof_dampings().copy();damping[0,fi]=15.;robot.set_dof_dampings(damping,ix)
 report['finger_control']={'stiffness_N_m':400.,'damping_Ns_m':15.,'effort_cap_N':15.}
 filters=[]
 for p in stage.Traverse():
  if not p.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get():continue
  own=p
  while own and not own.HasAPI(UsdPhysics.RigidBodyAPI):own=own.GetParent()
  path=str(own.GetPath()) if own else str(p.GetPath())
  if path not in filters:filters.append(path)
 for n in robot.shared_metatype.link_names:
  path='/World/Robot/'+n
  if path not in filters:filters.append(path)
 contacts=HostObservations(view.create_rigid_contact_view(['/wire1/Segment_*','/wire1/_0347300_ring_terminal','/ScrewTray/Nest_1/Screw_1'],[filters]*3,4096))
 grips=HostObservations(view.create_rigid_contact_view('/wire1/_0347300_ring_terminal',['/World/Robot/panda_leftfinger','/World/Robot/panda_rightfinger'],4096))
 jp0=np.array([stage.GetPrimAtPath(f'/wire1/Bend_{i:02d}').GetAttribute('physics:localPos0').Get() for i in range(23)])
 jp1=np.array([stage.GetPrimAtPath(f'/wire1/Bend_{i:02d}').GetAttribute('physics:localPos1').Get() for i in range(23)])
 report['contact_filters']=filters;latched=False
 fixtures=[p for p in filters if not any(x in p for x in ['/wire1','/World/Robot','CableSensor'])]
 robot_paths=['/World/Robot/'+n for n in robot.shared_metatype.link_names]
 robot_contacts=HostObservations(view.create_rigid_contact_view(robot_paths,[fixtures]*len(robot_paths),2048))
 report['robot_contact_sources']=robot_paths

 screw_contacts=HostObservations(view.create_rigid_contact_view('/ScrewTray/Nest_1/Screw_1',['/NutPlate_Coarse/Nut_1/IndependentNut','/wire1/_0347300_ring_terminal','/PCB_4_Terminal/SingleSeat'],2048))
 step_physics=measured('physics_step')(sim.step)
 latest={}
 @measured("state_contact_readback")
 def capture_states():
  normal_data=screw_contacts.native.get_contact_data(dt)
  # PhysX reuses the pair index buffers for friction data on the same contact view.
  normal_data=tuple(normal_data[:-2])+tuple(x.clone() for x in normal_data[-2:])
  requests={
   'segments':seg.native.get_transforms(),
   'ring':rv.native.get_transforms(),
   'screw':sv.native.get_transforms(),
   'ring_velocity':rv.native.get_velocities(),
   'joints':robot.native.get_dof_positions(),
   'joint_velocities':robot.native.get_dof_velocities(),
   'wrench':sensor.native.get_link_incoming_joint_force(),
   'contacts':contacts.native.get_contact_data(dt),
   'robot_contacts':robot_contacts.native.get_contact_data(dt),
   'grip_matrix':grips.native.get_contact_force_matrix(dt),
   'screw_contacts':normal_data,
   'screw_friction':screw_contacts.native.get_friction_data(dt),
   'screw_matrix':screw_contacts.native.get_contact_force_matrix(dt)}
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
   if stepnum%(physics_hz//(120 if args.record else 60)):continue
   capture_states()
   analysis_started=time.perf_counter()
   tr=latest['segments'].copy();xyz=tr[:,:3];ring=latest['ring'][0].copy()
   w=latest['wrench'][0].copy()
   gaps=np.linalg.norm(xyz[:-1]+Rotation.from_quat(tr[:-1,3:]).apply(jp0)-xyz[1:]-Rotation.from_quat(tr[1:,3:]).apply(jp1),axis=1)
   a=xyz[1:-1]-xyz[:-2];b=xyz[2:]-xyz[1:-1];c=xyz[2:]-xyz[:-2]
   rad=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)*np.linalg.norm(c,axis=1)/np.maximum(2*np.linalg.norm(np.cross(a,b),axis=1),1e-12)
   f,p,no,dep,cnt,st=latest['contacts']
   minimum=0.;deepest='';num=int(cnt.sum())
   if num>=4096:raise RuntimeError('contact buffer saturated')
   for k in np.flatnonzero(cnt):
    nn=cnt.ravel()[k];ss=st.ravel()[k]
    if nn:
     d=float(dep[int(ss):int(ss+nn)].min())
     if d<minimum:minimum=d;deepest=str(contacts.sensor_paths[k//contacts.filter_count])+'|'+filters[k%contacts.filter_count]
   rf,rp,rn,rd,rc,rs=latest['robot_contacts']
   robot_load=0.;robot_pair=''
   for kk in np.flatnonzero(rc):
    nn=rc.ravel()[kk];ss=rs.ravel()[kk]
    if nn:
     load=float(np.abs(rf[int(ss):int(ss+nn)]).sum());robot_load+=load
     if load>0:robot_pair=str(robot_contacts.sensor_paths[kk//len(fixtures)])+'|'+fixtures[kk%len(fixtures)]
     d=float(rd[int(ss):int(ss+nn)].min())
     if d<minimum:minimum=d;deepest=str(robot_contacts.sensor_paths[kk//len(fixtures)])+'|'+fixtures[kk%len(fixtures)]
   gf=latest['grip_matrix'].reshape(1,2,3)
   root=float(np.linalg.norm(w[ci,:3]));cable=float(np.linalg.norm(w[si,:3]))
   edge_load=0.;seat_load=0.
   for kk in np.flatnonzero(cnt):
    nn=cnt.ravel()[kk];ss=st.ravel()[kk]
    path=filters[kk%contacts.filter_count]
    if nn and ('Chassis/Panel' in path or 'SingleSeat' in path):
     mag=float(np.abs(f[int(ss):int(ss+nn)]).sum())
     if 'Chassis/Panel' in path:edge_load+=mag
     if 'SingleSeat' in path:seat_load+=mag

   sp,sax,slat,storque=screw_measure()
   radius=float(rad.min());active=phase not in ['settle','calibration','initializing']
   exceeds=False
   if active and exceeds:latched=True
   rec={'time_s':stepnum*dt,'phase':phase,'screw_axial_N':sax,'screw_lateral_N':slat,'screw_torque_Nm':storque,'screw_tip_z':float(sp[2]),'screw_tilt_deg':float(np.degrees(np.arccos(np.clip(Rotation.from_quat(sp[3:]).apply([0,0,1])[2],-1,1)))),'edge_normal_load_N':edge_load,'seat_normal_load_N':seat_load,'root_axial_N':float(abs(w[ci,1])),'root_lateral_N':float(np.linalg.norm(w[ci,[0,2]])),'root_force_N':root,'cable_interface_force_N':cable,'root_moment_Nm':float(np.linalg.norm(w[ci,3:])),'min_radius_m':radius,'tightest_segment':int(rad.argmin()+1),'joint_gap_m':float(gaps.max()),'contact_separation_m':minimum,'deepest_pair':deepest,'contact_points':num,'left_grip_N':float(np.linalg.norm(gf[0,0])),'right_grip_N':float(np.linalg.norm(gf[0,1])),'ring_x':float(ring[0]),'ring_y':float(ring[1]),'ring_z':float(ring[2]),'ring_speed_m_s':float(np.linalg.norm(latest['ring_velocity'][0,:3])),'robot_fixture_and_screw_normal_sum_N':robot_load,'robot_fixture_or_screw_pair':robot_pair}
   rec['screw_x_m']=float(sp[0]);rec['screw_y_m']=float(sp[1])
   sq=Rotation.from_quat(sp[3:])
   if not hasattr(tick,'last_screw_rotation'):
    tick.last_screw_rotation=sq;tick.screw_angle=0.
   tick.screw_angle+=float(np.dot((sq*tick.last_screw_rotation.inv()).as_rotvec(),sq.apply([0,0,1])))
   tick.last_screw_rotation=sq
   rec['screw_rotation_accumulated_deg']=float(np.degrees(tick.screw_angle))
   rec['robot_joint_positions']=latest['joints'][0].tolist()
   rec['robot_joint_velocities']=latest['joint_velocities'][0].tolist()
   rec['robot_joint_targets']=q.tolist()
   rec['root_wrench']=w[ci].tolist()
   rec['cable_interface_wrench']=w[si].tolist()
   mat=latest['screw_matrix'].reshape(3,3)
   rec['ring_contact_axial_N']=float(abs(mat[1,2]));rec['screw_nut_N']=float(np.linalg.norm(mat[0]))
   cf,cp,cn,cd,cc,cs=latest['screw_contacts']
   normal_sums=[]
   for nn,ss in zip(cc.ravel(),cs.ravel()):
    part=slice(int(ss),int(ss+nn));normal_sums.append((cf[part]*cn[part]).sum(axis=0))
   mismatch=float(np.max(np.abs(np.asarray(normal_sums)-mat)))
   if mismatch>.001:
    report['contact_readback_mismatch_N']=mismatch
    raise RuntimeError('detailed normal contacts do not match the pair force matrix')
   start=int(cs.ravel()[1]);count=int(cc.ravel()[1]);sl=slice(start,start+count)
   local=sq.inv().apply(cp[sl]-sp[:3])
   under_head=(np.linalg.norm(local[:,:2],axis=1)>.0055)&(np.abs(local[:,2]-.018)<.0005)
   head_force=(cf[sl]*cn[sl])[under_head].sum(axis=0)
   rec['clamp_N']=float(abs(np.dot(head_force,sq.apply([0,0,1]))))
   rec['ring_tilt_deg']=float(np.degrees(np.arccos(np.clip(Rotation.from_quat(ring[3:]).as_matrix()[2,2],-1,1))))
   records.append(rec)
   if stepnum%(physics_hz//60)==0:
    commanded=q[ai].copy()
    if hasattr(tick,'previous_command'):
     command_v=(commanded-tick.previous_command)*60.
     if hasattr(tick,'previous_command_v') and active:
      command_a=(command_v-tick.previous_command_v)*60.
      if np.max(np.abs(command_v))>2. or np.max(np.abs(command_a))>12.:
       report['smoothness_failure']={'phase':phase,'time_s':stepnum*dt,'peak_velocity_rad_s':float(np.max(np.abs(command_v))),'peak_acceleration_rad_s2':float(np.max(np.abs(command_a))),'joint':int(np.argmax(np.abs(command_a)))+1}
       raise RuntimeError('command smoothness limit exceeded')
     tick.previous_command_v=command_v
    tick.previous_command=commanded
   if not np.isfinite(xyz).all() or not np.isfinite(w).all():raise RuntimeError('nonfinite state')
   if root>100. or float(gaps.max())>.010:raise RuntimeError('structural instability')
   entry=performance_totals.setdefault('state_analysis',{'seconds':0.,'calls':0})
   entry['seconds']+=time.perf_counter()-analysis_started;entry['calls']+=1
   if args.record and stepnum%(physics_hz//10)==0:write_frame(rec,tr,ring)
   if stepnum%(physics_hz*5)==0:
    print('PERFORMANCE',json.dumps(performance_snapshot()),flush=True)
   if stepnum%physics_hz==0:
    print('STATE',json.dumps({k:rec[k] for k in ['time_s','phase','ring_x','ring_y','ring_z','ring_tilt_deg','screw_tip_z','screw_axial_N','clamp_N','screw_torque_Nm','robot_fixture_and_screw_normal_sum_N','robot_fixture_or_screw_pair']}),flush=True)
 def setup_render():
  import omni.replicator.core as rep
  if hasattr(setup_render,'anns'):return
  external=UsdGeom.Camera.Define(stage,'/World/ExternalCamera')
  external.CreateFocalLengthAttr(23.)
  external.CreateClippingRangeAttr(Gf.Vec2f(.005,10.))
  mat=Gf.Matrix4d().SetLookAt(Gf.Vec3d(.13,-.57,1.35),Gf.Vec3d(-.15,.055,.86),Gf.Vec3d(0,0,1)).GetInverse()
  UsdGeom.Xformable(external).AddTransformOp().Set(mat)
  paths={'external':'/World/ExternalCamera','wrist_left':'/World/Robot/panda_hand/WristCam_L','wrist_right':'/World/Robot/panda_hand/WristCam_R'}
  setup_render.anns={};setup_render.paths=paths;manifest={}
  for name,path in paths.items():
   p=stage.GetPrimAtPath(path)
   if not p or p.GetTypeName()!='Camera':raise RuntimeError('missing camera '+path)
   product=rep.create.render_product(path,(640,480))
   ann=rep.AnnotatorRegistry.get_annotator('rgb');ann.attach([product]);setup_render.anns[name]=ann
   c=UsdGeom.Camera(p)
   manifest[name]={'path':path,'resolution':[640,480],'focal_length':c.GetFocalLengthAttr().Get(),'horizontal_aperture':c.GetHorizontalApertureAttr().Get(),'vertical_aperture':c.GetVerticalApertureAttr().Get(),'local_matrix':np.asarray(UsdGeom.Xformable(p).GetLocalTransformation()).tolist()}
  manifest['external_configuration']='fixed collection camera; task2 has no saved external camera calibration'
  manifest['timestamp']='simulation time; all cameras rendered without intervening physics steps'
  setup_render.manifest=manifest
  for attempt in range(60):
   sim.render()
   if all(a.get_data() is not None and a.get_data().size for a in setup_render.anns.values()):break
  else:raise RuntimeError('three-camera render initialization failed')
 @measured("camera_render_readback_jpeg")
 def write_frame(rec,tr,ring):
  from PIL import Image
  setup_render();sim.render();images={}
  for name,ann in setup_render.anns.items():
   rgb=ann.get_data()
   if rgb is None or not rgb.size:raise RuntimeError('empty camera '+name)
   buf=io.BytesIO();Image.fromarray(rgb[:,:,:3]).save(buf,format='JPEG',quality=85);images[name]=buf.getvalue()
  image_frames.append(images)
  frames.append({'frame':len(frames),'time_s':rec['time_s'],'phase':phase,'ring_pose':ring.tolist(),'screw_pose':latest['screw'][0].tolist(),'robot_joints':latest['joints'][0].tolist(),'cable_poses':tr.tolist()})
 def event(name):
  row={'event':name,'time_s':stepnum*dt,'ring_pose':rv.get_transforms()[0].tolist()};events.append(row);print('EVENT',json.dumps(row),flush=True)
 def move(tcp,seconds=3.,width=None,rotation=None,stop_force=None):
  global q
  desired=Rdown if rotation is None else rotation
  start,rot=solver.compute_forward_kinematics('panda_hand',q[ai]);end=np.asarray(tcp)-desired@np.array([0,0,.1034])
  r0=Rotation.from_matrix(rot);r1=Rotation.from_matrix(desired);dr=(r1*r0.inv()).as_rotvec()
  steps=max(int(seconds*variation['duration_scale']*60),int(np.ceil(1.875*np.linalg.norm(dr)/1.2*60)));seed=q[ai].copy()
  for k in range(steps):
   t=(k+1)/steps;u=t*t*t*(10+t*(-15+6*t))
   p=start+(end-start)*u;rr=(Rotation.from_rotvec(dr*u)*r0).as_quat()
   nxt=solve(p,rr,seed)
   if np.max(np.abs(nxt-seed))>.08:raise RuntimeError('discontinuous IK target')
   seed=nxt;q[ai]=seed
   if width is not None:q[fi]=width/2
   robot.set_dof_position_targets(q[None].astype(np.float32),ix);tick(physics_hz//60)
  tick(physics_hz//4)
  hp,_=handpose()
  if np.linalg.norm(np.asarray(hp)-end)>.020:raise RuntimeError('hand tracking error')
 def body_move(body,goal,seconds=2.,rotation=None,force_target=None):
  global q
  initial=body.get_transforms()[0].copy();r0=Rotation.from_quat(initial[3:])
  hp,hr=solver.compute_forward_kinematics('panda_hand',q[ai]);grasp_offset=r0.inv().apply(np.asarray(hp)-initial[:3]);grasp_rotation=r0.inv().as_matrix()@hr
  r1=r0 if rotation is None else Rotation.from_matrix(rotation);dr=(r1*r0.inv()).as_rotvec()
  bias=np.zeros(3);angle_bias=np.zeros(3);admittance=0.
  steps=max(int(seconds*variation['duration_scale']*60),int(np.ceil(1.875*np.linalg.norm(dr)/1.2*60)));seed=q[ai].copy()
  for k in range(steps+30):
   t=min(1.,(k+1)/steps);u=t*t*t*(10+t*(-15+6*t))
   desired=initial[:3]+(np.asarray(goal)-initial[:3])*u
   if force_target is not None:
    admittance=float(np.clip(admittance+.000001*(records[-1]['screw_axial_N']-force_target),-.001,.002))
    desired[2]+=admittance
   drt=Rotation.from_rotvec(dr*u)*r0
   actual=(latest['ring'] if body is rv else latest['screw'])[0].copy();ar=Rotation.from_quat(actual[3:])
   error=desired-actual[:3];bias=np.clip(bias+.025*error,-.002,.002)
   if force_target is not None:bias[2]=0.
   angle_bias=np.clip(angle_bias+.015*(drt*ar.inv()).as_rotvec(),-.03,.03)
   rd=(Rotation.from_rotvec(angle_bias)*drt).as_matrix()
   hp=desired+bias+rd@grasp_offset;hr=rd@grasp_rotation
   nxt=solve(hp,Rotation.from_matrix(hr).as_quat(),seed)
   if np.max(np.abs(nxt-seed))>.08:raise RuntimeError('discontinuous IK body target')
   seed=nxt;q[ai]=seed;robot.set_dof_position_targets(q[None].astype(np.float32),ix);tick(physics_hz//60)
 def grip(width,seconds=1.):
  global q
  start=q[fi].copy();steps=int(seconds*60)
  for k in range(steps):
   q[fi]=start+(width/2-start)*(((k+1)/steps)**3*(10+((k+1)/steps)*(-15+6*((k+1)/steps))));robot.set_dof_position_targets(q[None].astype(np.float32),ix);tick(physics_hz//60)
 phase='settle';tick(physics_hz*2)
 target=np.array([-.092,-.071,.818])
 rp=rv.get_transforms()[0].copy();grasp=rp[:3]+Rotation.from_quat(rp[3:]).apply([0,.023,.007])
 phase='approach';move(grasp+[0,0,.09],2.,.08)
 phase='descend';move(grasp,2.,.08)
 phase='grasp';grip(0.,1.)
 phase='lift';move(grasp+[0,0,.065],2.5)
 report['lift_height_m']=float(rv.get_transforms()[0,2]-rp[2])
 if report['lift_height_m']<.045:raise RuntimeError('terminal grasp failed')
 phase='level';body_move(rv,rv.get_transforms()[0,:3],1.5,np.eye(3))
 via=(rp[:3]+target)/2;via[0]+=variation['waypoint_x_m'];via[1]+=variation['waypoint_y_m'];via[2]=variation['carry_height_m']
 phase='transport';body_move(rv,via,2.,np.eye(3));body_move(rv,np.r_[target[:2],variation['carry_height_m']],2.,np.eye(3))
 phase='seat';body_move(rv,target+[0,0,.0001],3.,np.eye(3))
 phase='release';grip(.06,.8)
 hp,hr=handpose();tcp=np.asarray(hp)+np.asarray(hr)@np.array([0,0,.1034])
 phase='retract';move(tcp+[0,0,.08],2.,.06,rotation=hr)
 phase='wire_verify';tick(physics_hz*2)
 hold=[r for r in records if r['phase']=='wire_verify']
 report['wire_seated']=all(np.linalg.norm(np.array([r['ring_x'],r['ring_y']])-target[:2])<.0015 and abs(r['ring_z']-target[2])<.0007 and r['ring_tilt_deg']<5 and r['ring_speed_m_s']<.01 for r in hold)
 report['wire_final_pose']=rv.get_transforms()[0].tolist()
 print('WIRE',json.dumps({k:report[k] for k in ['wire_seated','lift_height_m','wire_final_pose']}),flush=True)
 if not report['wire_seated']:raise RuntimeError('terminal not stably seated')
 if args.probe_wire:raise RuntimeError('wire probe finished; full task not recorded')
 # Pick up the free screw from its nest with the fingers.
 sp=sv.get_transforms()[0].copy();grasp=sp[:3]+Rotation.from_quat(sp[3:]).apply([0,0,.033])
 phase='screw_approach';move(grasp+[0,0,.09],2.,.06)
 phase='screw_descend';move(grasp,2.,.06)
 phase='screw_grasp';grip(0.,1.)
 phase='screw_lift';move(grasp+[0,0,.08],2.)
 if sv.get_transforms()[0,2]<sp[2]+.05:raise RuntimeError('screw grasp failed')
 phase='screw_transport';body_move(sv,np.r_[target[:2],.855],3.,np.eye(3))
 phase='screw_align';body_move(sv,np.r_[target[:2],.832],2.,np.eye(3))
 phase='screw_ring_entry';body_move(sv,np.r_[target[:2],.8165],3.,np.eye(3),force_target=1.)
 print('ENTRY_RING',sv.get_transforms()[0].tolist(),flush=True)
 if sv.get_transforms()[0,2]>.819:raise RuntimeError('blocked at ring bore')
 phase='thread_entry';body_move(sv,np.r_[target[:2],.812],3.,np.eye(3),force_target=.5)
 print('ENTRY_THREAD',sv.get_transforms()[0].tolist(),flush=True)
 if sv.get_transforms()[0,2]>.814:raise RuntimeError('blocked above thread entrance')
 turns=[];report['tightening_checks']=[]
 for turn in range(20):
  current=sv.get_transforms()[0].copy();rs=Rotation.from_quat(current[3:]).as_matrix()
  phase='thread_turn'
  turn_angle=min(120.,max(15.,(current[2]-.8031)/.002*360.))
  rot=Rotation.from_euler('z',-turn_angle,degrees=True).as_matrix()@rs
  body_move(sv,np.r_[target[:2],current[2]-.002*turn_angle/360.],2.5,rot,force_target=.5)
  actual=sv.get_transforms()[0].copy()
  turns.append({'turn':turn,'tip_z':float(actual[2]),'clamp_N':records[-1]['clamp_N'],'actual_rotation_deg':float(np.degrees((Rotation.from_quat(actual[3:])*Rotation.from_matrix(rs).inv()).as_rotvec()[2])),'advance_m':float(current[2]-actual[2])})
  print('TURN',json.dumps(turns[-1]),flush=True)
  opened=False
  if actual[2]<.804:
   phase='preload_release';grip(.040,.8);opened=True
   phase='preload_verify';tick(physics_hz)
   window=[r for r in records if r['phase']=='preload_verify'][-(120 if args.record else 60):]
   retained=float(np.mean([r['clamp_N'] for r in window]))
   check={'turn':turn,'retained_head_clamp_N':retained,'screw_tip_z':float(sv.get_transforms()[0,2])}
   report['tightening_checks'].append(check);print('PRELOAD',json.dumps(check),flush=True)
   if retained>.2 and all(r['ring_tilt_deg']<2. for r in window):break
  if actual[2]>.814:raise RuntimeError('thread engagement lost')
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
 phase='screw_release';grip(.06,1.)
 hp,hr=handpose();tcp=np.asarray(hp)+np.asarray(hr)@np.array([0,0,.1034])
 phase='screw_retract';move(tcp+[0,0,.09],2.,.06,rotation=hr)
 phase='full_verify';tick(physics_hz*3)
 hold=[r for r in records if r['phase']=='full_verify']
 report['screw_fastened']=all(abs(r['screw_tip_z']-.803)<.0007 and np.linalg.norm(np.array([r['screw_x_m'],r['screw_y_m']])-target[:2])<.001 and r['screw_tilt_deg']<5 for r in hold)
 report['ring_retained']=all(np.linalg.norm(np.array([r['ring_x'],r['ring_y']])-target[:2])<.0015 and abs(r['ring_z']-.818)<.0007 and r['ring_tilt_deg']<5 for r in hold)
 report['ring_hold_extrema']={'max_xy_error_m':float(max(np.linalg.norm(np.array([r['ring_x'],r['ring_y']])-target[:2]) for r in hold)),'max_z_error_m':float(max(abs(r['ring_z']-.818) for r in hold)),'max_tilt_deg':float(max(r['ring_tilt_deg'] for r in hold)),'max_native_speed_m_s':float(max(r['ring_speed_m_s'] for r in hold))}
 hold_positions=np.array([[r['ring_x'],r['ring_y'],r['ring_z']] for r in hold])
 report['ring_hold_extrema']['position_span_m']=np.ptp(hold_positions,axis=0).tolist()
 report['ring_hold_extrema']['max_pose_speed_m_s']=float(np.max(np.linalg.norm(np.diff(hold_positions,axis=0),axis=1))*(120 if args.record else 60))

 report['ring_pose_stable']=bool(report['ring_hold_extrema']['max_pose_speed_m_s']<.01 and np.linalg.norm(np.ptp(hold_positions,axis=0))<.0002)
 report['ring_retained']=bool(report['ring_retained'] and report['ring_pose_stable'])
 report['mean_clamp_N']=float(np.mean([r['clamp_N'] for r in hold]))
 report['geometry_valid']=max(r['joint_gap_m'] for r in records)<.0005 and min(r['contact_separation_m'] for r in records)>-.0005
 report['full_task_completed']=bool(report['wire_seated'] and report['screw_fastened'] and report['ring_retained'] and report['mean_clamp_N']>.2 and report['geometry_valid'] and report['thread_engagement_verified'])
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
 report['smoothness']={'command_peak_velocity_rad_s':float(np.abs(velocity).max()),'command_peak_acceleration_rad_s2':float(np.abs(acceleration).max()),'rapid_joint_reversals':bool(rapid_reversals),'method':'60 Hz commanded joints; reject three substantial velocity sign reversals within 0.3 s'}
 acc_row,acc_joint=np.unravel_index(np.abs(acceleration).argmax(),acceleration.shape)
 sampled=active_records[1::2] if args.record else active_records
 report['smoothness']['peak_acceleration_phase']=sampled[int(acc_row)+2]['phase']
 report['smoothness']['peak_acceleration_joint']=int(acc_joint)+1
 report['smoothness_pass']=report['smoothness']['command_peak_velocity_rad_s']<=2. and report['smoothness']['command_peak_acceleration_rad_s2']<=12. and not rapid_reversals
 report['accepted']=report['full_task_completed'] and report['smoothness_pass']
 if not report['accepted']:raise RuntimeError('full assembly acceptance failed' if not report['full_task_completed'] else 'trajectory smoothness acceptance failed')
 report['status']='SUCCESS'
except Exception as e:
 report['status']='FAILED';report['error']=str(e);report['traceback']=traceback.format_exc();print(report['traceback'],flush=True)
finally:
 if records:
  report['max_contact_intrusion_m']=-min(r['contact_separation_m'] for r in records)
  report['max_joint_gap_m']=max(r['joint_gap_m'] for r in records)
 if report.get('accepted',False) and args.record:
  from PIL import Image
  sys.path.insert(0,str(R/'scene/single_wire_fastening_v003/video_dependencies'))
  import imageio_ffmpeg
  out.mkdir(parents=True,exist_ok=False)
  encoding_started=time.perf_counter()
  for name in ['external','wrist_left','wrist_right']:
   w=imageio_ffmpeg.write_frames(str(out/(name+'.mp4')),(640,480),fps=10.,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',output_params=['-preset','fast','-crf','20','-movflags','+faststart','-threads','2'])
   w.send(None)
   for images in image_frames:
    im=np.asarray(Image.open(io.BytesIO(images[name])).convert('RGB'))
    w.send(im)
   w.close()
  performance_totals['video_encoding']={'seconds':time.perf_counter()-encoding_started,'calls':1}
  report['performance']=performance_snapshot()
  (out/'frames.json').write_text(json.dumps(frames))
  (out/'cameras.json').write_text(json.dumps(setup_render.manifest,indent=2))
  (out/'events.json').write_text(json.dumps(events))
  with (out/'metrics.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
  (out/'result.json').write_text(json.dumps(report,indent=2))
 print('PERFORMANCE',json.dumps(performance_snapshot()),flush=True)
 print('RESULT',json.dumps(report),flush=True)
 if app is not None:app.close()
