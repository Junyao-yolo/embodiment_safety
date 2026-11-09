"""Synchronized three-camera recordings of contact-driven assembly attempts."""
from pathlib import Path
import argparse,sys,json,csv,traceback,time
import numpy as np
from scipy.spatial.transform import Rotation
R=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io')
pa=argparse.ArgumentParser();pa.add_argument('--name',default='trial_001');pa.add_argument('--settle-only',action='store_true');pa.add_argument('--record',action='store_true');pa.add_argument('--bend-angle',type=float,default=125.);pa.add_argument('--case',default='reference');pa.add_argument('--resume',action='store_true');args=pa.parse_args()
out=R/'scene/three_camera_capture_v013'/args.name;out.mkdir(parents=True,exist_ok=True)
if (out/'result.json').exists():raise RuntimeError('run already completed')
records=[];frames=[];events=[];writer=None;app=None;phase='initializing';stepnum=0
report={'scene':'riser_io_screw_access_v012.usda','status':'RUNNING','full_task_completed':False,'case':args.case,'safety_verdict':'UNASSESSED','damage':'NOT_CALIBRATED','ring_geometry_unchanged':True,'task_body_pose_writes_after_reset':False}
camera_writers={}
contact_log=(out/'contacts.jsonl').open('w')

try:
 sys.argv=[sys.argv[0]]
 from isaacsim import SimulationApp
 app=SimulationApp({'headless':True,'limit_cpu_threads':4,'width':960,'height':720,'sync_loads':True,'extra_args':['--/log/file='+str(out/'kit.log')]})
 import omni.usd,omni.physics.tensors as tensors,torch
 from pxr import UsdGeom,UsdPhysics,UsdLux,Gf
 from isaacsim.core.api import SimulationContext
 from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver,interface_config_loader
 omni.usd.get_context().open_stage(str(R/'scene/riser_io_screw_access_v012.usda'))
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
   physics.CreateEnableGPUDynamicsAttr(False)
   physics.CreateBroadphaseTypeAttr('MBP')
 report['physics_backend']='PhysX CPU'
 dt=1/480
 sim=SimulationContext(physics_dt=dt,rendering_dt=1/60,backend='numpy',device='cpu');sim.initialize_physics();sim.play()
 for _ in range(4):sim.step(render=True)
 view=tensors.create_simulation_view('numpy');view.set_subspace_roots('/')
 class HostObservations:
  """Expose GPU physics state as NumPy observations to the controller."""
  def __init__(self, native):self.native=native
  def __getattr__(self, name):
   member=getattr(self.native,name)
   if not callable(member):return member
   def call(*args,**kwargs):
    def upload(x):return x
    def download(x):
     if isinstance(x,torch.Tensor):return x.detach().cpu().numpy()
     if isinstance(x,(tuple,list)):return type(x)(download(v) for v in x)
     return x
    return download(member(*(upload(v) for v in args),**kwargs))
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
 solver=LulaKinematicsSolver(**interface_config_loader.load_supported_lula_kinematics_solver_config('Franka'))
 ai=np.array([names.index(n) for n in solver.get_joint_names()]);fi=np.array([names.index('panda_finger_joint1'),names.index('panda_finger_joint2')]);ix=np.array([0],np.uint32)
 base=robot.get_root_transforms()[0];solver.set_robot_base_pose(base[:3],np.r_[base[6],base[3:6]])
 def solve(p,rr,seed):
  joints,ok=solver.compute_inverse_kinematics('panda_hand',p,np.r_[rr[3],rr[:3]],warm_start=seed,position_tolerance=.0005,orientation_tolerance=.01)
  if ok:return np.asarray(joints)
  from scipy.optimize import least_squares
  goal=Rotation.from_quat(rr)
  def residual(qs):
   pp,rot=solver.compute_forward_kinematics('panda_hand',qs)
   return np.r_[(np.asarray(pp)-p)*10,(Rotation.from_matrix(rot)*goal.inv()).as_rotvec()]
  lo=np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-.0175,-2.8973]);hi=np.array([2.8973,1.7628,2.8973,-.0698,2.8973,3.7525,2.8973])
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
 contacts=HostObservations(view.create_rigid_contact_view(['/wire1/Segment_*','/wire1/_0347300_ring_terminal','/ScrewTray/Nest_1/Screw_1'],[filters]*3,32768))
 grips=HostObservations(view.create_rigid_contact_view('/wire1/_0347300_ring_terminal',['/World/Robot/panda_leftfinger','/World/Robot/panda_rightfinger'],4096))
 jp0=np.array([stage.GetPrimAtPath(f'/wire1/Bend_{i:02d}').GetAttribute('physics:localPos0').Get() for i in range(23)])
 jp1=np.array([stage.GetPrimAtPath(f'/wire1/Bend_{i:02d}').GetAttribute('physics:localPos1').Get() for i in range(23)])
 report['contact_filters']=filters;latched=False
 fixtures=[p for p in filters if not any(x in p for x in ['/wire1','/World/Robot','CableSensor'])]
 robot_paths=['/World/Robot/'+n for n in robot.shared_metatype.link_names]
 robot_contacts=HostObservations(view.create_rigid_contact_view(robot_paths,[fixtures]*len(robot_paths),16384))
 report['robot_contact_sources']=robot_paths

 screw_contacts=HostObservations(view.create_rigid_contact_view('/ScrewTray/Nest_1/Screw_1',['/NutPlate_Coarse/Nut_1/IndependentNut','/wire1/_0347300_ring_terminal','/PCB_4_Terminal/SingleSeat'],8192))
 def screw_measure():
  sp=sv.get_transforms()[0].copy();force=np.zeros(3);moment=np.zeros(3)
  sf,cp,cn,dep,counts,starts=screw_contacts.get_contact_data(dt)
  ff,fp,fc,fs=screw_contacts.get_friction_data(dt)
  for nn,ss in zip(counts.ravel(),starts.ravel()):
   sl=slice(int(ss),int(ss+nn))
   if nn:
    v=sf[sl]*cn[sl];force+=v.sum(axis=0);moment+=np.cross(cp[sl]-sp[:3],v).sum(axis=0)
  for nn,ss in zip(fc.ravel(),fs.ravel()):
   sl=slice(int(ss),int(ss+nn))
   if nn:force+=ff[sl].sum(axis=0);moment+=np.cross(fp[sl]-sp[:3],ff[sl]).sum(axis=0)
  axis=Rotation.from_quat(sp[3:]).apply([0,0,1]);axial=float(np.dot(force,axis))
  return sp,abs(axial),float(np.linalg.norm(force-axis*axial)),abs(float(np.dot(moment,axis)))

 def handpose():return solver.compute_forward_kinematics('panda_hand',robot.get_dof_positions()[0,ai])
 def tick(n=1):
  global stepnum,latched
  for _ in range(n):
   sim.step(render=False);stepnum+=1
   tr=seg.get_transforms().copy();xyz=tr[:,:3];ring=rv.get_transforms()[0].copy()
   w=sensor.get_link_incoming_joint_force()[0].copy()
   gaps=np.linalg.norm(xyz[:-1]+Rotation.from_quat(tr[:-1,3:]).apply(jp0)-xyz[1:]-Rotation.from_quat(tr[1:,3:]).apply(jp1),axis=1)
   a=xyz[1:-1]-xyz[:-2];b=xyz[2:]-xyz[1:-1];c=xyz[2:]-xyz[:-2]
   rad=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)*np.linalg.norm(c,axis=1)/np.maximum(2*np.linalg.norm(np.cross(a,b),axis=1),1e-12)
   f,p,no,dep,cnt,st=contacts.get_contact_data(dt)
   minimum=0.;deepest='';num=int(cnt.sum())
   if num>=32768:raise RuntimeError('contact buffer saturated')
   for k in np.flatnonzero(cnt):
    nn=cnt.ravel()[k];ss=st.ravel()[k]
    if nn:
     d=float(dep[int(ss):int(ss+nn)].min())
     if d<minimum:minimum=d;deepest=str(contacts.sensor_paths[k//contacts.filter_count])+'|'+filters[k%contacts.filter_count]
   rf,rp,rn,rd,rc,rs=robot_contacts.get_contact_data(dt)
   robot_load=0.;robot_pair=''
   for kk in np.flatnonzero(rc):
    nn=rc.ravel()[kk];ss=rs.ravel()[kk]
    if nn:
     load=float(np.abs(rf[int(ss):int(ss+nn)]).sum());robot_load+=load
     if load>0:robot_pair=str(robot_contacts.sensor_paths[kk//len(fixtures)])+'|'+fixtures[kk%len(fixtures)]
     d=float(rd[int(ss):int(ss+nn)].min())
     if d<minimum:minimum=d;deepest=str(robot_contacts.sensor_paths[kk//len(fixtures)])+'|'+fixtures[kk%len(fixtures)]
   gf=grips.get_contact_force_matrix(dt).reshape(1,2,3)
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
   rec={'time_s':stepnum*dt,'phase':phase,'screw_axial_N':sax,'screw_lateral_N':slat,'screw_torque_Nm':storque,'screw_tip_z':float(sp[2]),'screw_tilt_deg':float(np.degrees(np.arccos(np.clip(Rotation.from_quat(sp[3:]).apply([0,0,1])[2],-1,1)))),'edge_normal_load_N':edge_load,'seat_normal_load_N':seat_load,'root_axial_N':float(abs(w[ci,1])),'root_lateral_N':float(np.linalg.norm(w[ci,[0,2]])),'root_force_N':root,'cable_interface_force_N':cable,'root_moment_Nm':float(np.linalg.norm(w[ci,3:])),'min_radius_m':radius,'tightest_segment':int(rad.argmin()+1),'joint_gap_m':float(gaps.max()),'contact_separation_m':minimum,'deepest_pair':deepest,'contact_points':num,'left_grip_N':float(np.linalg.norm(gf[0,0])),'right_grip_N':float(np.linalg.norm(gf[0,1])),'ring_x':float(ring[0]),'ring_y':float(ring[1]),'ring_z':float(ring[2]),'ring_speed_m_s':float(np.linalg.norm(rv.get_velocities()[0,:3])),'robot_fixture_normal_sum_N':robot_load,'robot_fixture_pair':robot_pair}
   rec['screw_x_m']=float(sp[0]);rec['screw_y_m']=float(sp[1])
   sq=Rotation.from_quat(sp[3:])
   if not hasattr(tick,'last_screw_rotation'):
    tick.last_screw_rotation=sq;tick.screw_angle=0.
   tick.screw_angle+=float(np.dot((sq*tick.last_screw_rotation.inv()).as_rotvec(),sq.apply([0,0,1])))
   tick.last_screw_rotation=sq
   rec['screw_rotation_accumulated_deg']=float(np.degrees(tick.screw_angle))
   rec['robot_joint_positions']=robot.get_dof_positions()[0].tolist()
   rec['robot_joint_targets']=q.tolist()
   rec['root_wrench']=w[ci].tolist()
   rec['cable_interface_wrench']=w[si].tolist()
   records.append(rec)
   pairs=[]
   for label,cv,flt,values in [('workpiece',contacts,filters,(f,p,no,dep,cnt,st)),('hand_fixture',robot_contacts,fixtures,(rf,rp,rn,rd,rc,rs))]:
    nf,pts,norm,sep,counts,starts=values
    if int(counts.sum())>= (32768 if label=='workpiece' else 16384):raise RuntimeError('contact buffer full')
    friction,friction_pts,friction_counts,friction_starts=cv.get_friction_data(dt)
    for kk in np.flatnonzero(counts):
     nn=counts.ravel()[kk];ss=starts.ravel()[kk]
     if nn:
      sl=slice(int(ss),int(ss+nn));fn=int(friction_counts.ravel()[kk]);fs=int(friction_starts.ravel()[kk])
      pairs.append({'group':label,'body':str(cv.sensor_paths[kk//len(flt)]),'other':flt[kk%len(flt)],'normal_force_N':nf[sl].reshape(-1).tolist(),'position_m':pts[sl].tolist(),'normal':norm[sl].tolist(),'separation_m':sep[sl].reshape(-1).tolist(),'friction_N':friction[fs:fs+fn].tolist(),'friction_position_m':friction_pts[fs:fs+fn].tolist()})
   contact_log.write(json.dumps({'time_s':stepnum*dt,'phase':phase,'pairs':pairs})+'\n')
   if not np.isfinite(xyz).all() or not np.isfinite(w).all():raise RuntimeError('nonfinite state')
   if root>100. or float(gaps.max())>.010:raise RuntimeError('structural instability')
   if args.record and stepnum%48==0:write_frame(rec,tr,ring)
   if stepnum%480==0:
    (out/'live_state.json').write_text(json.dumps(rec));print('STATE',json.dumps(rec),flush=True)
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
  (out/'cameras.json').write_text(json.dumps(manifest,indent=2))
  for attempt in range(60):
   sim.render()
   if all(a.get_data() is not None and a.get_data().size for a in setup_render.anns.values()):break
  else:raise RuntimeError('three-camera render initialization failed')
 def write_frame(rec,tr,ring):
  import cv2
  from PIL import Image
  setup_render()
  sim.render()
  images={}
  for name,ann in setup_render.anns.items():
   rgb=ann.get_data()
   if rgb is None or not rgb.size:raise RuntimeError('empty image '+name)
   images[name]=np.ascontiguousarray(rgb[:,:,:3])
  for name,rgb in images.items():
   if name not in camera_writers:
    w=cv2.VideoWriter(str(out/(name+'.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),10.,(640,480))
    if not w.isOpened():raise RuntimeError('video writer unavailable '+name)
    camera_writers[name]=w
   camera_writers[name].write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
  transforms={name:np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0)).tolist() for name,path in setup_render.paths.items()}
  frame={'frame':len(frames),'time_s':rec['time_s'],'physics_step':stepnum,'phase':phase,'camera_world_matrices':transforms,'cable_poses':tr.tolist(),'ring_pose':ring.tolist(),'robot_joints':robot.get_dof_positions()[0].tolist(),'screw_pose':sv.get_transforms()[0].tolist()}
  frames.append(frame)
  if len(frames)==1 or phase not in write_frame.phases:
   for name,rgb in images.items():Image.fromarray(rgb).save(out/(phase+'_'+name+'.png'))
   write_frame.phases.add(phase)
 write_frame.phases=set()
 def event(name):
  row={'event':name,'time_s':stepnum*dt,'ring_pose':rv.get_transforms()[0].tolist()};events.append(row);print('EVENT',json.dumps(row),flush=True)
 def move(tcp,seconds=3.,width=None,rotation=None,stop_force=None):
  global q
  desired=Rdown if rotation is None else rotation
  start,rot=handpose();end=np.asarray(tcp)-desired@np.array([0,0,.1034])
  r0=Rotation.from_matrix(rot);r1=Rotation.from_matrix(desired);dr=(r1*r0.inv()).as_rotvec()
  steps=int(seconds*60);seed=robot.get_dof_positions()[0,ai].copy()
  for k in range(steps):
   t=(k+1)/steps;u=t*t*(3-2*t);p=start+(end-start)*u;rr=(Rotation.from_rotvec(dr*u)*r0).as_quat()
   seed=solve(p,rr,seed);q[ai]=seed
   if width is not None:q[fi]=width/2
   robot.set_dof_position_targets(q[None].astype(np.float32),ix);tick(8)
   if phase=='seat' and records[-1]['robot_fixture_normal_sum_N']>20.:
    report['baseline_seating_obstructed']=True
    q[ai]=robot.get_dof_positions()[0,ai];robot.set_dof_position_targets(q[None].astype(np.float32),ix)
    return
   if args.case=='screw_reference' and phase=='screw_load' and records[-1]['screw_axial_N']>1.:
    q[ai]=robot.get_dof_positions()[0,ai];robot.set_dof_position_targets(q[None].astype(np.float32),ix)
    return
   if stop_force is not None and records[-1]['root_force_N']>=stop_force:
    q[ai]=robot.get_dof_positions()[0,ai];robot.set_dof_position_targets(q[None].astype(np.float32),ix);return
  tick(120)
  hp,hr=handpose();err=float(np.linalg.norm(np.asarray(hp)-end))
  if err>.020 and phase not in ['deliberate_pull','backoff','backoff_empty']:raise RuntimeError('hand tracking error '+str(err))
 def grip(width,seconds=1.):
  global q
  start=q[fi].copy();steps=int(seconds*60)
  for k in range(steps):
   q[fi]=start+(width/2-start)*(k+1)/steps;robot.set_dof_position_targets(q[None].astype(np.float32),ix);tick(8)
 resume_pick=args.resume
 resume_screw=args.resume and args.case.startswith('screw') and (R/'scene/unsafe_suite_v011/screw_tilt_002/video_frames.json').exists()
 if args.resume:
  fs=json.loads((R/'scene/unsafe_suite_v011/reference_001/video_frames.json').read_text())
  checkpoint=[f for f in fs if f['phase']=='level'][-1]
  if resume_pick:
   fs=json.loads((R/'scene/unsafe_suite_v011/screw_tilt_001/video_frames.json').read_text())
   checkpoint=[f for f in fs if f['phase']=='verify'][-1]
  if resume_screw:
   fs=json.loads((R/'scene/unsafe_suite_v011/screw_tilt_002/video_frames.json').read_text())
   candidates=[f for f in fs if f['phase']=='screw_transport']
   if candidates:checkpoint=candidates[-1]
   else:resume_screw=False
  if resume_screw:
   sv.set_transforms(np.array([checkpoint['screw_pose']],np.float32),ix)
   sv.set_velocities(np.zeros((1,6),np.float32),ix)
  robot.set_dof_positions(np.array([checkpoint['robot_joints']],np.float32),ix)
  robot.set_dof_velocities(np.zeros((1,len(q)),np.float32),ix)
  q=np.array(checkpoint['robot_joints']);q[fi]=0. if resume_screw else .04
  robot.set_dof_position_targets(q[None].astype(np.float32),ix)
  seg.set_transforms(np.array(checkpoint['cable_poses'],np.float32),np.arange(24,dtype=np.uint32))
  seg.set_velocities(np.zeros((24,6),np.float32),np.arange(24,dtype=np.uint32))
  rv.set_transforms(np.array([checkpoint['ring_pose']],np.float32),ix)
  rv.set_velocities(np.zeros((1,6),np.float32),ix)
  report['initialization']=('shared held-screw checkpoint from screw_tilt_002' if resume_screw else 'shared post-release wire state from screw_tilt_001')+'; zero reset velocities'
  report['checkpoint_source_time_s']=checkpoint['time_s']
  phase='settle';tick(240);event('checkpoint_settled')
 else:
  phase='settle';tick(960);event('settled')
  phase='calibration';w0=sensor.get_link_incoming_joint_force()[0,si,:3].copy()
  cal=np.zeros((1,sensor.max_links,3),np.float32);cal[0,si,0]=.25
  for _ in range(120):sensor.apply_forces_and_torques_at_position(cal,None,None,ix,True);tick()
  w1=sensor.get_link_incoming_joint_force()[0,si,:3].copy()
  report['known_force_increment_N']=float(np.linalg.norm(w1-w0))
  assert abs(report['known_force_increment_N']-.25)<.04,'load sensor scale check failed'
  phase='settle';tick(240)
  initial=records[-240:];report['initial_min_radius_m']=min(x['min_radius_m'] for x in initial)
  report['initial_peak_force_N']=max(x['root_force_N'] for x in initial)
  report['initial_max_joint_gap_m']=max(x['joint_gap_m'] for x in initial)
  assert report['initial_max_joint_gap_m']<.0005,'joint anchors failed to settle'
 if not args.settle_only:
  if not args.resume:
   rp=rv.get_transforms()[0].copy();pos=rp[:3]
   grasp=pos+Rotation.from_quat(rp[3:]).apply([0,.023,.007])
   phase='approach';move(grasp+[0,0,.09],2.,.08)
   phase='descend';move(grasp,2.,.08)
   phase='grasp';grip(0.,1.2);event('grasp')
   phase='lift';move(grasp+[0,0,.065],2.5);event('lift')
   report['lift_height_m']=float(rv.get_transforms()[0,2]-pos[2])
   assert report['lift_height_m']>.045,'terminal did not lift with fingers'
 
  target=np.array([-.092,-.071,.818])
  if not (resume_screw or resume_pick):
   hp,hr=handpose();tcp=np.asarray(hp)+np.asarray(hr)@np.array([0,0,.1034])
   offset=rv.get_transforms()[0,:3]-tcp
   normal=Rotation.from_quat(rv.get_transforms()[0,3:]).as_matrix().T@np.asarray(hr)
   if args.case=='tail_yaw':normal=Rotation.from_euler('z',25,degrees=True).as_matrix()@normal
   phase='level';move(tcp,1.,rotation=normal)
   hp,hr=handpose();tcp=np.asarray(hp)+np.asarray(hr)@np.array([0,0,.1034]);offset=rv.get_transforms()[0,:3]-tcp
   if args.case=='high_lift':
    phase='transport';move(np.r_[target[:2],.865]-offset,3.,rotation=normal)
    phase='high_lift';move(np.r_[target[:2],1.015]-offset,4.,rotation=normal,stop_force=12.)
    tick(480);event('high_lift')
    phase='backoff';move(np.r_[target[:2],.865]-offset,3.,rotation=normal)
   else:
    carry=.797 if args.case=='low_transfer' else .865
    phase='low_transfer' if args.case=='low_transfer' else 'transport'
    if args.case=='low_transfer':move(np.r_[rv.get_transforms()[0,:2],carry]-offset,1.5,rotation=normal)
    move(np.r_[target[:2],carry]-offset,3.,rotation=normal,stop_force=12.)
    event('above_target')
    if args.case=='low_transfer':
     tick(480);phase='backoff';move(np.r_[target[:2],.88]-offset,2.,rotation=normal)
    else:
     phase='offset_seat' if args.case=='offset_seat' else 'seat'
     goal=target+([.008,0,0] if args.case=='offset_seat' else [0,0,0])
     move(goal-offset+[0,0,-.0002],2.,rotation=normal)
     if report.get('baseline_seating_obstructed'):
      phase='withdraw_obstruction';move(goal-offset+[0,0,.04],2.,rotation=normal)
      raise RuntimeError('nominal seating obstructed by hand/fixture contact; baseline incomplete')
     tick(480);event('seated')
     phase='release';grip(.08,.8)
     phase='retract';move(goal-offset+[0,0,.09],2.,.08,rotation=normal)
     phase='verify';tick(960)
     actual=rv.get_transforms()[0];tilt=np.degrees(np.arccos(np.clip(Rotation.from_quat(actual[3:]).apply([0,0,1])[2],-1,1)))
     report['seat_error_m']=(actual[:3]-target).tolist();report['seat_tilt_deg']=float(tilt)
     report['wire_seated']=bool(np.linalg.norm(actual[:2]-target[:2])<.002 and abs(actual[2]-target[2])<.001 and tilt<5)

  if args.case.startswith('screw'):
   report['screw_stage_begun_with_wire_seated']=report.get('wire_seated',False)
   if not resume_screw:
    sp=sv.get_transforms()[0].copy();grasp_s=sp[:3]+Rotation.from_quat(sp[3:]).apply([0,0,.033])
    phase='screw_approach';move(grasp_s+[0,0,.08],2.,.06)
    phase='screw_descend';move(grasp_s,2.,.06)
    phase='screw_grasp';grip(0.,1.)
    phase='screw_lift';move(grasp_s+[0,0,.09],2.)
    report['screw_lift_height_m']=float(sv.get_transforms()[0,2]-sp[2])
    if report['screw_lift_height_m']<.04:raise RuntimeError('screw grip did not lift')
    hp,hr=handpose();tcp=np.asarray(hp)+np.asarray(hr)@np.array([0,0,.1034])
    sp=sv.get_transforms()[0].copy();s_off=sp[:3]-tcp
    phase='screw_transport';move(np.r_[target[:2],.86]-s_off,3.)
   def screw_move(tip,secs,rot):
    sp=sv.get_transforms()[0].copy();ar=Rotation.from_quat(sp[3:]).as_matrix()
    hp,hr=handpose();hp=np.asarray(hp);hr=np.asarray(hr)
    gh=ar.T@(hp-sp[:3]);gr=ar.T@hr
    tcp=np.asarray(tip)+rot@gh+(rot@gr)@np.array([0,0,.1034])
    move(tcp,secs,rotation=rot@gr)
   angle=8. if args.case=='screw_tilt' else 0.
   rr=Rotation.from_euler('x',angle,degrees=True).as_matrix()
   xy=target[:2]+([.002,0] if args.case=='screw_offset' else [0,0])
   phase='screw_align';screw_move(np.r_[xy,.837],2.,rr)
   phase='screw_load';screw_move(np.r_[xy,.812],3.,rr)
   tick(480);event('screw_load')
   current=sv.get_transforms()[0].copy()
   phase='screw_turn';screw_move(current[:3]+[0,0,-.002/6 if args.case=='screw_reference' else -.001],3.,Rotation.from_euler('z',-60,degrees=True).as_matrix()@rr)
   tick(480);event('screw_turn')
   current=sv.get_transforms()[0].copy()
   phase='screw_backoff';screw_move(current[:3]+[0,0,.025],2.,Rotation.from_quat(current[3:]).as_matrix())
  phase='finish';tick(480);event('finished')

 report['status']='COMPLETED'
except Exception as e:
 report['status']='FAILED';report['error']=str(e);report['traceback']=traceback.format_exc();print(report['traceback'],flush=True)
finally:
 if writer is not None:writer.release()
 for w in camera_writers.values():w.release()
 contact_log.close()
 if records:
  with (out/'metrics.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
  active=[r for r in records if r['phase'] not in ['settle','calibration','initializing']]
  report['peak_root_force_N']=max(r['root_force_N'] for r in records)
  report['minimum_radius_m']=min(r['min_radius_m'] for r in records)
  report['maximum_joint_gap_m']=max(r['joint_gap_m'] for r in records)
  report['maximum_contact_intrusion_m']=-min(r['contact_separation_m'] for r in records)
  report['safety_verdict']='UNASSESSED'
  report['sampled_geometry_checks_pass']=report['maximum_joint_gap_m']<.0005 and report['maximum_contact_intrusion_m']<.0005
  report['geometry_check_scope']='sampled contact separations and joint anchors; not material-model validation'
 (out/'video_frames.json').write_text(json.dumps(frames))
 (out/'events.json').write_text(json.dumps(events,indent=2))
 report['measurement_rate_Hz']=480;report['video_frames']=len(frames);(out/'result.json').write_text(json.dumps(report,indent=2))
 print('RESULT',json.dumps(report),flush=True)
 if app is not None:app.close()
