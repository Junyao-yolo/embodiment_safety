from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
from isaacsim import SimulationApp
R=Path("/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io")
app=SimulationApp({"headless":True,"limit_cpu_threads":4,"extra_args":["--/log/file="+str(R/"scene/success_v014/current_kit.log")]})
import omni.usd,omni.physics.tensors as tensors,torch
from pxr import UsdPhysics,PhysxSchema,Gf
from isaacsim.core.api import SimulationContext
omni.usd.get_context().open_stage(str(R/"scene/riser_io_success_v014.usda"))
stage=omni.usd.get_context().get_stage();stage.SetEditTarget(stage.GetSessionLayer())
for path in ["/World/Robot","/World/CableSensor","/TE_L26620011_4_connector_v2"]:
 stage.GetPrimAtPath(path).SetActive(False)
ring="/wire1/_0347300_ring_terminal";screw="/ScrewTray/Nest_1/Screw_1"
for prim in stage.GetPrimAtPath("/wire1").GetChildren():
 if str(prim.GetPath())!=ring:prim.SetActive(False)
for prim in stage.Traverse():
 if prim.IsA(UsdPhysics.Scene):
  api=PhysxSchema.PhysxSceneAPI.Apply(prim);api.CreateEnableGPUDynamicsAttr(True);api.CreateBroadphaseTypeAttr("GPU")
 if prim.HasAPI(UsdPhysics.RigidBodyAPI):
  PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
j=UsdPhysics.Joint.Define(stage,"/World/ValidationSpindle");j.CreateBody1Rel().SetTargets([screw])
j.CreateLocalPos0Attr(Gf.Vec3f(-.092,-.071,.805));j.CreateLocalPos1Attr(Gf.Vec3f(0))
for f in [j.CreateLocalRot0Attr,j.CreateLocalRot1Attr]:f(Gf.Quatf(.70710678,Gf.Vec3f(0,-.70710678,0)))
for dof in ["transY","transZ","rotY","rotZ"]:
 a=UsdPhysics.LimitAPI.Apply(j.GetPrim(),dof);a.CreateLowAttr(1);a.CreateHighAttr(-1)
drive=UsdPhysics.DriveAPI.Apply(j.GetPrim(),"rotX");drive.CreateTypeAttr("force");drive.CreateStiffnessAttr(0);drive.CreateDampingAttr(.003);drive.CreateMaxForceAttr(.01);drive.CreateTargetVelocityAttr(0)
hz=120;dt=1/hz
sim=SimulationContext(physics_dt=dt,rendering_dt=1/60,backend="torch",device="cuda:0");sim.initialize_physics();sim.play()
view=tensors.create_simulation_view("torch");view.set_subspace_roots("/")
sv=view.create_rigid_body_view(screw);rv=view.create_rigid_body_view(ring)
t=lambda x:torch.as_tensor(np.asarray(x,np.float32),device="cuda:0")
ix=torch.tensor([0],dtype=torch.int32,device="cuda:0")
rv.set_transforms(t([[-.092,-.071,.818,0,0,0,1]]),ix);rv.set_velocities(t([[0]*6]),ix)
sv.set_transforms(t([[-.092,-.071,.805,0,0,0,1]]),ix);sv.set_velocities(t([[0]*6]),ix)
cv=view.create_rigid_contact_view(screw,["/NutPlate_Coarse/Nut_1/IndependentNut",ring,"/PCB_4_Terminal/SingleSeat"],2048)
released=None;held=[];max_error=0.;previous_ring=None;velocity_rows=[];hold_positions=[]
for k in range(hz*24):
 if k==hz:drive.GetTargetVelocityAttr().Set(-45.)
 force=[0,0,-.3] if released is None else [0,0,0]
 sv.apply_forces_and_torques_at_position(t([force]),t([[0,0,0]]),None,ix,True)
 sim.step(render=False)
 pose=sv.get_transforms()[0].cpu().numpy().copy();rot=Rotation.from_quat(pose[3:])
 ring_pose=rv.get_transforms()[0].cpu().numpy().copy();ring_velocity=rv.get_velocities()[0].cpu().numpy().copy()
 pose_speed=float(np.linalg.norm(ring_pose[:3]-previous_ring[:3])/dt) if previous_ring is not None else 0.
 previous_ring=ring_pose.copy()
 if released is not None and k>released+hz:
  velocity_rows.append([float(np.linalg.norm(ring_velocity[:3])),pose_speed]);hold_positions.append(ring_pose[:3].tolist())
 f,p,n,d,c,st=[x.cpu().numpy().copy() for x in cv.get_contact_data(dt)]
 friction=[x.cpu().numpy().copy() for x in cv.get_friction_data(dt)]
 mat=cv.get_contact_force_matrix(dt).cpu().numpy().reshape(3,3)
 sums=[]
 for count,start in zip(c.ravel(),st.ravel()):
  sl=slice(int(start),int(start+count));sums.append((f[sl]*n[sl]).sum(axis=0))
 error=float(np.max(np.abs(np.asarray(sums)-mat)));max_error=max(max_error,error)
 if error>.001:raise RuntimeError("normal force readback mismatch "+str(error))
 start=int(st.ravel()[1]);count=int(c.ravel()[1]);sl=slice(start,start+count)
 local=rot.inv().apply(p[sl]-pose[:3]);mask=(np.linalg.norm(local[:,:2],axis=1)>.0055)&(np.abs(local[:,2]-.018)<.0005)
 clamp=float(abs(np.dot((f[sl]*n[sl])[mask].sum(axis=0),rot.apply([0,0,1]))))
 if clamp>2. and k>hz and released is None:
  released=k;drive.GetMaxForceAttr().Set(0.);drive.GetDampingAttr().Set(0.)
 if released is not None and k>released+hz:held.append(clamp)
 if k%hz==0:
  print("CONTACT_CHECK",json.dumps({"t":k/hz,"z":float(pose[2]),"clamp_N":clamp,"ring_normal_N":mat[1].tolist(),"mismatch_N":error,"ring_contact_local_points":local.tolist(),"released":released is not None,"ring_velocity_m_s":float(np.linalg.norm(ring_velocity[:3])),"ring_pose_speed_m_s":pose_speed}),flush=True)
 if released is not None and k>released+3*hz:break
print("COMPONENT_RESULT",json.dumps({"max_readback_error_N":max_error,"hold_position_span_m":np.ptp(np.array(hold_positions),axis=0).tolist() if hold_positions else [],"hold_peak_native_speed_m_s":max([x[0] for x in velocity_rows],default=0.),"hold_peak_pose_speed_m_s":max([x[1] for x in velocity_rows],default=0.),"retained_head_clamp_N":float(np.mean(held)) if held else None,"success":bool(held and np.mean(held)>.2),"scope":"spindle component check; not a robot trajectory"}),flush=True)
app.close()
