"""Rate-limited joint DRIVE targets only. Never teleports joints or bodies."""
import numpy as np
import time
from kinematics import PoseResidual
from scipy.spatial.transform import Rotation
class FrankaTeleop:
 def __init__(self,config):
  self.cfg=config;self.ready=False;self.armed=False;self.error=None;self.commands=0;self.elapsed=0.
  self.goal=None;self.width_goal=None;self.last_action=None;self.initial_q=None;self.last_q=None
 def initialize(self,physics_sim_view=None):
  import omni.usd,omni.physics.tensors as tensors
  from pxr import UsdPhysics
  from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver,interface_config_loader
  root=self.cfg["paths"]["robot"]
  roots=[str(p.GetPath()) for p in omni.usd.get_context().get_stage().Traverse()
    if (str(p.GetPath())==root or str(p.GetPath()).startswith(root+"/")) and p.HasAPI(UsdPhysics.ArticulationRootAPI)]
  if len(roots)!=1:raise RuntimeError("Expected one physical Franka articulation root: "+repr(roots))
  self.sim_view=physics_sim_view if physics_sim_view is not None else tensors.create_simulation_view("numpy")
  self.sim_view.set_subspace_roots("/")
  self.view=self.sim_view.create_articulation_view(roots[0])
  if self.view.count!=1:raise RuntimeError("Franka articulation did not initialize")
  self.robot=self
  self.solver=LulaKinematicsSolver(**interface_config_loader.load_supported_lula_kinematics_solver_config("Franka"))
  names=list(self.view.shared_metatype.dof_names);self.dof_names=names;self.arm_indices=np.array([names.index(n) for n in self.solver.get_joint_names()],dtype=int)
  self.fingers=np.array([names.index("panda_finger_joint1"),names.index("panda_finger_joint2")],dtype=int)
  self.initial_q=np.asarray(self.robot.get_joint_positions()).copy()
  if len(self.initial_q)!=len(names) or not np.isfinite(self.initial_q).all():raise RuntimeError("Invalid measured joint state")
  self.lower=[];self.upper=[]
  from pxr import UsdPhysics
  stage=omni.usd.get_context().get_stage()
  joints={p.GetName():p for p in stage.Traverse() if str(p.GetPath()).startswith(root+"/") and p.IsA(UsdPhysics.RevoluteJoint)}
  for name in self.solver.get_joint_names():
   j=UsdPhysics.RevoluteJoint(joints[name])
   self.lower.append(np.deg2rad(j.GetLowerLimitAttr().Get()));self.upper.append(np.deg2rad(j.GetUpperLimitAttr().Get()))
  self.lower=np.array(self.lower);self.upper=np.array(self.upper)
  self.ik_diagnostics={};self.fallback_count=0;self.last_solution=None
  velocities=self.view.get_dof_velocities()[0]
  if np.max(np.abs(velocities[self.arm_indices]))>.01:raise RuntimeError("Robot still moving; wait for settling before initialization")
  self.hold_bias=self.view.get_dof_position_targets()[0].copy()-self.initial_q
  if np.max(np.abs(self.hold_bias[self.arm_indices]))>.01:raise RuntimeError("Existing drive targets are too far from measured pose")
  self.ready=True;self.armed=False;self.sync_goal()
  self.width_goal=float(np.sum(self.initial_q[self.fingers]));self.last_q=self.initial_q.copy()
  return names
 def get_joint_positions(self):
  return self.view.get_dof_positions()[0].copy()
 def get_world_pose(self):
  p=self.view.get_root_transforms()[0].copy()
  q=np.asarray(np.r_[p[6],p[3:6]],dtype=float);q/=np.linalg.norm(q)
  return np.asarray(p[:3],dtype=float),q
 def apply_targets(self,q):
  self.view.set_dof_position_targets(np.asarray(q,dtype=np.float32).reshape(1,-1),np.array([0],dtype=np.uint32))
 def pose(self):
  p,q=self.get_world_pose();self.solver.set_robot_base_pose(p,q)
  return self.solver.compute_forward_kinematics(self.cfg["control"]["ee_frame"],self.get_joint_positions()[self.arm_indices])
 def sync_goal(self):
  p,R=self.pose();self.goal=(np.array(p).copy(),Rotation.from_matrix(R));self.cart_command=(self.goal[0].copy(),self.goal[1])
 def arm(self):
  if not self.ready:raise RuntimeError("Play and initialize robot first")
  measured=self.get_joint_positions()
  targets=self.view.get_dof_position_targets()[0].copy()
  self.hold_bias=targets-measured
  if np.max(np.abs(self.hold_bias[self.arm_indices]))>.01:raise RuntimeError("Drive/measurement mismatch; reinitialize after settling")
  self.sync_goal();self.last_action=targets;self.last_solution=None;self.armed=True;self.error=None;self.elapsed=0
 def stop(self,reason="USER STOP"):
  self.armed=False;self.error=reason
  if self.ready:
   try:
    q=np.asarray(self.robot.get_joint_positions()).copy()
    q[self.arm_indices]+=self.hold_bias[self.arm_indices]
    self.apply_targets(q)
    self.last_action=q.copy();self.last_q=q.copy()
    self.width_goal=float(np.sum(q[self.fingers]));self.sync_goal()
   except Exception as e:
    self.ready=False;self.error=reason+"; physics handle unavailable: "+repr(e)
 def jog(self,axis,sign,rotate=False):
  if not self.armed:raise RuntimeError("Arm manual control first")
  p,R=self.goal;v=np.zeros(3)
  if rotate:
   v[axis]=sign*self.cfg["control"]["rotation_jog_rad"]
   R=Rotation.from_rotvec(v)*R
  else:p=p.copy();p[axis]+=sign*self.cfg["control"]["translation_jog_m"]
  actual,_=self.pose()
  if np.linalg.norm(p-actual)>self.cfg["control"]["tracking_error_stop_m"]:raise RuntimeError("Queued target exceeds 3 mm tracking envelope")
  self.goal=(p,R)
 def gripper(self,open_gripper):
  if not self.armed:raise RuntimeError("Arm manual control first")
  self.width_goal=.08 if open_gripper else 0.
 def tick(self,dt):
  if not self.ready:return
  self.elapsed+=dt
  if not self.armed or self.elapsed<self.cfg["control"]["ik_period_s"]:return
  h=min(self.elapsed,.05);self.elapsed=0
  p,Rm=self.pose();R=Rotation.from_matrix(Rm);gp,gR=self.goal
  d=gp-p;dist=np.linalg.norm(d)
  if dist>self.cfg["control"]["tracking_error_stop_m"]:self.stop("EE TRACKING ERROR");return
  cp,cR=self.cart_command;cd=gp-cp;cdist=np.linalg.norm(cd)
  p_cmd=cp+cd*min(1.,self.cfg["control"]["translation_speed_m_s"]*h/max(cdist,1e-12))
  rv=(gR*cR.inv()).as_rotvec();a=np.linalg.norm(rv)
  R_cmd=Rotation.from_rotvec(rv*min(1.,self.cfg["control"]["rotation_speed_rad_s"]*h/max(a,1e-12)))*cR
  qq=R_cmd.as_quat();wxyz=np.r_[qq[3],qq[:3]]
  solve_start=time.perf_counter()
  problem=PoseResidual(self.solver,self.cfg["control"]["ee_frame"],p_cmd,R_cmd)
  errors=problem.errors
  pos_tol=self.cfg["control"].get("ik_position_tolerance_m",1e-6)
  rot_tol=self.cfg["control"].get("ik_rotation_tolerance_rad",1e-4)
  candidate=self.last_solution
  pe,re=errors(candidate) if candidate is not None else (np.full(3,np.inf),np.full(3,np.inf))
  ok=candidate is not None and np.linalg.norm(pe)<=pos_tol and np.linalg.norm(re)<=rot_tol
  used_cache=bool(ok)
  if not ok:
   # Local IK only: bounded evaluations, previous solution seed, no global random search.
   from scipy.optimize import least_squares
   from solver_metrics import install
   install()
   residual=problem
   from batched_fd import jacobian as batch_jacobian
   fd_evaluations=0
   def jacobian(x):
    nonlocal fd_evaluations
    fd_evaluations+=1
    return batch_jacobian(problem,x)
   seed=candidate if candidate is not None else self.get_joint_positions()[self.arm_indices].astype(float)
   seed=np.clip(np.asarray(seed,dtype=float),self.lower+1e-8,self.upper-1e-8)
   result=least_squares(residual,seed,bounds=(self.lower,self.upper),jac=jacobian,
     max_nfev=self.cfg["control"].get("ik_max_evaluations",40),xtol=1e-10,ftol=1e-10,gtol=1e-10)
   candidate=result.x;self.fallback_count+=1;pe,re=errors(candidate)
   ok=bool(result.success)
  self.ik_diagnostics={"seconds":time.perf_counter()-solve_start,"residual_calls":problem.residual_calls,"jacobian_calls":fd_evaluations if not used_cache else 0,"fk_calls":problem.fk_calls,"iterations":int(result.nit) if not used_cache else 0,"nfev":int(result.nfev) if not used_cache else 0,"njev":int(result.njev) if not used_cache else 0,"solver_calls":int(not used_cache),"retries":0,"local_solver_success":bool(ok),"used_verified_cache":used_cache,"position_error_m":float(np.linalg.norm(pe)),
    "orientation_error_rad":float(np.linalg.norm(re)),"fallback_count":self.fallback_count}
  self.ik_diagnostics.update(
    candidate_joint_positions=np.asarray(candidate).tolist(),
    measured_joint_positions=self.get_joint_positions()[self.arm_indices].tolist(),
    lower_limits_rad=self.lower.tolist(),upper_limits_rad=self.upper.tolist(),
    joint_margin_rad=np.minimum(candidate-self.lower,self.upper-candidate).tolist(),
    position_tolerance_m=pos_tol,orientation_tolerance_rad=rot_tol,
    requested_goal_position_m=np.asarray(gp).tolist(),requested_goal_quaternion_xyzw=gR.as_quat().tolist(),
    commanded_position_m=np.asarray(p_cmd).tolist(),commanded_quaternion_xyzw=R_cmd.as_quat().tolist(),
    measured_ee_position_m=np.asarray(p).tolist(),measured_ee_quaternion_xyzw=R.as_quat().tolist())
  if not np.isfinite(candidate).all() or np.any(candidate<self.lower) or np.any(candidate>self.upper) or np.linalg.norm(pe)>pos_tol or np.linalg.norm(re)>rot_tol:
   self.stop("IK RESIDUAL/LIMIT FAILURE - no new motion");return
  self.last_solution=candidate.copy()
  self.cart_command=(p_cmd,R_cmd);desired=np.clip(candidate+self.hold_bias[self.arm_indices],self.lower,self.upper)
  q=np.asarray(self.last_action).copy()
  desired=np.asarray(desired,dtype=float)
  indices=self.arm_indices
  delta=np.clip(desired-q[indices],-self.cfg["control"]["arm_joint_speed_rad_s"]*h,self.cfg["control"]["arm_joint_speed_rad_s"]*h)
  q[indices]+=delta
  measured=self.robot.get_joint_positions()
  each=self.width_goal/2.
  finger_step=self.cfg["control"]["gripper_speed_m_s"]*h/2.
  q[self.fingers]=self.last_action[self.fingers]+np.clip(each-self.last_action[self.fingers],-finger_step,finger_step)
  self.apply_targets(q)
  self.last_action=q.copy();self.last_q=np.array(measured).copy();self.commands+=1
