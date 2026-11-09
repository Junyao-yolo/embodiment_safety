"""State-queryable expert commands, smooth perturbations and DAgger execution."""
import numpy as np
from scipy.spatial.transform import Rotation
from interface import JOINT_MIN,JOINT_MAX

class JointCommandLimits:
 def __init__(self,initial,hz=60.,velocity=1.8,acceleration=10.):
  self.previous=np.asarray(initial).copy();self.velocity=np.zeros(7)
  self.hz=hz;self.vmax=velocity;self.amax=acceleration
 def preview(self,desired):
  desired=np.asarray(desired);v=(desired[:7]-self.previous[:7])*self.hz
  step=self.amax/self.hz
  distance_low=np.maximum(0.,self.previous[:7]-JOINT_MIN)
  distance_high=np.maximum(0.,JOINT_MAX-self.previous[:7])
  brake_low=np.sqrt(step*step+2*self.amax*distance_low)-step
  brake_high=np.sqrt(step*step+2*self.amax*distance_high)-step
  lower=np.maximum(np.maximum(-self.vmax,self.velocity-step),-brake_low)
  upper=np.minimum(np.minimum(self.vmax,self.velocity+step),brake_high)
  v=np.clip(v,lower,upper)
  q=self.previous.copy();q[:7]=np.clip(q[:7]+v/self.hz,JOINT_MIN,JOINT_MAX)
  width=np.clip(desired[7:9].sum(),0.,.08);old=q[7:9].sum()
  q[7:9]=(old+np.clip(width-old,-.15/self.hz,.15/self.hz))/2
  return q
 def commit(self,desired):
  q=self.preview(desired);self.velocity=(q[:7]-self.previous[:7])*self.hz
  self.previous=q.copy();return q

class FeedbackExpert:
 """Query a reachable correction to an explicit task reference from current physics state.

 Task memory consists of the current reference and measured grasp transform.
 Querying does not advance the task stage or mutate the physics state.
 """
 def __init__(self,solve,fk,limits):
  self.solve=solve;self.fk=fk;self.limits=limits;self.reference=None
  self.linear_m_s=.15;self.angular_rad_s=1.85
 def set_motion_limits(self,linear_m_s,angular_rad_s):
  self.linear_m_s=float(linear_m_s);self.angular_rad_s=float(angular_rad_s)
 def set_reference(self,position,rotation,width,body=None,grasp_offset=None,grasp_rotation=None):
  self.reference=dict(position=np.asarray(position).copy(),rotation=np.asarray(rotation).copy(),
   width=float(width),body=body,grasp_offset=grasp_offset,grasp_rotation=grasp_rotation)
 def query(self,state):
  ref=self.reference
  if ref is None:raise RuntimeError("Expert needs a task reference")
  p=ref["position"].copy();rot=ref["rotation"].copy()
  if ref["body"] is not None:
   actual=np.asarray(state[ref["body"]])[0]
   if not np.isfinite(actual).all():raise RuntimeError("Nonfinite body state")
   # Pose feedback compensates contact compliance using the measured held body.
   p+=np.clip(.25*(p-actual[:3]),-.002,.002)
   error=(Rotation.from_matrix(rot)*Rotation.from_quat(actual[3:]).inv()).as_rotvec()
   rot=Rotation.from_rotvec(np.clip(.15*error,-.03,.03)).as_matrix()@rot
   p+=rot@ref["grasp_offset"];rot=rot@ref["grasp_rotation"]
  if ref["body"] is None:
   measured,_=self.fk(np.asarray(state["joints"])[0,:7])
   p+=np.clip(.1*(p-np.asarray(measured)),-.001,.001)
  previous=self.limits.previous
  origin,orient=self.fk(previous[:7])
  dp=p-np.asarray(origin);angle=(Rotation.from_matrix(rot)*Rotation.from_matrix(orient).inv()).as_rotvec()
  # A query remains locally reachable even after a policy departure from the reference.
  dp*=min(1.,(self.linear_m_s/self.limits.hz)/max(np.linalg.norm(dp),1e-12))
  angle*=min(1.,(self.angular_rad_s/self.limits.hz)/max(np.linalg.norm(angle),1e-12))
  target=previous.copy()
  target[:7]=self.solve(np.asarray(origin)+dp,
   (Rotation.from_rotvec(angle)*Rotation.from_matrix(orient)).as_quat(),previous[:7])
  target[7:9]=ref["width"]/2
  return self.limits.preview(target)
 def tracking_error(self,state):
  ref=self.reference
  if ref["body"] is None:
   p,r=self.fk(np.asarray(state["joints"])[0,:7])
   position=np.linalg.norm(np.asarray(p)-ref["position"])
   angle=np.linalg.norm((Rotation.from_matrix(ref["rotation"])*Rotation.from_matrix(r).inv()).as_rotvec())
  else:
   actual=np.asarray(state[ref["body"]])[0]
   position=np.linalg.norm(actual[:3]-ref["position"])
   angle=np.linalg.norm((Rotation.from_matrix(ref["rotation"])*Rotation.from_quat(actual[3:]).inv()).as_rotvec())
  return float(max(position,.03*angle))

class SmoothVelocityNoise:
 """Bounded correlated Cartesian velocity noise, specified in metres/second."""
 def __init__(self,seed,std_m_s=.002,tau_s=.35,acceleration_m_s2=.015):
  self.rng=np.random.default_rng(seed);self.std=std_m_s;self.tau=tau_s
  self.acceleration=acceleration_m_s2;self.value=np.zeros(3)
 def sample(self,dt,scale=1.):
  alpha=np.exp(-dt/self.tau)
  desired=alpha*self.value+np.sqrt(1-alpha*alpha)*self.std*scale*self.rng.normal(size=3)
  desired=np.clip(desired,-3*self.std*scale,3*self.std*scale)
  self.value+=np.clip(desired-self.value,-self.acceleration*dt,self.acceleration*dt)
  return self.value.copy()

class RecoveryRouter:
 """Choose contiguous student and teacher segments and label visited states."""
 def __init__(self,expert,config,seed,policy=None):
  self.expert=expert;self.config=config;self.policy=policy
  self.rng=np.random.default_rng(seed+991)
  self.noise=SmoothVelocityNoise(seed+177,config.get("noise_mm_s",2.)/1000.)
  self.pending=[];self.selected="expert";self.last_observation=None
  self.counts={k:0 for k in ("expert","dart","policy","recovery","hold")}
  self.interventions=0;self.was_recovering=False;self.teacher_override=False
  self.segment_blocks_left=0;self.next_segment="expert";self.segment_recovery=False
 def clear_policy_history(self):
  self.pending=[];self.last_observation=None
  if self.policy is not None:self.policy.clear_history()
 def begin_segment(self,kind,recovery=False):
  self.selected=kind
  self.segment_blocks_left=int(self.config.get("student_segment_blocks",8) if kind=="policy"
                               else self.config.get("expert_segment_blocks",24))
  self.next_segment="expert" if kind=="policy" else "policy"
  self.segment_recovery=bool(recovery);self.pending=[]
  if self.policy is not None:self.policy.clear_actions()
 def command(self,state,phase,jacobian,images=None,observation=None,observation_id=None,force_expert=False):
  label=self.expert.query(state)
  mode=self.config["method"]
  error=self.expert.tracking_error(state)
  forced=bool(force_expert or self.teacher_override)
  contact=any(word in phase for word in ("seat","entry","thread","grasp","release"))
  distance=self.config.get("contact_recovery_distance_m",.002) if contact else self.config.get("recovery_distance_m",.015)
  recovery=forced or error>distance
  if recovery and not self.was_recovering:self.interventions+=1
  self.was_recovering=recovery
  new_observation=observation_id!=self.last_observation and images is not None
  if mode=="dagger":
   if recovery and self.selected=="policy":self.begin_segment("expert",recovery=True)
   if new_observation:
    self.last_observation=observation_id
    if self.segment_blocks_left<=0:
     self.begin_segment("expert" if recovery else self.next_segment,recovery=recovery)
    self.pending=list(self.policy.act(images,observation))
    self.segment_blocks_left-=1
  desired=label.copy();source="expert"
  if mode=="dagger" and self.pending:
   candidate=self.pending.pop(0)
   if self.selected=="policy" and not recovery:
    width_error=abs(float(candidate[7:9].sum()-label[7:9].sum()))
    if width_error>self.config.get("gripper_recovery_difference_m",.005):
     self.begin_segment("expert",recovery=True)
     recovery=True;self.interventions+=int(not self.was_recovering);self.was_recovering=True
    else:desired=candidate;source="policy"
   elif self.segment_recovery:source="recovery"
  if recovery:source="recovery"
  elif mode=="dart":
   contact=any(word in phase for word in ("seat","entry","thread","grasp","release"))
   velocity=self.noise.sample(1/60,.1 if contact else 1.)
   j=np.asarray(jacobian)[:3,:7]
   dq=j.T@np.linalg.solve(j@j.T+1e-6*np.eye(3),velocity)
   desired[:7]+=dq/60;source="dart"
  executed=self.expert.limits.commit(desired)
  self.counts[source]+=1
  return label,executed,source
