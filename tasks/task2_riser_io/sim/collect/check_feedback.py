"""Tests for feedback labels and bounded executed-action perturbations."""
import unittest
import numpy as np
from feedback import FeedbackExpert,JointCommandLimits,RecoveryRouter,SmoothVelocityNoise
def initial():return np.array([0.,-.5,0.,-2.,0.,1.7,.7,.03,.03])
def fk(q):return np.asarray(q[:3]),np.eye(3)
def solve(p,r,q):
 q=q.copy();q[:3]=p;return q
class FeedbackChecks(unittest.TestCase):
 def expert(self):
  q=initial();e=FeedbackExpert(solve,fk,JointCommandLimits(q))
  e.set_reference(q[:3],np.eye(3),.06)
  return q,e
 def test_labels_respond_to_measured_state(self):
  q,e=self.expert();state={"joints":q[None]}
  a=e.query(state);shift=q.copy();shift[0]+=.001
  b=e.query({"joints":shift[None]})
  self.assertLess(b[0],a[0])
  np.testing.assert_array_equal(e.limits.previous,q)
 def test_noise_is_correlated_and_acceleration_bounded(self):
  n=SmoothVelocityNoise(42);v=np.array([n.sample(1/60) for _ in range(600)])
  self.assertLessEqual(np.max(np.abs(np.diff(v,axis=0)))*60,.0150001)
  self.assertGreater(np.corrcoef(v[:-1,0],v[1:,0])[0,1],.8)
 def test_dart_does_not_change_labels(self):
  q,e=self.expert();router=RecoveryRouter(e,dict(method="dart",noise_mm_s=2.),42)
  state={"joints":q[None]};expected=e.query(state)
  j=np.zeros((6,7));j[:3,:3]=np.eye(3)
  label,execution,source=router.command(state,"transport",j)
  np.testing.assert_array_equal(label,expected)
  self.assertGreater(np.linalg.norm(execution-label),0)
  self.assertEqual(source,"dart")
 def test_policy_visit_then_expert_recovery(self):
  q,e=self.expert()
  class Policy:
   def act(self,images,state):
    a=np.tile(q,(6,1));a[:,0]+=.003;return a
  router=RecoveryRouter(e,dict(method="dagger",expert_probability=0.,recovery_distance_m=.015),42,Policy())
  j=np.zeros((6,7));j[:3,:3]=np.eye(3)
  label,execution,source=router.command({"joints":q[None]},"transport",j,{},q,1)
  self.assertEqual(source,"policy");self.assertGreater(execution[0],label[0])
  displaced=q.copy();displaced[0]+=.02
  label,execution,source=router.command({"joints":displaced[None]},"transport",j)
  self.assertEqual(source,"recovery");np.testing.assert_array_equal(label,execution)
  self.assertEqual(router.interventions,1)
 def test_executor_acceleration(self):
  q=initial();limits=JointCommandLimits(q);trace=[q.copy()]
  for sign in ([1.]*20+[-1.]*20+[0.]*20):
   target=q.copy();target[0]+=.5*sign;trace.append(limits.commit(target))
  velocity=np.diff(trace,axis=0)[:,:7]*60
  self.assertLessEqual(abs(velocity).max(),1.800001)
  self.assertLessEqual(abs(np.diff(velocity,axis=0)*60).max(),10.000001)
 def test_joint_limit_braking(self):
  from interface import JOINT_MIN,JOINT_MAX
  for sign in (-1.,1.):
   q=initial();q[6]=(JOINT_MAX[6]-.15) if sign>0 else (JOINT_MIN[6]+.15)
   limits=JointCommandLimits(q);trace=[q.copy()]
   target=q.copy();target[6]+=sign
   for _ in range(180):trace.append(limits.commit(target))
   trace=np.asarray(trace)
   velocity=np.diff(trace,axis=0)[:,:7]*60
   self.assertLessEqual(abs(np.diff(velocity,axis=0)*60).max(),10.00001)
   self.assertTrue(np.all(trace[:,:7]<=JOINT_MAX+1e-8))
   self.assertTrue(np.all(trace[:,:7]>=JOINT_MIN-1e-8))
if __name__=="__main__":unittest.main()
