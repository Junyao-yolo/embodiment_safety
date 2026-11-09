"""Contract checks using distinguishable expert and executed commands."""
from pathlib import Path
import tempfile, unittest, json
import numpy as np
from interface import CAMERAS,save_episode,validate_episode

class CollectionContract(unittest.TestCase):
 def test_expert_label_and_previous_execution(self):
  records=[];frames=[];images=[]
  for k in range(300):
   q=np.array([0.,-.5,0.,-2.,0.,1.7,.7,.03,.03])
   q[0]=(k+1)*1e-5
   expert=q.copy();expert[0]+=.002
   records.append(dict(time_s=(k+1)/120,robot_joint_positions=q.tolist(),
    robot_joint_velocities=[0.]*9,robot_joint_targets=q.tolist(),
    expert_joint_targets=expert.tolist(),action_source="dart"))
   if (k+1)%12==0:
    frames.append(dict(time_s=(k+1)/120,robot_joints=q.tolist(),phase="transport"))
    images.append({c:np.zeros((128,128,3),np.uint8) for c in CAMERAS})
  with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
   p=Path(tmp)/"episode_0001"
   save_episode(p,records,frames,images,dict(accepted=True,physics_rate_Hz=120,collection_method="dart"),{})
   self.assertTrue(validate_episode(p)["passed"])
   x=np.load(p/"state.npy");u=np.load(p/"action.npy");v=np.load(p/"executed_action.npy")
   self.assertAlmostEqual(x[0,15],records[11]["robot_joint_targets"][0])
   self.assertAlmostEqual(u[0,0],records[12]["expert_joint_targets"][0])
   self.assertAlmostEqual(v[0,0],records[12]["robot_joint_targets"][0])
   self.assertAlmostEqual(float(u[0,0]-v[0,0]),.002)
   self.assertTrue(np.allclose(np.diff(np.load(p/"time.npy")),.1))
if __name__=="__main__":unittest.main()
