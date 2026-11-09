"""Check joint preservation and fixture-relative acceptance across reset seeds."""
from pathlib import Path
import json
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics,Gf
from randomization import sample_reset,apply_reset,TARGET,capture_initial_transforms,restore_initial_transforms
R=Path(__file__).resolve().parents[2]
ARTIFACTS=Path("/mnt/data2/wjy_SERL/Embodied_Safety/outputs/task2_riser_io")
def anchors(stage):
 cache=UsdGeom.XformCache();out={}
 for p in stage.Traverse():
  if not p.IsA(UsdPhysics.Joint) or str(p.GetPath()).startswith("/World/Robot"):continue
  j=UsdPhysics.Joint(p);points=[]
  for side in (0,1):
   targets=p.GetRelationship("physics:body"+str(side)).GetTargets()
   point=Gf.Vec3d(*p.GetAttribute("physics:localPos"+str(side)).Get())
   if targets:point=cache.GetLocalToWorldTransform(stage.GetPrimAtPath(targets[0])).Transform(point)
   points.append(np.asarray(point))
  out[str(p.GetPath())]=np.linalg.norm(points[0]-points[1])
 return out
def main():
 s=Usd.Stage.Open(str(R/"scene/riser_io_success_v014.usda"))
 s.SetEditTarget(s.GetSessionLayer());initial=capture_initial_transforms(s)
 before=anchors(s);worst=0.;stage_identifier=s.GetRootLayer().identifier
 print("World children:",[str(p.GetPath()) for p in s.GetPrimAtPath("/World").GetChildren()])
 for seed in (100,104,108,112,116,120):
  restore_initial_transforms(s,initial)
  assert s.GetRootLayer().identifier==stage_identifier
  assert max(abs(before[k]-anchors(s)[k]) for k in before)<2e-6
  spec=sample_reset(seed);target,rot=apply_reset(s,spec)
  after=anchors(s);change=max(abs(before[k]-after[k]) for k in before)
  assert change<2e-6,(seed,change)
  assert np.linalg.norm(target-TARGET)>.0001
  assert np.allclose(rot.T@rot,np.eye(3))
  worst=max(worst,change)
 result=dict(passed=True,seeds=6,max_anchor_gap_change_m=worst,same_stage_reused=True,scope="authored transforms and in-place restoration; physics reset not yet tested")
 (ARTIFACTS/"implement-stage/randomization_check.json").write_text(json.dumps(result,indent=2));print(result)
if __name__=="__main__":main()
