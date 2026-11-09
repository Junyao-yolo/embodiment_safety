"""Offline comparison against the original world-frame finite differences."""
import sys,json,time
from pathlib import Path
ROOT=Path("/mnt/data2/wjy_SERL/Embodied_Safety")
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,"/mnt/data2/isaacsim-5.0.0/exts/isaacsim.robot_motion.lula/pip_prebundle")
sys.path.insert(0,"/mnt/data2/isaacsim-5.0.0/exts/omni.pip.compute/pip_prebundle")
import lula,numpy as np
from scipy.spatial.transform import Rotation
from types import SimpleNamespace
from kinematics import PoseResidual
M=Path("/mnt/data2/isaacsim-5.0.0/exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/franka")
model=lula.load_robot(str(M/"rmpflow/robot_descriptor.yaml"),str(M/"lula_franka_gen.urdf"))
kin=model.kinematics()
r=ROOT/"outputs/runs/task1/diffusion_policy/insertion_diagnosis_v001/recorded_full/eval/episode_01"
files=sorted(r.glob("raw_*.json"),key=lambda p:int(p.stem.split("_")[-1]))[::10]
results=[]
for i,p in enumerate(files):
    q=np.asarray(json.loads(p.read_text())["joint_positions"][:7])
    baseR=Rotation.from_rotvec([.21,-.37,.16]).as_matrix()
    basep=np.array([.2,-.1,.8])
    solver=SimpleNamespace(_kinematics=kin,_robot_rot=baseR,_robot_pos=basep,_meters_per_unit=1.)
    pose=kin.pose(q.reshape(-1,1),"panda_hand")
    targetp=baseR@pose.translation+basep+np.array([.001,-.002,.003])
    targetR=Rotation.from_rotvec([.02,-.03,.01])*Rotation.from_matrix(baseR@pose.rotation.matrix())
    f=PoseResidual(solver,"panda_hand",targetp,targetR)
    def original(x):
        pose=kin.pose(x.reshape(-1,1),"panda_hand")
        dp=baseR@pose.translation+basep-targetp
        dr=(Rotation.from_matrix(baseR@pose.rotation.matrix())*targetR.inv()).as_rotvec()
        return np.concatenate((dp*1000,dr*100))
    cols=[]
    for k in range(7):
        hi=q.copy();lo=q.copy();hi[k]+=1e-5;lo[k]-=1e-5
        cols.append((original(hi)-original(lo))/2e-5)
    numeric=np.column_stack(cols)
    analytic=f.jacobian(q)
    results.append(dict(index=int(p.stem.split("_")[-1]),
        max_weighted_jacobian_error=float(np.max(np.abs(analytic-numeric))),
        max_residual_error=float(np.max(np.abs(original(q)-f(q))))))
out=dict(samples=len(results),max_weighted_jacobian_error=max(x["max_weighted_jacobian_error"] for x in results),
         max_residual_error=max(x["max_residual_error"] for x in results),rows=results)
out["passed"]=out["max_weighted_jacobian_error"]<=1e-5 and out["max_residual_error"]<=1e-10
dest=ROOT/"outputs/runs/task1/diffusion_policy/performance_retest_v004"
dest.mkdir(exist_ok=True)
(dest/"jacobian_audit.json").write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!="rows"},indent=2))
if not out["passed"]:raise SystemExit(1)
