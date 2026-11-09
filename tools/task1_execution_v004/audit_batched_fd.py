"""Offline equivalence and cost comparison at recorded Franka configurations."""
import sys,json,time,cProfile,pstats,io
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
for p in [HERE,Path("/mnt/data2/isaacsim-5.0.0/exts/isaacsim.robot_motion.lula/pip_prebundle"),Path("/mnt/data2/isaacsim-5.0.0/exts/omni.pip.compute/pip_prebundle")]:sys.path.insert(0,str(p))
import lula,numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from kinematics import PoseResidual
from batched_fd import jacobian
M=Path("/mnt/data2/isaacsim-5.0.0/exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/franka")
model=lula.load_robot(str(M/"rmpflow/robot_descriptor.yaml"),str(M/"lula_franka_gen.urdf"))
kin=model.kinematics()
OUT=ROOT/"outputs/runs/task1/diffusion_policy/performance_retest_v004"
rows=[json.loads(l) for l in (OUT/"cached_fd_full/ik_calls.jsonl").read_text().splitlines()]
def setup(row):
    q=np.array(row["measured_joint_positions"])
    pose=kin.pose(q.reshape(-1,1),"panda_hand")
    R=Rotation.from_quat(row["measured_ee_quaternion_xyzw"]).as_matrix()@pose.rotation.matrix().T
    p=np.array(row["measured_ee_position_m"])-R@pose.translation
    solver=SimpleNamespace(_kinematics=kin,_robot_rot=R,_robot_pos=p,_meters_per_unit=1.)
    return PoseResidual(solver,"panda_hand",np.array(row["commanded_position_m"]),Rotation.from_quat(row["commanded_quaternion_xyzw"]))
def scalar(problem,x):
    cols=[]
    for k in range(len(x)):
        hi=x.copy();lo=x.copy();hi[k]+=1e-5;lo[k]-=1e-5
        cols.append((problem(hi)-problem(lo))/2e-5)
    return np.column_stack(cols)
samples=[];times={"scalar":0.,"batched":0.}
for row in rows[::16]:
    x=np.array(row["candidate_joint_positions"])
    p=setup(row)
    st=time.perf_counter();a=scalar(p,x);times["scalar"]+=time.perf_counter()-st
    st=time.perf_counter();b=jacobian(p,x);times["batched"]+=time.perf_counter()-st
    samples.append({"sim_time":row["sim_time"],"exact_equal":bool(np.array_equal(a,b)),"max_error":float(np.max(np.abs(a-b)))})
solves=[]
if all(s["exact_equal"] for s in samples):
    for i in [0,min(200,len(rows)-1),min(600,len(rows)-1),min(1200,len(rows)-1),len(rows)-1]:
        row=rows[i];seed=np.array(row["measured_joint_positions"])
        results={};elapsed={}
        for name,method in [("scalar",scalar),("batched",jacobian)]:
            p=setup(row);start=time.perf_counter()
            r=least_squares(p,seed,bounds=(row["lower_limits_rad"],row["upper_limits_rad"]),jac=lambda x:method(p,x),max_nfev=400,xtol=1e-10,ftol=1e-10,gtol=1e-10)
            elapsed[name]=time.perf_counter()-start
            results[name]=r
        a,b=results["scalar"],results["batched"]
        solves.append({"index":i,"exact_solution":bool(np.array_equal(a.x,b.x)),"max_joint_difference":float(np.max(np.abs(a.x-b.x))),"nfev":[a.nfev,b.nfev],"njev":[a.njev,b.njev],"seconds":elapsed})
profile=cProfile.Profile()
p=setup(rows[0]);x=np.array(rows[0]["candidate_joint_positions"])
profile.enable()
for _ in range(100):scalar(p,x)
profile.disable()
s=io.StringIO();pstats.Stats(profile,stream=s).sort_stats("cumtime").print_stats(25)
(OUT/"scalar_fd_profile.txt").write_text(s.getvalue())
out={"jacobian_samples":len(samples),"all_jacobians_bitwise_equal":all(s["exact_equal"] for s in samples),"jacobian_seconds":times,"samples":samples,"solves":solves,"passed":all(s["exact_equal"] for s in samples) and bool(solves) and all(s["exact_solution"] and len(set(s["nfev"]))==1 and len(set(s["njev"]))==1 for s in solves),"scope":"Same recorded targets, inferred base transform from measured FK; offline timing excludes Isaac and is not full-replay speed."}
(OUT/"batched_fd_audit.json").write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!="samples"},indent=2))
