"""Check task RGB against the successful replay at fixed observation indices."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path("/mnt/data2/wjy_SERL/Embodied_Safety")
BASE=ROOT/"outputs/runs/task1/diffusion_policy/insertion_diagnosis_v001/recorded_full/eval/episode_01"
def check(path,output):
    ge=next(BASE.glob("episodes/*/transitions.jsonl")).parent
    pe=next(path.glob("episodes/*/transitions.jsonl")).parent
    limit=json.loads((Path(__file__).parent/"contract.json").read_text())["camera_mae_limit_8bit"]
    rows=[]
    for i in [0,10,20,30,40,50,60,490,870,1100,1349]:
        p=path/f"raw_{i:03d}.json"
        if not p.exists():continue
        a=json.loads((BASE/f"raw_{i:03d}.json").read_text());b=json.loads(p.read_text())
        for cam in a["images"]:
            aa=np.asarray(Image.open(ge/a["images"][cam]["file"])).astype(float)
            bb=np.asarray(Image.open(pe/b["images"][cam]["file"])).astype(float)
            if aa.shape!=bb.shape:raise ValueError("Task camera shape changed")
            rows.append(dict(index=i,camera=cam,mae=float(np.abs(aa-bb).mean()),shape=list(aa.shape)))
    out=dict(limit_mae_8bit=limit,rows=rows,maximum_mae=max(x["mae"] for x in rows),passed=all(x["mae"]<=limit for x in rows),
        basis="Preregistered 0.5 8-bit intensity MAE ceiling before batched_io tests; identical-configuration repeats at initial seven states measured max 0.2044. This is an observation regression gate, not a policy-success criterion.")
    output.write_text(json.dumps(out,indent=2))
    print(json.dumps({k:v for k,v in out.items() if k!="rows"},indent=2))
    return out["passed"]
if __name__=="__main__":
    raise SystemExit(0 if check(Path(sys.argv[1]),Path(sys.argv[2])) else 1)
