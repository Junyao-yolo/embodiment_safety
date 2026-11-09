"""Sequential short-window gate for the frozen successful replay."""
from pathlib import Path
import subprocess,sys,json
ROOT=Path("/mnt/data2/wjy_SERL/Embodied_Safety")
HERE=Path(__file__).resolve().parent
OUT=ROOT/"outputs/runs/task1/diffusion_policy/performance_retest_v004"
def run(variant,name):
    rc=subprocess.run(["/usr/bin/python3","-B",str(HERE/"run.py"),"--variant",variant,"--steps","60","--name",name],cwd=ROOT).returncode
    if rc:raise RuntimeError(name+" did not complete")
    result=json.loads((OUT/name/"result.json").read_text())
    if result.get("failure_reason"):raise RuntimeError(str(result))
    return OUT/name
baseline=run("baseline","baseline_short_r2")
candidate=run("analytic","analytic_short")
PY=ROOT/"algos/diffusion_policy/.venv/bin/python"
subprocess.run([str(PY),"-B",str(HERE/"summarize.py"),str(baseline),str(candidate)],check=True,cwd=ROOT)
subprocess.run([str(PY),"-B",str(HERE/"compare.py"),str(baseline),str(candidate),str(OUT/"short_comparison.json")],check=True,cwd=ROOT)
subprocess.run([str(PY),"-B",str(HERE/"compare.py"),str(ROOT/"outputs/runs/task1/diffusion_policy/insertion_diagnosis_v001/recorded_full/eval/episode_01"),str(baseline),str(OUT/"baseline_reference_comparison.json")],check=True,cwd=ROOT)
(OUT/"short_gate.json").write_text(json.dumps(dict(passed=True,baseline=str(baseline),candidate=str(candidate)),indent=2))
