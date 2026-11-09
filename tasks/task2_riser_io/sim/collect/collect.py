"""Launch Task2 collection with a persistent Isaac Sim process."""
from pathlib import Path
import subprocess,sys
TASK=Path(__file__).resolve().parents[2]
COLLECT=Path(__file__).resolve().parent
log_path=Path("/mnt/data2/wjy_SERL/Embodied_Safety/outputs/task2_riser_io/logs/worker.log")
log_path.parent.mkdir(parents=True,exist_ok=True)
with log_path.open("w") as log:
    result=subprocess.run(["/mnt/data2/isaacsim-5.0.0/python.sh","-B",str(COLLECT/"collect_persistent.py"),*sys.argv[1:]],cwd=TASK,stdout=log,stderr=subprocess.STDOUT)
raise SystemExit(result.returncode)
