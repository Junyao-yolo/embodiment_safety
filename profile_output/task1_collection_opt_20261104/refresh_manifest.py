"""Refresh the new-wrist manifest while the resumed collection is running."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path("/mnt/data2/wjy_SERL/Embodied_Safety")
sys.path.insert(0, str(ROOT))
from tools.task1_lab_parallel_v001.merge_manifests import merge

output = ROOT/"profile_output/task1_collection_opt_20261104"
collection = ROOT/"outputs/datasets/task1/parallel_diverse_v006"
sources = [ROOT/"outputs/datasets/task1/parallel_diverse_v005", collection]
pid = json.loads((collection/"run_config.json").read_text())["pid"]
previous_count = None
while True:
    summary = json.loads((collection/"summary.json").read_text())
    process = Path(f"/proc/{pid}/status")
    running = process.exists() and "State:\tZ" not in process.read_text()
    count = summary["accepted_count"]
    if count != previous_count or not running:
        combined = merge(sources, ROOT/"outputs/datasets/task1/parallel_diverse_new_wrist_v001")
        previous_count = count
        record = dict(status="running" if running else summary["status"], pid=pid,
                      accepted_v006=count, combined_accepted=combined["successful_episodes"], updated_at=time.time())
        path = output/"resume_status.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=2))
        tmp.replace(path)
        print(json.dumps(record), flush=True)
    if not running:
        break
    time.sleep(30)
