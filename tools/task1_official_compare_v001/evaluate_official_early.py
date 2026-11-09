"""Run the official checkpoint's twenty paired Task1 evaluations."""
import json, os, runpy, shlex, subprocess, time
from pathlib import Path

ROOT = Path("/mnt/data2/wjy_SERL/Embodied_Safety")
TOOLS = ROOT / "tools/task1_official_compare_v001"
m = runpy.run_path(str(TOOLS / "lifecycle_3090.py"))
OUT, PYTHON = m["OUT"], m["PYTHON"]
TARGET = OUT / "official_eval20"
CHECKPOINT = OUT / "outputs/official/checkpoints/best.pt"
IMPORTED = OUT / "official_workspace_eval.ckpt"

def status(phase, **values):
    data = dict(status=phase, pid=os.getpid(), updated_at=time.time(), **values)
    (OUT / "official_eval_launch.json").write_text(json.dumps(data, indent=2))
    print(json.dumps(data), flush=True)

def main():
    status("syncing_checkpoint")
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    with (OUT / "official_checkpoint_transfer.log").open("a") as log:
        subprocess.run(["rsync", "-a", "--partial", "--info=progress2", "-e", shlex.join(m["SSH"]),
                        m["HOST"] + ":" + m["REMOTE"] + "/outputs/official/checkpoints/best.pt",
                        str(CHECKPOINT)], stdout=log, stderr=subprocess.STDOUT, check=True)
    status("importing_checkpoint")
    with (OUT / "official_eval20.log").open("a") as log:
        if not IMPORTED.exists():
            subprocess.run([str(PYTHON), "-B", "-m", "learning.import_checkpoint",
                            "--source", str(CHECKPOINT), "--task", "task1_official_image",
                            "--output", str(IMPORTED)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        while subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                      "--format=csv,noheader,nounits"], text=True).strip():
            status("waiting_for_gpu")
            time.sleep(30)
        status("evaluating", requested=20, checkpoint=str(CHECKPOINT), weights="ema")
        subprocess.run([str(PYTHON), "-u", str(ROOT / "eval.py"), "--checkpoint", str(IMPORTED),
                        "--output-dir", str(TARGET), "--device", "cuda:0",
                        "task.env_runner.n_test=20", "task.env_runner.test_start_seed=2026110301",
                        "task.env_runner.max_steps=3000", "task.env_runner.wall_seconds=2700",
                        "task.env_runner.safety_mode=warn", "n_action_steps=8"],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    rollout = sorted((TARGET / "rollouts").glob("round_*/summary.json"))[-1]
    report = json.loads(rollout.read_text())
    assert report["completed"] == 20
    report.update(checkpoint=str(CHECKPOINT), weights="ema", framework="learning_workspace",
                  inference_checkpoint=str(IMPORTED), rollout_summary=str(rollout),
                  test_start_seed=2026110301)
    (TARGET / "summary.json").write_text(json.dumps(report, indent=2))
    status("complete", completed=report["completed"], task_success_rate=report["task_success_rate"])

if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        status("failed", error=repr(error))
        raise
