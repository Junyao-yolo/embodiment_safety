"""Queue fixed-checkpoint Task1 policy rollouts after the complete training budget."""
import argparse,collections,csv,json,os,signal,subprocess,time,shutil,sys
from pathlib import Path
from runners.common import ROOT
from runners.eval_diffusion import write_json,stop_group,label_summary,reconcile_result

def process_start(pid):
    p=Path(f"/proc/{pid}/stat")
    return p.read_text().split(") ",1)[1].split()[19] if p.exists() else None

def active_compute():
    result=subprocess.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True)
    active=[]
    for value in result.stdout.splitlines():
        if not value.strip().isdigit():continue
        pid=int(value);p=Path(f"/proc/{pid}/stat")
        if p.exists() and p.read_text().split(") ",1)[1].split()[0] not in ("T","t","Z"):active.append(pid)
    return active

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--training-run",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--episodes",type=int,default=50)
    ap.add_argument("--wall-timeout-s",type=int,default=2700)
    ap.add_argument("--goal-lead-horizon-s",type=float,help="must match the collection profile of the training data")
    ap.add_argument("--max-free-goal-lead-m",type=float)
    ap.add_argument("--ik-rotation-tolerance-rad",type=float)
    a=ap.parse_args()
    if (a.goal_lead_horizon_s is None)!=(a.max_free_goal_lead_m is None):raise ValueError("Give both goal-lead options or neither")
    lead={} if a.goal_lead_horizon_s is None else dict(TASK1_GOAL_LEAD_HORIZON_S=str(a.goal_lead_horizon_s),
                                                         TASK1_MAX_FREE_GOAL_LEAD_M=str(a.max_free_goal_lead_m))
    if a.ik_rotation_tolerance_rad is not None:
        lead["TASK1_IK_ROTATION_TOLERANCE_RAD"]=str(a.ik_rotation_tolerance_rad)
    train=(ROOT/a.training_run).resolve();out=(ROOT/a.output).resolve()
    if not out.is_relative_to(ROOT/"outputs") or a.episodes<1 or a.wall_timeout_s<1:raise ValueError("Invalid evaluation output/count/timeout")
    if out.exists():
        previous=json.loads((out/"summary.json").read_text())
        if previous["status"]!="waiting_for_training" or previous["episodes"]:
            raise ValueError("Only an unstarted waiting queue can be relaunched")
    else:out.mkdir(parents=True)
    py=ROOT/"algos/diffusion_policy/.venv/bin/python"
    actor=ROOT/"tools/task1_execution_v004/actor.py"
    render=ROOT/"tools/task1_execution_v004/render_replay.py"
    gate=json.loads((ROOT/"outputs/runs/task1/diffusion_policy/performance_retest_v004/controlled_execution_acceptance.json").read_text())
    if not gate["execution_equivalence_ready"]:raise RuntimeError("Execution prerequisite has not passed")
    state=dict(status="waiting_for_training",pid=os.getpid(),training_run=str(train),required_training_steps=100000,goal_lead=lead,
        requested=a.episodes,episodes=[],completed=0,wall_timeout_s=a.wall_timeout_s,execution_variant="batched_io",
        safety_mode="warn",diagnostic_only=True,network_updates=0,seed_base=2026110300,
        checkpoint_selection="Compare best.pt and last.pt EMA on all held-out validation windows after 100000 updates; minimum action MSE; fixed for the requested tests",
        created_at=time.time())
    def save():
        state["completed"]=len(state["episodes"])
        state["task_completions"]=sum(bool(x.get("task_completion_observed")) for x in state["episodes"])
        state["full_task_successes"]=sum(bool(x.get("full_task_success")) for x in state["episodes"])
        state["stage_counts"]=dict(collections.Counter(stage for x in state["episodes"] for stage in set(x.get("stages",[]))))
        state["failure_counts"]=dict(collections.Counter(x.get("failure_reason") or x.get("stage_failure") or "not_completed" for x in state["episodes"] if not x.get("task_completion_observed")))
        state["updated_at"]=time.time();write_json(out/"summary.json",state)
        with (out/"episodes.csv").open("w",newline="") as f:
            fields=["index","seed","task_completion_observed","full_task_success","failure_reason","stage_failure","reward","wall_s","video_ready","video"]
            w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(state["episodes"])
    save()
    try:
        while True:
            path=train/"train/result.json"
            result=json.loads(path.read_text()) if path.exists() else {}
            status=json.loads((train/"status.json").read_text())
            if status.get("phase")=="error":raise RuntimeError("Formal training failed; no test checkpoint selected")
            if result.get("status")=="finished":
                if result.get("steps")!=100000:raise RuntimeError("Training finished with a different budget")
                break
            state.update(training_status=result.get("status"),training_steps=result.get("steps"));save();time.sleep(30)
        meta=json.loads((train/"train/last.json").read_text())
        if meta["training_steps"]!=100000 or meta["config"]["batch_size"]!=16:raise RuntimeError("Checkpoint budget or batch mismatch")
        while active_compute():
            state.update(status="waiting_for_validation_gpu");save();time.sleep(30)
        state.update(status="comparing_checkpoints",training_steps=100000);save()
        comparison_dir=out/"model_selection"
        with (out/"model_selection.log").open("w") as log:
            subprocess.run([str(py),"-B","-m","runners.compare_diffusion_checkpoints",
                "--training-run",str(train),"--output",str(comparison_dir)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        comparison=json.loads((comparison_dir/"validation_comparison.json").read_text())
        if comparison["status"]!="finished":raise RuntimeError("Checkpoint comparison did not finish")
        name=comparison["selected"]
        checkpoint_dir=out/"checkpoint";checkpoint_dir.mkdir()
        for suffix in (".pt",".json"):shutil.copy2(train/"train"/(name+suffix),checkpoint_dir/(name+suffix))
        cp=checkpoint_dir/(name+".pt")
        state['training_completed_at']=json.loads((train/'status.json').read_text()).get('finished_at')
        state.update(checkpoint=str(cp),selected_model=name,model_comparison=str(comparison_dir/"validation_comparison.json"),training_steps=100000)
        for index in range(1,a.episodes+1):
            while True:
                active=active_compute()
                if not active:break
                state.update(status="waiting_for_gpu",active_compute=active,next_episode=index);save();time.sleep(30)
            seed=state["seed_base"]+index
            base=out/f"episode_{index:02d}";dest=base/"eval/episode_01";base.mkdir();dest.parent.mkdir()
            env=os.environ.copy()
            for name in ("TASK1_DIAGNOSTIC_STOP_STAGE","TASK1_WINDOW_GATES","TASK1_NATIVE_PROFILE",
                         "TASK1_GOAL_LEAD_HORIZON_S","TASK1_MAX_FREE_GOAL_LEAD_M"):env.pop(name,None)
            env.update(lead)
            env.update(CUDA_VISIBLE_DEVICES="0",TASK1_POLICY_BACKEND="diffusion_policy",TASK1_EVAL="1",
                TASK1_ONLINE_LEARN="0",TASK1_MAX_STEPS="0",TASK1_SMOKE="0",TASK1_EPISODE_SEED=str(seed),
                TASK1_SAFETY_MODE="warn",SERL_INPUT_CHECKPOINT=str(cp),TASK1_EXECUTION_VARIANT="batched_io")
            cmd=["/mnt/data2/isaacsim-5.0.0/python.sh","-B",str(actor),str(dest)]
            state.update(status="evaluating",current_episode=index,current_seed=seed,current_output=str(base),command=cmd);save()
            started=time.monotonic();timed_out=False
            with (base/"actor.log").open("w") as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state["actor_pid"]=proc.pid;save()
                try:rc=proc.wait(timeout=a.wall_timeout_s)
                except subprocess.TimeoutExpired:timed_out=True;stop_group(proc);rc=proc.returncode
            row=dict(index=index,seed=seed,output=str(base),wall_s=time.monotonic()-started,returncode=rc,
                wall_timeout=timed_out,task_completion_observed=False,full_task_success=False,completed=True)
            result_path=dest/"result.json"
            if timed_out:
                row.update(failure_reason="wall_time_limit",truncated=True,execution_completed=False)
            elif result_path.exists():
                result=json.loads(result_path.read_text())
                if not result.get("source_unchanged") or result.get("network_updates")!=0:raise RuntimeError("Evaluation mutated source or updated network")
                row.update(result)
            else:
                errors=[]
                for name in ("actor_error.json","worker_error.json"):
                    path=dest/name
                    if path.exists():errors.append(json.loads(path.read_text()))
                row.update(failure_reason="execution_error",execution_completed=False,errors=errors)
            row=reconcile_result(row,label_summary(dest))
            raw=sorted(dest.glob('raw_*.json'),key=lambda p:int(p.stem.rsplit('_',1)[1]))
            if raw:
                first=json.loads(raw[0].read_text())['sim_time']
                last=json.loads(raw[-1].read_text())['sim_time']
                row['simulation_elapsed_s']=last-first

            if not row.get("failure_reason") and not row.get("task_completion_observed"):row["failure_reason"]=row.get("stage_failure") or "task_not_completed"
            write_json(dest.parent/"summary.json",row)
            state.update(status="rendering",current_episode=index);save()
            with (base/"render.log").open("w") as log:
                rendered=subprocess.run([str(py),"-B",str(render),str(base)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            video=base/"replay/actual_policy_replay.mp4"
            row.update(video_returncode=rendered.returncode,video_ready=rendered.returncode==0 and video.exists(),video=str(video))
            write_json(base/"result.json",row);state["episodes"].append(row);save()
            # Record policy failures and continue; stop on an unusable infrastructure.
            if row.get("failure_reason")=="execution_error":
                text=json.dumps(row.get("errors",[]))
                if "controller_fault:" not in text and "controller_disarmed:" not in text:
                    raise RuntimeError("Infrastructure error: "+str(base/"actor.log"))
        state.update(status="finished",task_completion_rate=state["task_completions"]/a.episodes,
                     full_task_success_rate=state["full_task_successes"]/a.episodes,finished_at=time.time())
        save()
        print(json.dumps({k:v for k,v in state.items() if k!="episodes"},indent=2),flush=True)
    except BaseException as exc:
        state.update(status="error",error=repr(exc));save();raise

if __name__=="__main__":main()
