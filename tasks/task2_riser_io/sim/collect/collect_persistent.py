"""Collect Task2 episodes with one loaded scene per worker."""
from pathlib import Path
import gc,json,sys,time,shutil,traceback,argparse
R=Path(__file__).resolve().parents[2]
D=Path(__file__).resolve().parent
ARTIFACTS=Path("/mnt/data2/wjy_SERL/Embodied_Safety/outputs/task2_riser_io")
parser=argparse.ArgumentParser()
parser.add_argument("--collection-config",type=Path)
parser.add_argument("--additional-successes",type=int)
parser.add_argument("--successes",type=int)
parser.add_argument("--seed",type=int)
parser.add_argument("--max-attempts",type=int,default=200)
parser.add_argument("--output-root",type=Path)
parser.add_argument("--worker-id",type=int,default=0)
parser.add_argument("--num-workers",type=int,default=1)
opts=parser.parse_args()
if not 0<=opts.worker_id<opts.num_workers:parser.error("worker-id must be inside num-workers")
if opts.collection_config:opts.collection_config=opts.collection_config.resolve()
RUN=opts.output_root.resolve() if opts.output_root else ARTIFACTS
RAW=RUN/"raw" if opts.output_root else ARTIFACTS/"data/raw"
RUN.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True)
suffix=f"_worker_{opts.worker_id}" if opts.num_workers>1 else ""
PROGRESS=RUN/("progress"+suffix+".json");LOG=RUN/("collection"+suffix+".log")
KIT=RUN/("kit"+suffix+".log");RESET=RUN/("reset_requested"+suffix)
DIAGNOSTICS=RUN/"attempt_results";DIAGNOSTICS.mkdir(exist_ok=True)
def accepted():
    found=[]
    for p in sorted(RAW.glob("episode_*")):
        if not (p/"result.json").exists():continue
        try:result=json.loads((p/"result.json").read_text())
        except json.JSONDecodeError:continue
        owner=result.get("runtime_metadata",{}).get("worker_id",result.get("seed",0)%opts.num_workers)
        if result.get("accepted") and (opts.num_workers==1 or owner==opts.worker_id):found.append(p)
    return found
sys.argv=[sys.argv[0]]
done=accepted();starting_successes=len(done)
target=opts.successes if opts.successes is not None else (starting_successes+opts.additional_successes if opts.additional_successes is not None else json.loads((D/"collection.json").read_text())["target"])
prior=[int(p.name.split("_")[-1]) for p in done]
prior += [int(p.stem.split("_")[-1]) for p in DIAGNOSTICS.glob("collection_*.json")
          if int(p.stem.split("_")[-1])%opts.num_workers==opts.worker_id]
seed=max(opts.seed if opts.seed is not None else 6,max(prior,default=-opts.num_workers)+opts.num_workers)
seed+=(opts.worker_id-seed)%opts.num_workers
batch_started=time.perf_counter();failed=0;episode_index=0
state={"status":"STARTING","target_successes":target,"completed_successes":len(done),"seed":seed,"started_unix":time.time(),"mode":"persistent_stage","worker_id":opts.worker_id}
PROGRESS.write_text(json.dumps(state,indent=2))
from isaacsim import SimulationApp
app=SimulationApp({"headless":True,"limit_cpu_threads":4,"width":128,"height":128,"sync_loads":True,"extra_args":["--/log/file="+str(KIT)]})
engine_startup=time.perf_counter()-batch_started
scene_session={}
try:
    while len(done)<target and episode_index<opts.max_attempts:
        start=time.perf_counter()
        state={"status":"RUNNING","target_successes":target,"completed_successes":len(done),"seed":seed,"started_unix":time.time(),"mode":"persistent_stage","worker_id":opts.worker_id,"episode_index_in_process":episode_index,"engine_startup_wall_s":engine_startup}
        def progress(row,performance):
            state.update(row,performance=performance,wall_seconds=time.perf_counter()-start)
            PROGRESS.write_text(json.dumps(state,indent=2))
        progress({}, {})
        sys.argv=[str(R/"scripts/run_dp_bc.py"),"--seed",str(seed),"--record"]
        if opts.collection_config:sys.argv+=["--collection-config",str(opts.collection_config)]
        namespace={"__name__":"__main__","_simulation_app":app,"_progress_callback":progress,"_scene_session":scene_session,
                   "_episode_root":RAW,"_diagnostics_root":DIAGNOSTICS,"_kit_log":KIT,"_reset_request":RESET,
                   "_runtime_metadata":{"persistent":True,"worker_id":opts.worker_id,"num_workers":opts.num_workers,"episode_index_in_process":episode_index,"engine_startup_wall_s":engine_startup}}
        exec(compile((R/"scripts/run_dp_bc.py").read_text(),str(R/"scripts/run_dp_bc.py"),"exec"),namespace)
        result=namespace["report"]
        ok=bool(result.get("accepted") and (RAW/f"episode_{seed:04d}/result.json").exists())
        render_setup=namespace.get("setup_render")
        render={name:getattr(render_setup,name) for name in ("anns","products","paths","manifest")} if render_setup is not None and hasattr(render_setup,"anns") else {}
        if "sim" in namespace:
            namespace["sim"].stop()
            scene_session={"sim":namespace["sim"],"initial_transforms":namespace["initial_transforms"],"assembly_geometry":namespace["assembly_geometry"],"render":render}
        if "view" in namespace:namespace["view"].invalidate()
        namespace.clear();del render_setup;gc.collect()
        done=accepted();failed=0 if ok else failed+1
        if not ok:
            partial=RAW/f"episode_{seed:04d}"
            if partial.exists():shutil.rmtree(partial)
        mode_counts={}
        for episode in done:
            mode=json.loads((episode/"result.json").read_text()).get("route_mode","legacy")
            mode_counts[mode]=mode_counts.get(mode,0)+1
        state.update(completed_successes=len(done),route_success_counts=mode_counts,last_accepted=ok,last_error=result.get("error"),
                     last_attempt_wall_s=time.perf_counter()-start,batch_wall_s=time.perf_counter()-batch_started)
        with LOG.open("a") as log:log.write(json.dumps(state)+"\n")
        PROGRESS.write_text(json.dumps(state,indent=2))
        if failed>=8:
            state["status"]="NEEDS_REPAIR";break
        seed+=opts.num_workers;episode_index+=1
    else:state["status"]="COMPLETE" if len(done)>=target else "ATTEMPT_BUDGET_EXHAUSTED"
except Exception as exc:
    state.update(status="NEEDS_REPAIR",last_error=str(exc))
    traceback.print_exc();sys.stdout.flush();sys.stderr.flush()
finally:
    PROGRESS.write_text(json.dumps(state,indent=2))
    for ann in scene_session.get("render",{}).get("anns",{}).values():ann.detach()
    for product in scene_session.get("render",{}).get("products",[]):product.destroy()
    app.close()
if state["status"]!="COMPLETE":raise SystemExit(2)
