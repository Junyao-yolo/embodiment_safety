"""Disjoint main-thread interval attribution plus asynchronous GPU samples."""
import json,collections,statistics
from pathlib import Path
ROOT=Path("/mnt/data2/wjy_SERL/Embodied_Safety")
OUT=ROOT/"outputs/runs/task1/diffusion_policy/performance_retest_v004"
rows=[json.loads(l) for l in (OUT/"native_profile_short/native_profile.jsonl").read_text().splitlines()]
totals=collections.Counter();wait_inside=0.;gpu=[];gpu_names=collections.defaultdict(list)
seen=set();physics_counts=0;render_pass_counts=collections.Counter()
for row in rows:
    ns=row["cpu_main_thread"]
    roots=[n for n in ns if n["name"]=="App Main loop"]
    if len(roots)!=1:raise ValueError("Expected one main-thread frame")
    root=roots[0]
    if root["start_time"] in seen:continue
    seen.add(root["start_time"])
    events=[]
    for n in ns:
        name=n["name"]
        g=None
        if name=="App Main loop":g="root"
        elif name=="post-step update subscription update":g="post_step_callbacks"
        elif name=="FlushBatchedContactReport":g="contact_report_callbacks"
        elif name=="Thread waiting...":g="thread_wait"
        elif name=="PhysXUpdateNonRender":g="physics_scope"
        elif name in ("UsdContext hydraRender","UsdContext isRenderFinished","SceneRenderer-rtx: render"):g="render_scope"
        if g is not None:
            a=n["start_time"];b=a+round(n["duration"]*100000.)
            events.extend([(a,g,1),(b,g,-1)])
    events.sort()
    active=collections.Counter();prev=events[0][0]
    for t,g,d in events:
        if active["root"]:
            dt=(t-prev)/100000.
            if active["post_step_callbacks"]:label="post_step_callbacks"
            elif active["contact_report_callbacks"]:label="contact_report_callbacks"
            elif active["thread_wait"]:
                label="thread_wait"
                if active["physics_scope"]:wait_inside+=dt
            elif active["physics_scope"]:label="physics_scope_other_host_work"
            elif active["render_scope"]:label="render_scope_other_host_work"
            else:label="other_main_thread"
            totals[label]+=dt
        active[g]+=d;prev=t
    gr=collections.Counter()
    for dev in row["gpu_devices"]:
        for n in dev:
            if n["indent"]==0:gr[n["name"]]+=n["duration"]
            if n["indent"]==1:render_pass_counts[n["name"]]+=1
    gpu.append(sum(gr.values()))
    for name,v in gr.items():gpu_names[name].append(v)
out=dict(unique_cpu_frames=len(seen),sample_count=len(rows),main_thread_exclusive_ms=dict(totals),
    main_thread_total_ms=sum(totals.values()),wait_inside_physics_scope_ms=wait_inside,
    gpu_root_sample_ms=dict(mean=statistics.mean(gpu),median=statistics.median(gpu),maximum=max(gpu)),
    gpu_root_names_mean_ms={k:statistics.mean(v) for k,v in gpu_names.items()},
    gpu_pass_counts=dict(render_pass_counts),
    source_header="/mnt/data2/isaacsim-5.0.0/kit/dev/include/carb/profiler/IProfileMonitor.h",
    timestamp_unit="start_time:10 ns;duration:ms",
    semantics="CPU categories partition main-thread frame intervals. PhysX host scope includes asynchronous worker/GPU waits; neither these waits nor total PhysXUpdate are pure solver compute. GPU samples are asynchronous, roots only; never added to CPU totals. Per-camera GPU mapping unavailable in returned node names.",
    performance_scope="Detailed 10-action profiling run, not a lightweight benchmark.")
(OUT/"native_profile_summary.json").write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
