"""Exclusive CPU wall timing. Native Kit time is not attributed to PhysX alone."""
import time,json,functools
from collections import defaultdict
from contextlib import contextmanager
class Timings:
    def __init__(self,out):
        self.out=out;self.stack=["unattributed"];self.last=time.perf_counter()
        self.seconds=defaultdict(float);self.counts=defaultdict(int)
        self.previous={};self.previous_counts={};self.previous_wall=self.last
        self.file=(out/"cycle_timing.jsonl").open("x",buffering=1)
        self.ik_file=(out/"ik_calls.jsonl").open("x",buffering=1)
    def charge(self):
        now=time.perf_counter();self.seconds[self.stack[-1]]+=now-self.last;self.last=now
    @contextmanager
    def span(self,name):
        self.charge();self.stack.append(name);self.counts[name]+=1
        try:yield
        finally:self.charge();self.stack.pop()
    def wrap(self,obj,name,category):
        original=getattr(obj,name)
        @functools.wraps(original)
        def measured(*args,**kwargs):
            with self.span(category):return original(*args,**kwargs)
        setattr(obj,name,measured)
    def boundary(self,index,sim_time,metrics):
        self.charge();now=self.last
        row=dict(index=index,sim_time=sim_time,wall_s=now-self.previous_wall,
            exclusive_s={k:v-self.previous.get(k,0.) for k,v in self.seconds.items()},
            counts={k:v-self.previous_counts.get(k,0) for k,v in self.counts.items()},progress=metrics)
        self.previous=dict(self.seconds);self.previous_counts=dict(self.counts);self.previous_wall=now
        self.file.write(json.dumps(row,allow_nan=False)+"\n")
    def install(self,runtime,vision,np,Episode):
        from PIL import Image
        for name,cat in [("physics_step","physics_callback"),("on_contacts","contact_report"),("measure_contacts","contact_metrics"),("observe","state_and_image_snapshot"),("update","camera_update")]:
            self.wrap(runtime.CollectionPanel,name,cat)
        self.wrap(Image.Image,"save","image_encode_write")
        self.wrap(vision,"load_observation","image_read_decode")
        self.wrap(np,"savez_compressed","observation_serialize_write")
        self.wrap(np,"savez","observation_serialize_write")
        self.wrap(Episode,"physics","physics_log_serialize_write")
        self.wrap(Episode,"append","transition_serialize_write")
    def attach(self,p):
        original=p.controller.tick
        def tick(dt):
            previous=p.controller.ik_diagnostics
            with self.span("controller_tick"):
                start=time.perf_counter();out=original(dt);elapsed=time.perf_counter()-start
            if p.controller.ik_diagnostics is not previous:
                self.counts["ik_calls"]+=1
                d=p.controller.ik_diagnostics
                self.ik_file.write(json.dumps(dict(sim_time=p.t,tick_wall_s=elapsed,**d))+"\n")
            return out
        p.controller.tick=tick
        for name,cam in p.cameras.items():
            self.wrap(cam,"get_current_frame","camera_fetch_"+name)
    def close(self):
        self.charge();self.file.close();self.ik_file.close()
        (self.out/"timing_semantics.json").write_text(json.dumps(dict(
            cpu_wall_exclusive=True,gpu_synchronization_added=False,
            native_physics_gpu_s=None,task_render_gpu_s=None,other_render_gpu_s=None,
            note="Kit native update contains physics, rendering and driver waits; Python callbacks and camera fetch are measured separately. GPU render/physics are not separable with these CPU timers.",
            optimizer_iterations_note="TRF returns its existing iteration counter; nfev/njev, residual and retry counts are recorded per solve."),indent=2))
