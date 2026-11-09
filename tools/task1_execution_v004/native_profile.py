"""Short-window native CPU/GPU event capture using installed Kit APIs."""
import json
class NativeProfile:
    def __init__(self,out):
        import carb.profiler,carb.settings
        import omni.hydra.engine.stats
        self.settings=carb.settings.get_settings()
        self.old_gpu=self.settings.get("/profiler/enabled")
        self.cpu=carb.profiler.acquire_profiler_interface(plugin_name="carb.profiler-cpu.plugin")
        self.old_mask=self.cpu.get_capture_mask()
        self.monitor=carb.profiler.acquire_profile_monitor_interface()
        self.engine=omni.hydra.engine.stats.HydraEngineStats("","")
        self.settings.set_bool("/profiler/enabled",True)
        self.cpu.set_capture_mask(0xffffffffffffffff)
        self.file=(out/"native_profile.jsonl").open("x",buffering=1)
        self.frame=0
    def sample(self,sim_time,action_index):
        if action_index<=0:return
        events=self.monitor.get_last_profile_events()
        nodes=events.get_profile_events(events.get_main_thread_id())
        gpu=self.engine.get_gpu_profiler_result()
        self.file.write(json.dumps(dict(sample=self.frame,sim_time=sim_time,action_index=action_index,cpu_main_thread=nodes,gpu_devices=gpu,
            timing_note="Profiler node durations are milliseconds. Nested nodes overlap. GPU results are delayed; sample time is read time, not GPU submission time."),default=str)+"\n")
        self.frame+=1
    def close(self):
        self.cpu.set_capture_mask(self.old_mask)
        self.settings.set_bool("/profiler/enabled",bool(self.old_gpu))
        self.file.close()
