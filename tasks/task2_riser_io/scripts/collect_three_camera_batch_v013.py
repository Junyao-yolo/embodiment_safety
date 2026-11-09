from pathlib import Path
import sys,gc,contextlib
R=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io')
out=R/'scene/three_camera_capture_v013'
sys.argv=[sys.argv[0]]
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'limit_cpu_threads':4,'width':640,'height':480,'sync_loads':True})
import omni.usd
from isaacsim.core.api import SimulationContext
source=(R/'scripts/collect_three_camera_v013.py').read_text()
line=next(x for x in source.splitlines() if ' app=SimulationApp(' in x)
source=source.replace(line,' app=shared_app').replace(' if app is not None:app.close()','')
cases=[('reference','baseline_full_003',False),('screw_reference','screw_reference_003',True),('screw_tilt','screw_tilt_003',True),('screw_axial','screw_axial_003',True)]
for case,name,resume in cases:
 if (out/name/'result.json').exists():continue
 print('BATCH_BEGIN',name,flush=True)
 sys.argv=['collect_three_camera_v013.py','--name',name,'--case',case,'--record']+(['--resume'] if resume else [])
 ns={'__name__':'__main__','shared_app':app}
 with (out/(name+'.log')).open('w') as log,contextlib.redirect_stdout(log):
  exec(compile(source,'collect_three_camera_v013.py','exec'),ns)
 if 'sim' in ns:ns['sim'].stop()
 if 'setup_render' in ns and hasattr(ns['setup_render'],'anns'):
  for ann in ns['setup_render'].anns.values():ann.detach()
 omni.usd.get_context().close_stage()
 SimulationContext.clear_instance()
 del ns
 gc.collect()
 for _ in range(5):app.update()
 print('BATCH_END',name,flush=True)
app.close()
