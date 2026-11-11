"""Build the experiment-local Python environment without changing system packages."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]

def main():
    spec=json.loads((ROOT/'env-spec.json').read_text())
    def state(phase,**extra):
        (ROOT/'setup_status.json').write_text(json.dumps(dict(phase=phase,time=time.time(),**extra),indent=2))
    state('building_environment')
    env=dict(os.environ,TMPDIR=str(ROOT/'tmp'),PIP_NO_CACHE_DIR='1',PIP_DISABLE_PIP_VERSION_CHECK='1')
    (ROOT/'tmp').mkdir(exist_ok=True)
    subprocess.run(['python3','-m','venv','--without-pip',str(ROOT/'.venv')],check=True,env=env)
    python=str(ROOT/'.venv/bin/python')
    subprocess.run(['python3','-m','pip','--python',python,'install','--index-url','https://pypi.tuna.tsinghua.edu.cn/simple','pip==25.1.1','setuptools==75.8.0','wheel'],check=True,env=env)
    for i,packages in enumerate(spec['pip_phases']):
        source='https://download.pytorch.org/whl/cu128' if i==0 else 'https://pypi.tuna.tsinghua.edu.cn/simple'
        subprocess.run([python,'-m','pip','install','--index-url',source,*packages],check=True,env=env)
    subprocess.run([python,'-m','pip','check'],check=True,env=env)
    (ROOT/'environment_installed.json').write_text(json.dumps(spec,indent=2))
    state('waiting_for_gpu_witness')

if __name__=='__main__':
    try:main()
    except Exception as e:
        (ROOT/'setup_status.json').write_text(json.dumps(dict(phase='failed',error=repr(e)),indent=2))
        raise
