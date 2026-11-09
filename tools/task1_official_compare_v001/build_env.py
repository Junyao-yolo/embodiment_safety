"""Install the declared training stack into the experiment-local virtualenv."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
spec = json.loads(Path(__file__).with_name('env-spec-3090.json').read_text())
tmp = root/'tmp';tmp.mkdir(exist_ok=True)
env = dict(os.environ,TMPDIR=str(tmp),PIP_NO_CACHE_DIR='1',PIP_DISABLE_PIP_VERSION_CHECK='1')
venv = root/'.venv'
if not (venv/'bin/python').exists():
    subprocess.run([sys.executable,'-m','venv',str(venv)],env=env,check=True)
python = str(venv/'bin/python')
for index,packages in enumerate(spec['pip_phases']):
    source='https://download.pytorch.org/whl/cu128' if index==0 else 'https://pypi.tuna.tsinghua.edu.cn/simple'
    subprocess.run([python,'-m','pip','install','--index-url',source,*packages],env=env,check=True)
subprocess.run([python,'-m','pip','check'],env=env,check=True)
(root/'environment_installed.json').write_text(json.dumps(spec,indent=2))
