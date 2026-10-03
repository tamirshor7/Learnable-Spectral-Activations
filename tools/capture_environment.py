"""Run with each historical experiment interpreter to recover its software provenance."""
import argparse
import importlib
import json
import platform
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
r={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'modules':{}}
for name in ['torch','numpy','pandas','scipy','soundfile','librosa','soxr','numba','llvmlite','PIL','yaml','skimage']:
    try:
        m=importlib.import_module(name)
        r['modules'][name]={'version':getattr(m,'__version__','unknown')}
        if name=='torch':
            r['torch_runtime']={'cuda':m.version.cuda,'cudnn':m.backends.cudnn.version(),
              'tf32_matmul':m.backends.cuda.matmul.allow_tf32,'cudnn_benchmark':m.backends.cudnn.benchmark,
              'deterministic_algorithms':m.are_deterministic_algorithms_enabled()}
        if name=='soundfile':r['libsndfile']=m.__libsndfile_version__
    except Exception as e:r['modules'][name]={'error':repr(e)}
for label,command in [('pip_freeze',[sys.executable,'-m','pip','freeze']),('gpu',['nvidia-smi','--query-gpu=index,name,driver_version,uuid','--format=csv,noheader']),('conda_explicit',['conda','list','--explicit'])]:
    try:r[label]=subprocess.check_output(command,text=True,stderr=subprocess.STDOUT)
    except Exception as e:r[label]={'error':repr(e)}
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(json.dumps(r,indent=2)+'\n')
print(a.out.resolve())
