"""CPU integration and original-versus-unified parity. Synthetic data, reduced budget."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.release import frozen_audio,getarg,setarg,write,sha,source_check


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,default=ROOT/'outputs/smoke')
    args=p.parse_args()
    out=args.output_root.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Smoke output must be empty. Choose a fresh --output-root.')
    out.mkdir(parents=True,exist_ok=True)
    source_check()
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    import numpy as np
    import soundfile as sf
    import torch
    from PIL import Image
    torch.set_num_threads(1)
    start=time.time()
    wav=out/'synthetic.wav'
    t=np.arange(16000,dtype=np.float64)/16000
    sf.write(wav,0.1+0.3*np.sin(2*np.pi*430*t)+0.07*np.sin(2*np.pi*710*t),16000,subtype='PCM_16')
    manifest=out/'synthetic.csv'
    with manifest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['clip_id','wav_path']);w.writeheader();w.writerow({'clip_id':'synthetic','wav_path':str(wav)})
    report={'kind':'synthetic_CPU_smoke_not_reference_reproduction','python':sys.version,'torch':torch.__version__,'checks':[]}
    env=os.environ.copy();env.update({'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    # Build an original tree with the supplied scripts, retaining identical baseline dependencies.
    original=out/'original_audio'
    shutil.copytree(ROOT/'experiments/audio',original,ignore=shutil.ignore_patterns('outputs','__pycache__'))
    for script in (ROOT/'provenance/original_audio_scripts').glob('*.py'):
        shutil.copy2(script,original/'scripts'/script.name)
    def strip_timing(d):
        return {k:v for k,v in d.items() if k not in ('wall_seconds','train_wall_clock_sec')}
    for spec in frozen_audio():
        argv=list(spec['argv'])
        for k,v in [('--manifest',manifest),('--device','cpu'),('--n_samples',96),('--max_steps',2),('--log_every',1),('--batch_size',16)]:
            setarg(argv,k,v)
        group=getarg(argv,'--group')
        results=[]
        for label,tree in [('original',original),('unified',ROOT/'experiments/audio')]:
            runout=out/label
            runenv=env.copy()
            if label=='unified':
                runenv['LSA_AUDIO_OUTPUT_ROOT']=str(runout)
            else:
                runenv.pop('LSA_AUDIO_OUTPUT_ROOT',None)
                runout=original/'outputs'
            log=out/f'{spec["name"]}_{label}.log'
            with log.open('w') as stream:
                subprocess.run([sys.executable,str(tree/'scripts'/spec['script']),*map(str,argv)],cwd=tree,env=runenv,stdout=stream,stderr=subprocess.STDOUT,check=True)
            summary=runout/'runs'/group/f'{group}_synthetic'/'summary.json'
            results.append((strip_timing(json.loads(summary.read_text())),summary.parent))
        if results[0][0]!=results[1][0]:
            raise AssertionError('Numerical parity failed: '+spec['name'])
        if spec['engine']=='core':
            old=torch.load(results[0][1]/'checkpoint_best.pt',weights_only=False)['model']
            new=torch.load(results[1][1]/'checkpoint_best.pt',weights_only=False)['model']
            if old.keys()!=new.keys() or not all(torch.equal(old[k],new[k]) for k in old):
                raise AssertionError('Checkpoint parity failed')
            if (results[0][1]/'convergence.csv').read_bytes()!=(results[1][1]/'convergence.csv').read_bytes():
                raise AssertionError('Trajectory parity failed')
        report['checks'].append({'name':spec['name'],'status':'PASS','psnr':results[1][0]['psnr'],'comparison':'exact summary except wall time, plus core checkpoint tensors and trajectory'})
        print('PASS audio parity '+spec['name'],flush=True)
    sys.path.insert(0,str(ROOT/'experiments/images'))
    from lsa.train import run
    import yaml
    image=out/'synthetic.png'
    rng=np.random.default_rng(1)
    Image.fromarray(rng.integers(0,256,(16,16,3),dtype=np.uint8)).save(image)
    core=yaml.safe_load((ROOT/'experiments/images/configs/kodak_core.yaml').read_text())
    conditions=[('lsa',{k:v for k,v in core['lsa'].items() if k!='candidates'}|{'lr':0.0005,'act_lr_mult':2.0}),('finer',core['finer']),('siren',core['siren']),
                ('staf',{'width':256,'hidden_layers':3,'first_omega':60.0,'hidden_omega':30.0,'tau':3,'lr':0.003,'schedule':'decay0p1'}),
                ('sl2a',{'width':256,'hidden_layers':3,'degree':256,'rank':128,'lr':0.004,'parameters':330243}),
                ('sl2a',{'width':256,'hidden_layers':3,'degree':192,'rank':64,'lr':0.004,'parameters':199171})]
    for index,(method,params) in enumerate(conditions):
        job={'method':method,'image':str(image),'out':str(out/'image'/str(index)),'params':params,
             'common':{'steps':2,'batch_size':16,'eval_every':1,'eval_chunk':64,'seed':0,'save_best_image':True}}
        result=run(job)
        if not np.isfinite(result['best_psnr']) or result['best_step']<1:
            raise AssertionError('Invalid image smoke result')
        report['checks'].append({'name':f'image_{method}_{index}','status':'PASS','psnr':result['best_psnr'],'parameters':result['num_params']})
        print('PASS image '+method,flush=True)
    report['elapsed_seconds']=time.time()-start
    report['status']='PASS'
    write(out/'smoke_report.json',report)
    print('SMOKE_STATUS=PASS')

if __name__=='__main__':
    main()
