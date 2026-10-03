"""Compare decoded/preprocessed targets across historical and unified environments."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.release import load_module,manifest_rows,environment,sha,write


def array_info(value):
    import numpy as np
    value=np.asarray(value,dtype=np.float32).reshape(-1)
    return {'float32_sha256':hashlib.sha256(value.tobytes()).hexdigest(),'n':len(value),
            'mean':float(value.mean()),'peak':float(abs(value).max()),'rms':float(np.sqrt(np.mean(value**2)))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args()
    import pandas as pd
    scripts=ROOT/'experiments/audio/scripts'
    core=load_module('audio_core_fingerprint',scripts/'run_core_audio.py')
    fjnb=load_module('audio_fjnb_fingerprint',scripts/'run_fjnb_audio.py')
    sl2a=load_module('audio_sl2a_fingerprint',scripts/'run_sl2a_audio.py')
    result={'environment':environment(),'samples':[]}
    for dataset in ('nsynth','librispeech'):
        for row in manifest_rows(dataset):
            path=args.data_root.resolve()/Path(row['wav_path']).relative_to('data')
            x,y=core.load_audio(str(path),48000,48000)
            f=fjnb.read_audio(path,48000,48000)
            s=sl2a.load_audio(pd.Series({'wav_path':str(path)}),48000,48000,'zero_mean_unit_peak').numpy()
            result['samples'].append({'dataset':dataset,'sample_id':row['clip_id'],'source_sha256':sha(path),
                'core_coordinates':array_info(x),'core_target':array_info(y),'fjnb_target':array_info(f),'sl2a_target':array_info(s)})
    write(args.out,result)
    print(f'AUDIO_FINGERPRINT_COMPLETE samples={len(result["samples"])} path={args.out.resolve()}')

if __name__=='__main__':
    main()
