"""Shared release orchestration. Numerical kernels remain in experiments/."""
import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / 'experiments/images'
AUDIO = ROOT / 'experiments/audio'
SUITES = ('audio', 'image-core', 'image-staf', 'image-sl2a')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def getarg(argv, key):
    return argv[argv.index(key) + 1]


def setarg(argv, key, value):
    argv[argv.index(key) + 1] = str(value)


def frozen_audio():
    return read(ROOT / 'configs/audio_jobs.json')


def source_check():
    bad = [r['release_path'] for r in read(ROOT / 'provenance/source_files.json')
           if not (ROOT/r['release_path']).is_file() or sha(ROOT/r['release_path']) != r['release_sha256']]
    if bad:
        raise RuntimeError('Authoritative source files changed: ' + ', '.join(bad))


def code_state():
    files = [ROOT/name for name in ('reproduce.py','requirements.txt','environment.yml','environment-cpu.yml')]
    for base in ('tools','configs','experiments','provenance','environments'):
        files.extend(p for p in sorted((ROOT/base).rglob('*'))
                     if p.is_file() and p.suffix in ('.py','.yaml','.json','.csv','.sh','.txt','.in')
                     and not any(x in p.parts for x in ('__pycache__','outputs','data')))
    return {str(p.relative_to(ROOT)):sha(p) for p in files}


def environment(require_cuda=False):
    import importlib.metadata as md
    import torch
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Full reference runs require a GPU. Use the smoke test for CPU validation.')
    packages = {d.metadata['Name']: d.version for d in md.distributions()}
    try:
        driver = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,driver_version,uuid', '--format=csv,noheader'], text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        driver = None
    return {'python': platform.python_version(), 'platform': platform.platform(), 'packages': packages,
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(), 'gpu_inventory': driver,
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}


def manifest_rows(dataset):
    name = 'nsynth_pilot10_manifest.csv' if dataset == 'nsynth' else 'librispeech_pilot40_manifest.csv'
    with (AUDIO/'manifests'/name).open() as f:
        return list(csv.DictReader(f))


def dataset_files(suites, data):
    result = {}
    if any(s.startswith('image-') for s in suites):
        for name, expected in read(IMAGE/'reference/kodak_sha256.json').items():
            path = data/'kodak'/name
            if not path.is_file():
                raise RuntimeError(f'Missing Kodak file: {path}')
            observed = sha(path)
            if observed != expected:
                raise RuntimeError(f'Kodak SHA-256 mismatch: {name}')
            result[str(path)] = observed
    if 'audio' in suites:
        import soundfile as sf
        for dataset in ('nsynth', 'librispeech'):
            rows = manifest_rows(dataset)
            ids = [r['clip_id'] for r in rows]
            if len(ids) != (10 if dataset == 'nsynth' else 40) or len(set(ids)) != len(ids):
                raise RuntimeError('Audio manifest identity/count error')
            for row in rows:
                relative = Path(row['wav_path']).relative_to('data')
                path = data/relative
                if not path.is_file():
                    raise RuntimeError(f'Missing audio file: {path}')
                info = sf.info(path)
                # Authoritative manifests refer to mono, 16 kHz source recordings.
                if info.channels != 1 or info.samplerate != 16000 or info.frames <= 0:
                    raise RuntimeError(f'Unexpected source audio format: {path}: {info}')
                result[str(path)] = sha(path)
    return result


def image_config(suite, data, out):
    import yaml
    config = yaml.safe_load((IMAGE/'configs'/f'kodak_{suite}.yaml').read_text())
    config['data_root'] = data/'kodak'
    config['output_root'] = out/'images'/f'kodak_{suite}'
    return config


def image_jobs(suite, data, out, phase='initial'):
    runner = load_module('image_runner', IMAGE/'scripts/run.py')
    c = image_config(suite, data, out)
    if suite == 'core':
        jobs = runner.core_jobs(c)
    elif suite == 'staf':
        jobs = runner.staf_jobs(c)
    elif phase == 'initial':
        jobs = runner.sl2a_calibration_jobs(c)
    else:
        selected = runner.select_sl2a_learning_rates(c)
        jobs = runner.sl2a_final_jobs(c, selected)
    return [{'name': f'image-{suite}__{j["name"]}', 'kind': 'image', 'suite': f'image-{suite}',
             'job': j, 'destination': j['out']} for j in jobs]


def audio_jobs(data, out):
    result = []
    for spec in frozen_audio():
        argv = list(spec['argv'])
        dataset = getarg(argv, '--dataset')
        path = out/'manifests'/f'{dataset}.csv'
        rows = manifest_rows(dataset)
        for row in rows:
            row['wav_path'] = str(data/Path(row['wav_path']).relative_to('data'))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        setarg(argv, '--manifest', path)
        group = getarg(argv, '--group')
        result.append({'name': 'audio__'+spec['name'], 'kind': 'audio', 'suite': 'audio',
                       'script': spec['script'], 'argv': argv, 'group': group,
                       'ids': [r['clip_id'] for r in rows],
                       'destination': str(out/'audio/runs'/group)})
    return result


def validate_job(job):
    directory = Path(job['destination'])
    paths = sorted(directory.rglob('summary.json'))
    if job['kind'] == 'image':
        if paths != [directory/'summary.json']:
            raise RuntimeError('Image summary missing or ambiguous')
        d = read(paths[0])
        spec = job['job']
        for key in ('method', 'params', 'common'):
            if d[key] != spec[key]:
                raise RuntimeError('Image summary configuration mismatch: '+key)
        if d['image'] != Path(spec['image']).stem:
            raise RuntimeError('Image identity mismatch')
        vals = [d['best_psnr'], d['best_ssim']]
        if not 1 <= d['best_step'] <= spec['common']['steps']:
            raise RuntimeError('Invalid image result')
        if not (directory/'metrics.csv').is_file():
            raise RuntimeError('Image trajectory missing')
    else:
        rows = [read(p) for p in paths]
        ids = [r['sample_id'] for r in rows]
        if len(ids) != len(job['ids']) or set(ids) != set(job['ids']):
            raise RuntimeError('Audio identity/count mismatch')
        vals = [r['psnr'] for r in rows]
        for row in rows:
            if row.get('status', 'ok') != 'ok' or row.get('best_step', -1) < 0:
                raise RuntimeError('Invalid audio result')
    if not all(math.isfinite(float(v)) for v in vals):
        raise RuntimeError('Nonfinite result')
    # Sentinel PSNRs used by historical runners must never pass a completeness check.
    if min(vals) < -1000:
        raise RuntimeError('Runner never recorded a valid evaluated checkpoint')
    files = {str(p): sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}
    if job['kind'] == 'audio':
        aggregate = directory.parents[1]/'summaries'/f'{job["group"]}.csv'
        if not aggregate.is_file():
            raise RuntimeError('Audio aggregate missing')
        files[str(aggregate)] = sha(aggregate)
    return files


def receipt_path(out, job):
    return out/'receipts'/f'{job["name"]}.json'


def completed(out, job, run_signature):
    p = receipt_path(out, job)
    if not p.is_file():
        return False
    try:
        receipt = read(p)
        return (receipt['signature'] == digest([run_signature, job]) and receipt['returncode'] == 0
                and receipt['files'] == validate_job(job))
    except (ValueError, KeyError, OSError, RuntimeError):
        return False


def run_jobs(jobs, gpus, out, run_signature):
    pending = []
    for job in jobs:
        if completed(out, job, run_signature):
            print('RESUME verified '+job['name'], flush=True)
            continue
        dest = Path(job['destination'])
        # Refuse mixed or partial outputs. Preserve evidence for diagnosis.
        if dest.exists() and any(dest.iterdir()):
            raise RuntimeError(f'Unverified or partial output at {dest}. Use a fresh --output-root or move that job directory and receipt aside before retrying.')
        pending.append(job)
    running, failed = {}, []
    (out/'logs').mkdir(exist_ok=True)
    while pending or running:
        for gpu in gpus:
            if gpu in running or not pending:
                continue
            job = pending.pop(0)
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            env['LSA_AUDIO_OUTPUT_ROOT'] = str(out/'audio')
            logpath = out/'logs'/f'{job["name"]}.log'
            if job['kind'] == 'image':
                spec = out/'jobs'/f'{job["name"]}.json'
                write(spec, job['job'])
                command = [sys.executable, '-m', 'lsa.train', '--job', str(spec)]
                cwd = IMAGE
            else:
                command = [sys.executable, str(AUDIO/'scripts'/job['script']), *job['argv']]
                cwd = AUDIO
            log = logpath.open('w')
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
            write(out/'jobs'/f'{job["name"]}.launch.json', {'pid':process.pid, 'gpu':gpu, 'command':command, 'cwd':str(cwd), 'log':str(logpath)})
            running[gpu] = job, process, log
            print(f'START {job["name"]} GPU={gpu} PID={process.pid} LOG={logpath}', flush=True)
        if running:
            time.sleep(0.2)
        for gpu, (job, process, log) in list(running.items()):
            rc = process.poll()
            if rc is None:
                continue
            log.close()
            files, error = {}, None
            if rc == 0:
                try:
                    files = validate_job(job)
                except Exception as exc:
                    error, rc = str(exc), 1
            write(receipt_path(out,job), {'signature':digest([run_signature,job]), 'returncode':rc,
                                         'files':files, 'error':error})
            (out/'logs'/f'{job["name"]}.rc').write_text(str(rc)+'\n')
            (out/'logs'/f'{job["name"]}.done').write_text('DONE\n')
            print(f'DONE {job["name"]} RC={rc}'+(f' {error}' if error else ''), flush=True)
            if rc:
                failed.append(job['name'])
            del running[gpu]
    if failed:
        raise RuntimeError('Failed jobs: '+', '.join(failed))


def all_jobs(suite, data, out, final=False):
    if suite == 'audio':
        return audio_jobs(data, out)
    short = suite.removeprefix('image-')
    jobs = image_jobs(short, data, out)
    if short == 'sl2a' and final:
        jobs += image_jobs(short, data, out, 'final')
    return jobs


def numeric_close(value, target, tolerance, label):
    if not math.isfinite(float(value)) or abs(float(value)-float(target)) > tolerance:
        raise RuntimeError(f'{label}: {value} vs {target}, tolerance {tolerance}')


def audit_audio(out):
    with (AUDIO/'results/expected/main_paper_audio_expected.csv').open() as f:
        expected = list(csv.DictReader(f))
    rows = []
    for r in expected:
        paths = sorted((out/'audio/runs'/r['group']).glob('*/summary.json'))
        values = [float(read(p)['psnr']) for p in paths]
        ids = [read(p)['sample_id'] for p in paths]
        target_ids = [x['clip_id'] for x in manifest_rows(r['dataset'])]
        if len(values) != int(r['target_n']) or len(set(ids)) != len(ids) or set(ids) != set(target_ids):
            raise RuntimeError('Audio cohort mismatch: '+r['group'])
        if not all(math.isfinite(v) for v in values):
            raise RuntimeError('Nonfinite audio PSNR')
        avg = statistics.mean(values)
        rows.append({**r, 'observed_mean_psnr':avg, 'difference_db':avg-float(r['expected_mean_psnr'])})
    write(out/'reports/audio_comparison.json', rows)
    for r in rows:
        numeric_close(r['observed_mean_psnr'], r['expected_mean_psnr'], 0.01, r['group'])
    return {'status':'PASS', 'groups':len(rows), 'fits':sum(int(r['target_n']) for r in rows),
            'scope':'manifest identity, completion, finite values, dataset means within 0.01 dB. No source per-clip reference values supplied.'}


def audit_images(suite, data, out):
    short = suite.removeprefix('image-')
    module = load_module('image_summary', IMAGE/'scripts/summarize.py')
    config = image_config(short, data, out)
    with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink):
        if short == 'core':
            module.core(config)
        else:
            getattr(module, short)(config, out/'images/kodak_core')
    observed = config['output_root']/f'{short}_per_image.csv'
    with observed.open() as f:
        rows = list(csv.DictReader(f))
    with (IMAGE/'reference'/f'{short}_per_image.csv').open() as f:
        refs = list(csv.DictReader(f))
    key = lambda r: (r.get('condition',''), r['image'])
    lookup = {key(r):r for r in rows}
    if len(rows) != len(refs) or len(lookup) != len(rows) or set(lookup) != {key(r) for r in refs}:
        raise RuntimeError('Image result identities do not match')
    differences = []
    for ref in refs:
        row = lookup[key(ref)]
        for field in ref:
            if ('psnr' in field or 'ssim' in field) and field in row:
                tolerance = 0.02 if 'psnr' in field else 0.002
                differences.append({'image':ref['image'], 'condition':ref.get('condition'), 'metric':field,
                                    'observed':float(row[field]),'reference':float(ref[field]),'tolerance':tolerance})
    write(out/'reports'/f'{suite}_comparison.json', differences)
    for r in differences:
        numeric_close(r['observed'],r['reference'],r['tolerance'],f'{suite}/{r["image"]}/{r["metric"]}')
    if short == 'core':
        summary = read(config['output_root']/'core_summary.json')
        if summary['selected_lsa_candidate'] != 'lr5em4_actlr2p0':
            raise RuntimeError('Selected LSA candidate differs from reference')
    if short == 'sl2a' and read(config['output_root']/'selected_learning_rates.json') != {'full':0.004,'parameter_matched':0.004}:
        raise RuntimeError('Selected SL2A learning rates differ from reference')
    return {'status':'PASS','rows':len(rows),'metric_checks':len(differences),
            'scope':'per-image PSNR within 0.02 dB, SSIM within 0.002, selection identity. These are diagnostic gates, not cross-hardware guarantees.'}


def audit(suites, data, out):
    source_check()
    report = {'status':'FAIL', 'suites':{}}
    for suite in suites:
        try:
            context = read(out/'contexts'/f'{suite}.json')
            if context['code'] != code_state() or context['data'] != dataset_files([suite], data):
                raise RuntimeError('Code/data differ from run provenance')
            signature = digest(context)
            jobs = all_jobs(suite, data, out, final=True)
            if not all(completed(out, job, signature) for job in jobs):
                raise RuntimeError('Missing, changed, failed, or unverified job outputs')
            if suite in ('image-staf','image-sl2a'):
                core = read(out/'contexts/image-core.json')
                if core['code'] != code_state() or core['data'] != dataset_files(['image-core'], data):
                    raise RuntimeError('Core source provenance mismatch')
                if not all(completed(out,j,digest(core)) for j in all_jobs('image-core',data,out)):
                    raise RuntimeError('Image comparisons require verified core results')
                audit_images('image-core',data,out)
            report['suites'][suite] = audit_audio(out) if suite == 'audio' else audit_images(suite,data,out)
        except Exception as exc:
            report['suites'][suite] = {'status':'FAIL', 'reason':str(exc)}
    report['status'] = 'PASS' if all(x['status']=='PASS' for x in report['suites'].values()) else 'FAIL'
    write(out/'reports/audit.json', report)
    print(json.dumps(report, indent=2))
    if report['status'] != 'PASS':
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['plan','doctor','check-data','run','audit','verify-source'])
    p.add_argument('--suite', choices=['all',*SUITES], default='all')
    p.add_argument('--data-root', type=Path, default=ROOT/'data')
    p.add_argument('--output-root', type=Path, default=ROOT/'outputs/reference')
    p.add_argument('--gpus', default='0', help='Physical GPU ordinals, one job per GPU. Unset CUDA_VISIBLE_DEVICES before launch.')
    args = p.parse_args()
    suites = list(SUITES) if args.suite=='all' else [args.suite]
    data, out = args.data_root.resolve(), args.output_root.resolve()
    if args.command=='verify-source':
        source_check()
        print('SOURCE_INTEGRITY=PASS')
    elif args.command=='plan':
        print(json.dumps({'audio':{'groups':18,'fits':450},'image-core':{'fits':120},
                          'image-staf':{'fits':1344},'image-sl2a':{'calibration_fits':60,'final_fits':48},
                          'total_fits':2022,'selected_suites':suites,'reference_status':'GPU_REPRODUCTION_PENDING'},indent=2))
    elif args.command=='doctor':
        print(json.dumps(environment(), indent=2))
    elif args.command=='check-data':
        files = dataset_files(suites,data)
        write(out/'data_inventory.json',files)
        print(f'DATA_CHECK=PASS files={len(files)}. Audio hashes identify this local copy, not historical byte equivalence.')
    elif args.command=='audit':
        audit(suites,data,out)
    elif args.command=='run':
        source_check()
        if os.environ.get('CUDA_VISIBLE_DEVICES'):
            raise RuntimeError('Unset CUDA_VISIBLE_DEVICES. The runner assigns physical GPUs using --gpus.')
        gpus = [int(x) for x in args.gpus.split(',')]
        if not gpus or len(gpus)!=len(set(gpus)) or min(gpus)<0:
            raise ValueError('GPU IDs must be unique nonnegative integers')
        runtime = environment(require_cuda=True)
        import torch
        if max(gpus)>=torch.cuda.device_count():
            raise ValueError('GPU ordinal outside visible inventory')
        # Check every requested dataset before starting an expensive suite.
        dataset_files(suites,data)
        if args.suite in ('image-staf','image-sl2a'):
            audit(['image-core'],data,out)
        out.mkdir(parents=True,exist_ok=True)
        import fcntl
        with (out/'.run.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            for suite in suites:
                context = {'code':code_state(),'data':dataset_files([suite],data),'environment':runtime}
                path = out/'contexts'/f'{suite}.json'
                if path.exists() and read(path)!=context:
                    raise RuntimeError('Run provenance changed. Use a fresh output root.')
                write(path,context)
                signature = digest(context)
                jobs = all_jobs(suite,data,out)
                run_jobs(jobs,gpus,out,signature)
                if suite=='image-sl2a':
                    run_jobs(image_jobs('sl2a',data,out,'final'),gpus,out,signature)
            audit(suites,data,out)


if __name__=='__main__':
    main()
