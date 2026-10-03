import ast
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import release as r
from tools.download_data import extract_selected

class ReleaseTests(unittest.TestCase):
    def setUp(self):
        root=ROOT/'outputs/tests'
        root.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=root)
        self.out=Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_source_integrity(self):
        r.source_check()

    def test_numerical_functions_unchanged(self):
        for path in (ROOT/'provenance/original_audio_scripts').glob('*.py'):
            def funcs(p):
                tree=ast.parse(p.read_text())
                return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name!='main'}
            self.assertEqual(funcs(path),funcs(ROOT/'experiments/audio/scripts'/path.name))

    def test_audio_commands_exact(self):
        lines=(ROOT/'experiments/audio/scripts/reproduce_main_audio.sh').read_text().splitlines()
        source=[shlex.split(x) for x in lines if x.startswith(('core ','fjnb ','sl2a '))]
        jobs=r.frozen_audio()
        self.assertEqual(len(jobs),18)
        for old,new in zip(source,jobs):
            self.assertEqual(old[:2],[new['engine'],new['name']])
            self.assertEqual(['cuda' if x=='$DEVICE' else x for x in old[2:]],new['argv'])
        total=sum(len(r.manifest_rows(r.getarg(j['argv'],'--dataset'))) for j in jobs)
        self.assertEqual(total,450)

    def test_image_job_counts_and_selections(self):
        data=self.out/'data'
        core=r.image_jobs('core',data,self.out)
        staf=r.image_jobs('staf',data,self.out)
        calibration=r.image_jobs('sl2a',data,self.out)
        self.assertEqual((len(core),len(staf),len(calibration)),(120,1344,60))
        self.assertEqual(len({j['name'] for j in staf}),1344)
        for j in calibration:
            spec=j['job']
            # Unique optimum depends only on six calibration images.
            r.write(Path(j['destination'])/'summary.json',{'best_psnr':40-abs(spec['params']['lr']-0.004)*100})
        finals=r.image_jobs('sl2a',data,self.out,'final')
        self.assertEqual(len(finals),48)
        self.assertTrue(all(j['job']['params']['lr']==0.004 for j in finals))
        self.assertEqual(r.read(self.out/'images/kodak_sl2a/selected_learning_rates.json'),{'full':0.004,'parameter_matched':0.004})

    def test_reject_nonfinite_metrics(self):
        for value in [float('nan'),float('inf'),-float('inf')]:
            with self.assertRaises(RuntimeError):r.numeric_close(value,1,0.02,'metric')
        r.numeric_close(1.01,1,0.02,'metric')
        with self.assertRaises(RuntimeError):r.numeric_close(1.1,1,0.02,'metric')

    def test_audio_identity_is_required(self):
        d=self.out/'audio/runs/group'
        r.write(d/'a/summary.json',{'sample_id':'wrong','psnr':90,'best_step':1})
        job={'kind':'audio','destination':str(d),'group':'group','ids':['expected']}
        with self.assertRaisesRegex(RuntimeError,'identity'):r.validate_job(job)

    def test_receipt_detects_changed_metrics_and_provenance(self):
        j=r.image_jobs('core',self.out/'data',self.out)[0]
        d=Path(j['destination'])
        r.write(d/'summary.json',{'image':'kodim01','method':'lsa','params':j['job']['params'],
             'common':j['job']['common'],'best_psnr':35.0,'best_ssim':0.9,'best_step':100})
        (d/'metrics.csv').write_text('step,psnr\n100,35\n')
        r.write(r.receipt_path(self.out,j),{'signature':r.digest(['original',j]),'returncode':0,'files':r.validate_job(j)})
        self.assertTrue(r.completed(self.out,j,'original'))
        self.assertFalse(r.completed(self.out,j,'different_environment'))
        (d/'metrics.csv').write_text('step,psnr\n100,99\n')
        self.assertFalse(r.completed(self.out,j,'original'))

    def test_empty_audit_fails(self):
        with self.assertRaises(SystemExit) as e:r.audit(['audio'],self.out/'data',self.out)
        self.assertEqual(e.exception.code,1)
        self.assertEqual(r.read(self.out/'reports/audit.json')['status'],'FAIL')

    def test_scheduler_failure_exit_and_receipt(self):
        job=r.image_jobs('core',self.out/'data',self.out)[0]
        real_popen=subprocess.Popen
        def failed_process(command,**kwargs):
            return real_popen([sys.executable,'-c','raise SystemExit(7)'],**kwargs)
        with patch.object(r.subprocess,'Popen',side_effect=failed_process):
            with self.assertRaisesRegex(RuntimeError,'Failed jobs'):
                r.run_jobs([job],[0],self.out,'signature')
        self.assertEqual(r.read(r.receipt_path(self.out,job))['returncode'],7)
        self.assertFalse(r.completed(self.out,job,'signature'))
        self.assertTrue((self.out/'logs'/f'{job["name"]}.done').exists())

    def test_resume_does_not_relaunch(self):
        job=r.image_jobs('core',self.out/'data',self.out)[0]
        with patch.object(r,'completed',return_value=True),patch.object(r.subprocess,'Popen') as popen:
            r.run_jobs([job],[0],self.out,'signature')
            popen.assert_not_called()

    def test_partial_output_not_silently_reused(self):
        job=r.image_jobs('core',self.out/'data',self.out)[0]
        path=Path(job['destination']);path.mkdir(parents=True)
        (path/'partial.txt').write_text('interrupted')
        with self.assertRaisesRegex(RuntimeError,'partial output'):
            r.run_jobs([job],[0],self.out,'signature')

    def test_scheduler_real_worker_and_resume(self):
        import numpy as np
        from PIL import Image
        image=self.out/'data with spaces'/'tiny.png'
        image.parent.mkdir(parents=True)
        Image.fromarray(np.full((8,8,3),128,dtype=np.uint8)).save(image)
        job=r.image_jobs('core',self.out/'data',self.out)[0]
        job['job']['image']=str(image)
        job['job']['common']={'steps':2,'batch_size':16,'eval_every':1,'eval_chunk':64,'seed':0,'save_best_image':False}
        with patch.dict(os.environ,{'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}):
            r.run_jobs([job],[0],self.out,'test')
        self.assertTrue(r.completed(self.out,job,'test'))
        with patch.object(r.subprocess,'Popen') as popen:
            r.run_jobs([job],[0],self.out,'test')
            popen.assert_not_called()

    def test_audio_audit_group_means_and_duplicate_identity(self):
        with (r.AUDIO/'results/expected/main_paper_audio_expected.csv').open() as f:
            expected=list(csv.DictReader(f))
        for row in expected:
            for sample in r.manifest_rows(row['dataset']):
                r.write(self.out/'audio/runs'/row['group']/sample['clip_id']/'summary.json',
                        {'sample_id':sample['clip_id'],'psnr':float(row['expected_mean_psnr']),'best_step':1})
        result=r.audit_audio(self.out)
        self.assertEqual(result['groups'],18)
        self.assertEqual(result['fits'],450)
        row=expected[0]
        paths=sorted((self.out/'audio/runs'/row['group']).glob('*/summary.json'))
        wrong=r.read(paths[0]);wrong['sample_id']=r.read(paths[1])['sample_id'];r.write(paths[0],wrong)
        with self.assertRaisesRegex(RuntimeError,'cohort mismatch'):
            r.audit_audio(self.out)

    def test_archive_root_without_trailing_slash(self):
        import io
        archive=self.out/'safe.tar.gz'
        with tarfile.open(archive,'w:gz') as f:
            root=tarfile.TarInfo('dataset');root.type=tarfile.DIRTYPE;f.addfile(root)
            item=tarfile.TarInfo('dataset/example');item.size=2;f.addfile(item,io.BytesIO(b'ok'))
        extract_selected(archive,self.out/'data','dataset/')
        self.assertEqual((self.out/'data/dataset/example').read_bytes(),b'ok')

    def test_archive_path_traversal_rejected(self):
        archive=self.out/'unsafe.tar.gz'
        with tarfile.open(archive,'w:gz') as f:
            item=tarfile.TarInfo('../bad');item.type=tarfile.DIRTYPE;f.addfile(item)
        with self.assertRaisesRegex(RuntimeError,'Unsafe'):
            extract_selected(archive,self.out/'data','dataset/')

if __name__=='__main__':
    unittest.main()
