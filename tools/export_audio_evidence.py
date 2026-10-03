"""Export per-clip historical summaries without collecting checkpoints or audio."""
import argparse
import csv
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.release import read,write,manifest_rows,sha

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--runs-root',required=True,type=Path,help='Historical directory containing group subdirectories')
p.add_argument('--out',required=True,type=Path)
a=p.parse_args()
with (ROOT/'experiments/audio/results/expected/main_paper_audio_expected.csv').open() as f:
    expected=list(csv.DictReader(f))
report={'source':'historical_external_outputs','groups':[],'status':'COMPLETE'}
for spec in expected:
    paths=sorted((a.runs_root/spec['group']).rglob('summary.json'))
    rows=[{'summary':read(path),'summary_sha256':sha(path),'path':str(path.resolve())} for path in paths]
    ids=[r['summary'].get('sample_id') for r in rows]
    target=[r['clip_id'] for r in manifest_rows(spec['dataset'])]
    complete=len(ids)==len(target) and set(ids)==set(target)
    if not complete:report['status']='INCOMPLETE'
    report['groups'].append({'group':spec['group'],'complete':complete,'expected_ids':target,'rows':rows})
write(a.out,report)
print(f'HISTORICAL_EXPORT={report["status"]} path={a.out.resolve()}')
if report['status']!='COMPLETE':raise SystemExit(1)
