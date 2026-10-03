"""Collect bounded validation evidence, excluding datasets and model checkpoints."""
import argparse
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output-root',type=Path,default=ROOT/'outputs/reference')
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
a.out.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(a.out,'w',zipfile.ZIP_DEFLATED) as z:
    for root,label in [(a.output_root,'run'),(ROOT/'validation','local_validation')]:
        if not root.exists():
            continue
        for f in sorted(root.rglob('*')):
            if not f.is_file() or f.suffix not in ('.json','.csv','.txt','.log','.rc','.done','.pid'):
                continue
            name=str(Path(label)/f.relative_to(root))
            if f.suffix=='.log':
                with f.open('rb') as h:
                    h.seek(max(0,f.stat().st_size-16384))
                    z.writestr(name,h.read())
            else:
                z.write(f,name)
print(a.out.resolve())
