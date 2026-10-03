"""Download official source datasets. Audio archives are large and have no bundled reference hashes."""
import argparse
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print('EXISTS '+str(path), flush=True)
        return
    partial = path.with_suffix(path.suffix+'.partial')
    print('DOWNLOAD '+url, flush=True)
    urllib.request.urlretrieve(url, partial)
    partial.replace(path)


def extract_selected(archive, root, prefix):
    root = root.resolve()
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            # Dataset tarballs contain normal files/directories. Never extract links.
            target = (root/member.name).resolve()
            if not target.is_relative_to(root) or not (member.isfile() or member.isdir()):
                raise RuntimeError('Unsafe archive member: '+member.name)
            if not target.relative_to(root).parts or target.relative_to(root).parts[0] != prefix.rstrip('/'):
                raise RuntimeError('Unexpected archive root: '+member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    continue
                temporary = target.with_suffix(target.suffix+'.partial')
                with tar.extractfile(member) as src, temporary.open('wb') as dst:
                    import shutil
                    shutil.copyfileobj(src,dst)
                temporary.replace(target)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', choices=['images','audio','all'], default='images')
    p.add_argument('--data-root', type=Path, default=ROOT/'data')
    args=p.parse_args()
    root=args.data_root.resolve()
    if args.suite in ('images','all'):
        subprocess.run([sys.executable,str(ROOT/'experiments/images/scripts/download_kodak.py'),'--out',str(root/'kodak')],check=True)
    if args.suite in ('audio','all'):
        sources=[('https://www.openslr.org/resources/12/dev-clean.tar.gz',root/'librispeech_raw','LibriSpeech/'),
                 ('https://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz',root,'nsynth-valid/')]
        for url,dest,prefix in sources:
            archive=root/'archives'/url.rsplit('/',1)[1]
            fetch(url,archive)
            extract_selected(archive,dest,prefix)
    print('DOWNLOAD_COMPLETE. Run reproduce.py check-data to verify the required files.')

if __name__=='__main__':
    main()
