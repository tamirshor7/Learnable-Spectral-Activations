import argparse
from pathlib import Path
import shutil
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "librispeech": ("https://www.openslr.org/resources/12/dev-clean.tar.gz", "librispeech_raw", "LibriSpeech"),
    "nsynth": ("https://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz", "", "nsynth-valid"),
}


def download(url, path):
    if path.exists():
        return
    print(f"Downloading {url}", flush=True)
    partial = path.with_suffix(path.suffix + ".partial")
    urllib.request.urlretrieve(url, partial)
    partial.replace(path)


def extract(archive, destination, folder):
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            path = (destination / member.name).resolve()
            if not path.is_relative_to(destination) or not (member.isfile() or member.isdir()):
                raise RuntimeError(f"Invalid archive entry: {member.name}")
            parts = path.relative_to(destination).parts
            if not parts or parts[0] != folder:
                raise RuntimeError(f"Unexpected archive folder: {member.name}")
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            elif not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = path.with_suffix(path.suffix + ".partial")
                with handle.extractfile(member) as source, partial.open("wb") as target:
                    shutil.copyfileobj(source, target)
                partial.replace(path)


def main():
    parser = argparse.ArgumentParser(description="Download LibriSpeech dev-clean and NSynth validation.")
    parser.add_argument("--dataset", choices=["nsynth", "librispeech", "both"], default="both")
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    root = args.out.resolve()
    (root / "archives").mkdir(parents=True, exist_ok=True)
    datasets = list(SOURCES) if args.dataset == "both" else [args.dataset]
    for dataset in datasets:
        url, subdir, folder = SOURCES[dataset]
        archive = root / "archives" / url.rsplit("/", 1)[1]
        download(url, archive)
        extract(archive, root / subdir, folder)
    print(f"Data saved to {root}")


if __name__ == "__main__":
    main()
