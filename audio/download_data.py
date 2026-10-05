import argparse
import csv
import hashlib
from pathlib import Path
import shutil
import tarfile
import urllib.request


ROOT = Path(__file__).resolve().parent

NSYNTH_SHA256 = "00dea2645fbe0069258567da30807a90825e0bab54c077d6481f253096c4e2a0"


SOURCES = {
    "librispeech": {
        "archives": [
            {
                "urls": [
                    "https://www.openslr.org/resources/12/dev-clean.tar.gz",
                ],
                "archive": "dev-clean.tar.gz",
                "subdir": "librispeech_raw",
                "folder": "LibriSpeech",
                "sha256": None,
            },
            {
                "urls": [
                    "https://www.openslr.org/resources/12/dev-other.tar.gz",
                ],
                "archive": "dev-other.tar.gz",
                "subdir": "librispeech_raw",
                "folder": "LibriSpeech",
                "sha256": None,
            },
        ],
        "manifest": ROOT / "manifests" / "librispeech_pilot40_manifest.csv",
        "expected_clips": 40,
    },
    "nsynth": {
        "archives": [
            {
                "urls": [
                    "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz",
                    "https://huggingface.co/datasets/confit/nsynth/resolve/main/nsynth-valid.jsonwav.tar.gz",
                ],
                "archive": "nsynth-valid.jsonwav.tar.gz",
                "subdir": "",
                "folder": "nsynth-valid",
                "sha256": NSYNTH_SHA256,
            },
        ],
        "manifest": ROOT / "manifests" / "nsynth_pilot10_manifest.csv",
        "expected_clips": 10,
    },
}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path, expected_sha256):
    if expected_sha256 is None:
        return

    actual = file_sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"Checksum mismatch for {path.name}: "
            f"expected {expected_sha256}, got {actual}"
        )


def download(urls, path, expected_sha256=None):
    if path.exists():
        try:
            verify_archive(path, expected_sha256)
            print(f"Using cached archive {path}", flush=True)
            return
        except RuntimeError as error:
            print(error, flush=True)
            print(f"Removing invalid cached archive {path}", flush=True)
            path.unlink()

    partial = path.with_suffix(path.suffix + ".partial")
    errors = []

    for url in urls:
        partial.unlink(missing_ok=True)
        print(f"Downloading {url}", flush=True)

        try:
            urllib.request.urlretrieve(url, partial)
            verify_archive(partial, expected_sha256)
            partial.replace(path)
            print(f"Downloaded {path.name}", flush=True)
            return
        except Exception as error:
            partial.unlink(missing_ok=True)
            errors.append(f"{url}: {error}")
            print(f"Download failed: {error}", flush=True)

    joined = "\n".join(errors)
    raise RuntimeError(
        f"Failed to download {path.name} from all sources:\n{joined}"
    )


def extract(archive, destination, folder):
    destination = destination.resolve()

    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            path = (destination / member.name).resolve()

            if not path.is_relative_to(destination) or not (
                member.isfile() or member.isdir()
            ):
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


def validate_manifest(dataset, root):
    config = SOURCES[dataset]
    manifest = config["manifest"]

    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    expected = config["expected_clips"]

    if len(rows) != expected:
        raise RuntimeError(
            f"{manifest.name} contains {len(rows)} clips, expected {expected}"
        )

    missing = []

    for row in rows:
        relative = Path(row["wav_path"])

        if relative.parts and relative.parts[0] == "data":
            relative = Path(*relative.parts[1:])

        path = root / relative

        if not path.is_file():
            missing.append(path)

    if missing:
        preview = "\n".join(str(path) for path in missing[:10])

        raise RuntimeError(
            f"{dataset}: {len(missing)}/{expected} required clips are missing:\n"
            f"{preview}"
        )

    print(f"Verified {expected}/{expected} {dataset} clips", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Download the audio datasets used by the paper."
    )
    parser.add_argument(
        "--dataset",
        choices=["nsynth", "librispeech", "both"],
        default="both",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data",
    )
    args = parser.parse_args()

    root = args.out.resolve()
    archives = root / "archives"
    archives.mkdir(parents=True, exist_ok=True)

    datasets = list(SOURCES) if args.dataset == "both" else [args.dataset]

    for dataset in datasets:
        config = SOURCES[dataset]

        for source in config["archives"]:
            archive = archives / source["archive"]

            download(
                source["urls"],
                archive,
                expected_sha256=source["sha256"],
            )

            extract(
                archive,
                root / source["subdir"],
                source["folder"],
            )

        validate_manifest(dataset, root)

    print(f"Data saved to {root}")


if __name__ == "__main__":
    main()
