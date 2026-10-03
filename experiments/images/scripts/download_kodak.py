import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HASHES = json.loads((ROOT / "reference" / "kodak_sha256.json").read_text())
BASE = "https://r0k.us/graphics/kodak/kodak"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/kodak")
    args = parser.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    for name, expected in HASHES.items():
        path = out / name
        if path.is_file() and sha256(path) == expected:
            print(f"OK {name}")
            continue
        urllib.request.urlretrieve(f"{BASE}/{name}", path)
        observed = sha256(path)
        if observed != expected:
            path.unlink(missing_ok=True)
            raise RuntimeError(f"hash mismatch for {name}: {observed}")
        print(f"DOWNLOADED {name}")


if __name__ == "__main__":
    main()
