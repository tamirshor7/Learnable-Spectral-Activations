"""Data paths and process launch shared by the audio experiments."""
import csv
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = {"nsynth": "nsynth_pilot10_manifest.csv",
             "librispeech": "librispeech_pilot40_manifest.csv"}


def add_arguments(parser):
    parser.add_argument("--dataset", choices=["nsynth", "librispeech", "both"], default="both",
                        help="Clip set to fit (default: both).")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index (default: 0).")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data",
                        help="Directory containing nsynth-valid/ and librispeech_raw/.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs",
                        help="Directory for runs, summaries, manifests and logs.")


def prepare_data(args):
    if args.gpu < 0:
        raise ValueError("--gpu must be nonnegative")
    datasets = list(MANIFESTS) if args.dataset == "both" else [args.dataset]
    manifests = {}
    for dataset in datasets:
        with (ROOT / "manifests" / MANIFESTS[dataset]).open() as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = list(reader)
        for row in rows:
            path = args.data_root.resolve() / Path(row["wav_path"]).relative_to("data")
            if not path.is_file():
                raise FileNotFoundError(f"Missing clip: {path}. Run audio/download_data.py --dataset {dataset} or set --data-root.")
            row["wav_path"] = str(path)
        destination = args.output_root.resolve() / "manifests" / MANIFESTS[dataset]
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        manifests[dataset] = destination
    return manifests


def run(script, parameters, args):
    command = [sys.executable, str(ROOT / "scripts" / script)]
    for key, value in parameters.items():
        if value is True:
            command.append("--" + key)
        elif value is not False:
            command.extend(["--" + key, str(value)])
    command.extend(["--device", args.device, "--output-root", str(args.output_root.resolve())])
    log = args.output_root.resolve() / "logs" / (parameters["group"] + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"
    print(f"Running {parameters['group']}\nLog: {log}", flush=True)
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    log.with_suffix(".rc").write_text(str(result.returncode) + "\n")
    if result.returncode:
        raise RuntimeError(f"Training failed. See {log}")
