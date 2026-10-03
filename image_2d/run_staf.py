import argparse
from pathlib import Path

from scripts import jobs, summarize

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Kodak staf experiments.")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/kodak")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/kodak_staf")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--core-output", type=Path, default=ROOT / "outputs/kodak_core")
    args = parser.parse_args()
    gpus = [int(gpu) for gpu in args.gpus.split(",")]
    if not gpus or len(set(gpus)) != len(gpus) or min(gpus) < 0:
        raise ValueError("Supply distinct nonnegative GPU IDs.")
    config = jobs.load_config(ROOT / "configs/kodak_staf.yaml")
    config["data_root"] = args.data_root.resolve()
    config["output_root"] = args.output_root.resolve()
    jobs.verify_data(config)
    config["output_root"].mkdir(parents=True, exist_ok=True)
    if not (args.core_output / "core_per_image.csv").is_file():
        raise RuntimeError("Run run_core.py first, or pass --core-output.")
    jobs.run_jobs(jobs.staf_jobs(config), gpus, not args.no_resume)
    summarize.staf(config, args.core_output.resolve())


if __name__ == "__main__":
    main()
