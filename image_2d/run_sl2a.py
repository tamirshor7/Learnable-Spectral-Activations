import argparse
from pathlib import Path

from scripts import jobs, summarize

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Kodak sl2a experiments.")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/kodak")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs", help="Parent of the Kodak experiment output directories.")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--core-output", type=Path, default=None, help="Core results, default: <output-root>/kodak_core.")
    args = parser.parse_args()
    args.core_output = args.core_output or args.output_root / "kodak_core"
    gpus = [int(gpu) for gpu in args.gpus.split(",")]
    if not gpus or len(set(gpus)) != len(gpus) or min(gpus) < 0:
        raise ValueError("Supply distinct nonnegative GPU IDs.")
    config = jobs.load_config(ROOT / "configs/kodak_sl2a.yaml")
    config["data_root"] = args.data_root.resolve()
    config["output_root"] = args.output_root.resolve() / "kodak_sl2a"
    jobs.verify_data(config)
    config["output_root"].mkdir(parents=True, exist_ok=True)
    if not (args.core_output / "core_per_image.csv").is_file():
        raise RuntimeError("Run run_core.py first, or pass --core-output.")
    jobs.run_jobs(jobs.sl2a_calibration_jobs(config), gpus, not args.no_resume)
    selected = jobs.select_sl2a_learning_rates(config)
    jobs.run_jobs(jobs.sl2a_final_jobs(config, selected), gpus, not args.no_resume)
    summarize.sl2a(config, args.core_output.resolve())


if __name__ == "__main__":
    main()
