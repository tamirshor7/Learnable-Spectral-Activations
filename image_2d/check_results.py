import argparse
import csv
import json
import math
from pathlib import Path

from scripts import jobs, summarize

ROOT = Path(__file__).resolve().parent


def check(suite, output, data):
    config = jobs.load_config(ROOT / f"configs/kodak_{suite}.yaml")
    config["output_root"] = output / f"kodak_{suite}"
    config["data_root"] = data
    core_output = output / "kodak_core"
    if suite == "core":
        expected_jobs = jobs.core_jobs(config)
    elif suite == "staf":
        expected_jobs = jobs.staf_jobs(config)
    else:
        expected_jobs = jobs.sl2a_calibration_jobs(config)
    for job in expected_jobs:
        if not jobs.completed(job):
            raise RuntimeError(f"Incomplete or changed run: {job['name']}")
    if suite == "sl2a":
        selected = jobs.select_sl2a_learning_rates(config)
        for job in jobs.sl2a_final_jobs(config, selected):
            if not jobs.completed(job):
                raise RuntimeError(f"Incomplete or changed run: {job['name']}")
        if selected != {"full": 0.004, "parameter_matched": 0.004}:
            raise RuntimeError("Selected SL2A learning rates differ from the reference")
    if suite == "core":
        summarize.core(config)
        selected = json.loads((config["output_root"] / "core_summary.json").read_text())["selected_lsa_candidate"]
        if selected != "lr5em4_actlr2p0":
            raise RuntimeError("Selected LSA candidate differs from the reference")
    else:
        getattr(summarize, suite)(config, core_output)
    with (ROOT / f"reference/{suite}_per_image.csv").open() as handle:
        reference = list(csv.DictReader(handle))
    with (config["output_root"] / f"{suite}_per_image.csv").open() as handle:
        observed = list(csv.DictReader(handle))
    key = lambda row: (row.get("condition", ""), row["image"])
    lookup = {key(row): row for row in observed}
    if len(observed) != len(reference) or len(lookup) != len(observed) or set(lookup) != {key(row) for row in reference}:
        raise RuntimeError("Image identities differ from the reference")
    for ref in reference:
        row = lookup[key(ref)]
        for metric in ref:
            if metric in row and ("psnr" in metric or "ssim" in metric):
                value = float(row[metric])
                tolerance = 0.02 if "psnr" in metric else 0.002
                if not math.isfinite(value) or abs(value - float(ref[metric])) > tolerance:
                    raise RuntimeError(f"{ref['image']} {metric}: {value} vs {ref[metric]}")
    print(f"PASS {suite}")


def main():
    parser = argparse.ArgumentParser(description="Compare Kodak fits with the reference results.")
    parser.add_argument("--suite", choices=["all", "core", "staf", "sl2a"], default="all")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/kodak")
    args = parser.parse_args()
    suites = ["core", "staf", "sl2a"] if args.suite == "all" else (["core", args.suite] if args.suite != "core" else ["core"])
    for suite in suites:
        check(suite, args.output_root.resolve(), args.data_root.resolve())


if __name__ == "__main__":
    main()
