import argparse
import csv
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def check(method, output):
    with (ROOT / "reference/results.csv").open() as handle:
        references = list(csv.DictReader(handle))
    rows = []
    for ref in references:
        if method != "all" and ref["method"].lower() != method:
            continue
        manifest = "nsynth_pilot10_manifest.csv" if ref["dataset"] == "nsynth" else "librispeech_pilot40_manifest.csv"
        with (ROOT / "manifests" / manifest).open() as handle:
            expected_ids = {row["clip_id"] for row in csv.DictReader(handle)}
        error = ""
        mean = float("nan")
        try:
            summaries = [json.loads(path.read_text()) for path in sorted((output / "runs" / ref["group"]).glob("*/summary.json"))]
            ids = [row["sample_id"] for row in summaries]
            if len(ids) != int(ref["target_n"]) or set(ids) != expected_ids:
                raise ValueError("missing or duplicate clips")
            values = [float(row["psnr"]) for row in summaries]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("nonfinite PSNR")
            if any(row.get("best_step", -1) < 0 for row in summaries):
                raise ValueError("missing evaluated checkpoint")
            mean = statistics.mean(values)
            if abs(mean - float(ref["expected_mean_psnr"])) > 0.01:
                raise ValueError("mean differs by more than 0.01 dB")
        except (OSError, KeyError, ValueError) as exc:
            error = str(exc)
        rows.append({**ref, "mean_psnr": mean, "status": "FAIL" if error else "PASS", "error": error})
        print(f'{rows[-1]["status"]} {ref["dataset"]} {ref["method"]} {ref["config"]}: {mean:.4f}' + (f" ({error})" if error else ""))
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"comparison_{method}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return all(row["status"] == "PASS" for row in rows)


def main():
    parser = argparse.ArgumentParser(description="Compare audio fits with the reference results.")
    parser.add_argument("--method", choices=["all", "lsa", "finer", "siren", "fjnb", "sl2a"], default="all")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    if not check(args.method, args.output_root.resolve()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
