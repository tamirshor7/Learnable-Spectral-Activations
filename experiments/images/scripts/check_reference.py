import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "reference" / "results.json").read_text())


def compare(path, expected, tolerance):
    observed = json.loads(Path(path).read_text())
    failures = []
    for key, value in expected.items():
        current = observed
        for part in key.split("."):
            current = current[part]
        if abs(float(current) - float(value)) > tolerance:
            failures.append((key, current, value))
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", default="outputs/kodak_core/core_summary.json")
    parser.add_argument("--staf", default="outputs/kodak_staf/staf_summary.json")
    parser.add_argument("--sl2a", default="outputs/kodak_sl2a/sl2a_summary.json")
    parser.add_argument("--tolerance", type=float, default=0.02)
    args = parser.parse_args()

    expected = {
        args.core: {
            "all24.lsa_global_psnr": REFERENCE["core"]["lsa_global_all24_psnr"],
            "all24.lsa_per_image_tuned_psnr": REFERENCE["core"]["lsa_per_image_tuned_all24_psnr"],
            "all24.finer_psnr": REFERENCE["core"]["finer_all24_psnr"],
            "all24.siren_psnr": REFERENCE["core"]["siren_all24_psnr"],
        },
        args.staf: {
            "lsa_mean_psnr": REFERENCE["staf"]["lsa_per_image_tuned_all24_psnr"],
            "staf_mean_psnr": REFERENCE["staf"]["staf_all24_psnr"],
            "mean_lsa_minus_staf_psnr": REFERENCE["staf"]["mean_lsa_minus_staf_psnr"],
        },
        args.sl2a: {
            "lsa.holdout18_psnr": REFERENCE["sl2a"]["lsa_global_holdout18_psnr"],
            "conditions.full.holdout18_psnr": REFERENCE["sl2a"]["full_holdout18_psnr"],
            "conditions.parameter_matched.holdout18_psnr": REFERENCE["sl2a"]["parameter_matched_holdout18_psnr"],
        },
    }

    failures = []
    for path, values in expected.items():
        failures.extend((path, *item) for item in compare(ROOT / path, values, args.tolerance))

    if failures:
        for path, key, observed, target in failures:
            print(f"FAIL {path} {key}: {observed} vs {target}")
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()
