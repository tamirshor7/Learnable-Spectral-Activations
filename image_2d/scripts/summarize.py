import argparse
import csv
import json
import statistics
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    for key in ("data_root", "output_root"):
        value = Path(config[key])
        if not value.is_absolute():
            value = ROOT / value
        config[key] = value
    return config


def read_summary(path):
    return json.loads(Path(path).read_text())


def mean(values):
    return statistics.mean(values)


def std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def core(config):
    output = config["output_root"]
    images = config["images"]
    calibration = set(config["calibration_images"])
    holdout = [image for image in images if image not in calibration]

    candidate_rows = []
    by_candidate = {}
    per_image = {image: [] for image in images}

    for candidate in config["lsa"]["candidates"]:
        name = candidate["name"]
        rows = []
        for image in images:
            summary = read_summary(output / "runs" / f"lsa__{image}__{name}" / "summary.json")
            row = {
                "image": image,
                "candidate": name,
                "lr": candidate["lr"],
                "act_lr_mult": candidate["act_lr_mult"],
                "psnr": summary["best_psnr"],
                "ssim": summary["best_ssim"],
                "best_step": summary["best_step"],
            }
            rows.append(row)
            per_image[image].append(row)
        by_candidate[name] = rows
        calibration_values = [row["psnr"] for row in rows if row["image"] in calibration]
        candidate_rows.append({
            "candidate": name,
            "lr": candidate["lr"],
            "act_lr_mult": candidate["act_lr_mult"],
            "calibration_mean_psnr": mean(calibration_values),
            "calibration_std_psnr": std(calibration_values),
        })

    candidate_rows.sort(key=lambda row: (-row["calibration_mean_psnr"], row["calibration_std_psnr"], row["lr"], row["act_lr_mult"]))
    selected = candidate_rows[0]["candidate"]
    selected_rows = by_candidate[selected]

    tuned = []
    for image in images:
        row = max(per_image[image], key=lambda value: value["psnr"])
        tuned.append(dict(row))

    baselines = {}
    for method in ("finer", "siren"):
        rows = []
        for image in images:
            summary = read_summary(output / "runs" / f"{method}__{image}" / "summary.json")
            rows.append({
                "image": image,
                "psnr": summary["best_psnr"],
                "ssim": summary["best_ssim"],
                "best_step": summary["best_step"],
            })
        baselines[method] = rows

    result = {
        "selected_lsa_candidate": selected,
        "candidate_selection": candidate_rows,
        "all24": {
            "lsa_global_psnr": mean([row["psnr"] for row in selected_rows]),
            "lsa_global_ssim": mean([row["ssim"] for row in selected_rows]),
            "lsa_per_image_tuned_psnr": mean([row["psnr"] for row in tuned]),
            "lsa_per_image_tuned_ssim": mean([row["ssim"] for row in tuned]),
            "finer_psnr": mean([row["psnr"] for row in baselines["finer"]]),
            "finer_ssim": mean([row["ssim"] for row in baselines["finer"]]),
            "siren_psnr": mean([row["psnr"] for row in baselines["siren"]]),
            "siren_ssim": mean([row["ssim"] for row in baselines["siren"]]),
        },
        "holdout18": {
            "lsa_global_psnr": mean([row["psnr"] for row in selected_rows if row["image"] in holdout]),
            "lsa_global_ssim": mean([row["ssim"] for row in selected_rows if row["image"] in holdout]),
        },
    }

    selected_lookup = {row["image"]: row for row in selected_rows}
    finer_lookup = {row["image"]: row for row in baselines["finer"]}
    siren_lookup = {row["image"]: row for row in baselines["siren"]}
    tuned_lookup = {row["image"]: row for row in tuned}
    rows = []
    for image in images:
        rows.append({
            "image": image,
            "lsa_global_psnr": selected_lookup[image]["psnr"],
            "lsa_global_ssim": selected_lookup[image]["ssim"],
            "lsa_tuned_psnr": tuned_lookup[image]["psnr"],
            "lsa_tuned_ssim": tuned_lookup[image]["ssim"],
            "lsa_tuned_candidate": tuned_lookup[image]["candidate"],
            "finer_psnr": finer_lookup[image]["psnr"],
            "finer_ssim": finer_lookup[image]["ssim"],
            "siren_psnr": siren_lookup[image]["psnr"],
            "siren_ssim": siren_lookup[image]["ssim"],
        })

    write_csv(output / "core_per_image.csv", rows)
    (output / "core_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def staf(config, core_output):
    output = config["output_root"]
    images = config["images"]
    core_rows = list(csv.DictReader((Path(core_output) / "core_per_image.csv").open()))
    lsa = {row["image"]: row for row in core_rows}

    rows = []
    for image in images:
        candidates = []
        for path in sorted((output / "runs").glob(f"staf__{image}__*/summary.json")):
            summary = read_summary(path)
            candidates.append((summary["best_psnr"], summary))
        if len(candidates) != 56:
            raise RuntimeError(f"{image}: expected 56 STAF runs, found {len(candidates)}")
        _, best = max(candidates, key=lambda item: item[0])
        lsa_psnr = float(lsa[image]["lsa_tuned_psnr"])
        lsa_ssim = float(lsa[image]["lsa_tuned_ssim"])
        rows.append({
            "image": image,
            "lsa_psnr": lsa_psnr,
            "lsa_ssim": lsa_ssim,
            "staf_psnr": best["best_psnr"],
            "staf_ssim": best["best_ssim"],
            "tau": best["params"]["tau"],
            "lr": best["params"]["lr"],
            "schedule": best["params"]["schedule"],
            "lsa_minus_staf_psnr": lsa_psnr - best["best_psnr"],
            "lsa_minus_staf_ssim": lsa_ssim - best["best_ssim"],
        })

    result = {
        "n_images": len(rows),
        "lsa_mean_psnr": mean([row["lsa_psnr"] for row in rows]),
        "staf_mean_psnr": mean([row["staf_psnr"] for row in rows]),
        "mean_lsa_minus_staf_psnr": mean([row["lsa_minus_staf_psnr"] for row in rows]),
        "lsa_psnr_wins": sum(row["lsa_minus_staf_psnr"] > 0 for row in rows),
        "lsa_mean_ssim": mean([row["lsa_ssim"] for row in rows]),
        "staf_mean_ssim": mean([row["staf_ssim"] for row in rows]),
        "lsa_ssim_wins": sum(row["lsa_minus_staf_ssim"] > 0 for row in rows),
    }
    write_csv(output / "staf_per_image.csv", rows)
    (output / "staf_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def sl2a(config, core_output):
    output = config["output_root"]
    images = config["images"]
    calibration = set(config["calibration_images"])
    holdout = [image for image in images if image not in calibration]
    selected = json.loads((output / "selected_learning_rates.json").read_text())
    core_rows = list(csv.DictReader((Path(core_output) / "core_per_image.csv").open()))
    lsa = {row["image"]: row for row in core_rows}

    rows = []
    result = {"selected_learning_rates": selected, "conditions": {}}
    for condition in config["conditions"]:
        condition_rows = []
        for image in images:
            summary = read_summary(output / "runs" / f"sl2a_final__{condition}__{image}" / "summary.json")
            row = {
                "condition": condition,
                "image": image,
                "psnr": summary["best_psnr"],
                "ssim": summary["best_ssim"],
                "parameters": summary["num_params"],
                "learning_rate": selected[condition],
            }
            rows.append(row)
            condition_rows.append(row)
        result["conditions"][condition] = {
            "parameters": condition_rows[0]["parameters"],
            "learning_rate": selected[condition],
            "all24_psnr": mean([row["psnr"] for row in condition_rows]),
            "holdout18_psnr": mean([row["psnr"] for row in condition_rows if row["image"] in holdout]),
            "all24_ssim": mean([row["ssim"] for row in condition_rows]),
            "holdout18_ssim": mean([row["ssim"] for row in condition_rows if row["image"] in holdout]),
        }

    lsa_holdout = [float(lsa[image]["lsa_global_psnr"]) for image in holdout]
    result["lsa"] = {
        "parameters": 199043,
        "all24_psnr": mean([float(lsa[image]["lsa_global_psnr"]) for image in images]),
        "holdout18_psnr": mean(lsa_holdout),
    }
    for condition in config["conditions"]:
        lookup = {row["image"]: row for row in rows if row["condition"] == condition}
        deltas = [float(lsa[image]["lsa_global_psnr"]) - lookup[image]["psnr"] for image in holdout]
        result["conditions"][condition]["lsa_minus_sl2a_holdout_psnr"] = mean(deltas)
        result["conditions"][condition]["lsa_holdout_wins"] = sum(delta > 0 for delta in deltas)

    write_csv(output / "sl2a_per_image.csv", rows)
    (output / "sl2a_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="suite", required=True)

    p = sub.add_parser("core")
    p.add_argument("--config", default="configs/kodak_core.yaml")

    p = sub.add_parser("staf")
    p.add_argument("--config", default="configs/kodak_staf.yaml")
    p.add_argument("--core-output", default="outputs/kodak_core")

    p = sub.add_parser("sl2a")
    p.add_argument("--config", default="configs/kodak_sl2a.yaml")
    p.add_argument("--core-output", default="outputs/kodak_core")

    args = parser.parse_args()
    if args.suite == "core":
        core(load_config(args.config))
    elif args.suite == "staf":
        staf(load_config(args.config), ROOT / args.core_output)
    else:
        sl2a(load_config(args.config), ROOT / args.core_output)


if __name__ == "__main__":
    main()
