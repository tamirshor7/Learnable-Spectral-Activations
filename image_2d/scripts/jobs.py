import argparse
import hashlib
import itertools
import math
import json
import os
import subprocess
import sys
import time
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


def verify_data(config):
    hashes = json.loads((ROOT / "reference" / "kodak_sha256.json").read_text())
    for image in config["images"]:
        name = f"{image}.png"
        path = config["data_root"] / name
        if not path.is_file():
            raise RuntimeError(f"missing Kodak image: {path}")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = hashes[name]
        if observed != expected:
            raise RuntimeError(f"Kodak hash mismatch: {name}")


def job_path(output_root, name):
    return output_root / "runs" / name


def make_job(config, method, image, name, params):
    out = job_path(config["output_root"], name)
    return {
        "name": name,
        "method": method,
        "image": str(config["data_root"] / f"{image}.png"),
        "out": str(out),
        "params": params,
        "common": config["common"],
    }


def core_jobs(config):
    jobs = []
    for image in config["images"]:
        for candidate in config["lsa"]["candidates"]:
            params = {k: v for k, v in config["lsa"].items() if k != "candidates"}
            params.update({k: v for k, v in candidate.items() if k != "name"})
            jobs.append(make_job(config, "lsa", image, f"lsa__{image}__{candidate['name']}", params))

        finer = dict(config["finer"])
        jobs.append(make_job(config, "finer", image, f"finer__{image}", finer))

        siren = dict(config["siren"])
        jobs.append(make_job(config, "siren", image, f"siren__{image}", siren))
    return jobs


def staf_jobs(config):
    jobs = []
    seen = set()
    for grid in (config["base_grid"], config["boundary_grid"]):
        for tau, lr, schedule in itertools.product(grid["tau"], grid["lr"], grid["schedule"]):
            key = (tau, lr, schedule)
            if key in seen:
                continue
            seen.add(key)
            for image in config["images"]:
                params = dict(config["model"])
                params.update({"tau": tau, "lr": lr, "schedule": schedule})
                tag = f"tau{tau}_lr{lr:g}_{schedule}".replace(".", "p")
                jobs.append(make_job(config, "staf", image, f"staf__{image}__{tag}", params))
    return jobs


def sl2a_calibration_jobs(config):
    jobs = []
    for condition, model in config["conditions"].items():
        for lr in config["learning_rates"]:
            for image in config["calibration_images"]:
                params = dict(model)
                params.update({"lr": lr, "schedule": "decay0p1"})
                tag = f"lr{lr:g}".replace(".", "p")
                jobs.append(make_job(config, "sl2a", image, f"sl2a_cal__{condition}__{image}__{tag}", params))
    return jobs


def select_sl2a_learning_rates(config):
    selected = {}
    for condition in config["conditions"]:
        rows = []
        for lr in config["learning_rates"]:
            values = []
            tag = f"lr{lr:g}".replace(".", "p")
            for image in config["calibration_images"]:
                path = job_path(config["output_root"], f"sl2a_cal__{condition}__{image}__{tag}") / "summary.json"
                if not path.is_file():
                    raise RuntimeError(f"missing calibration result: {path}")
                value = json.loads(path.read_text())["best_psnr"]
                if not math.isfinite(value):
                    raise RuntimeError(f"nonfinite calibration result: {path}")
                values.append(value)
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
            rows.append((mean, variance ** 0.5, lr))
        rows.sort(key=lambda x: (-x[0], x[1], x[2]))
        selected[condition] = rows[0][2]

    path = config["output_root"] / "selected_learning_rates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selected, indent=2) + "\n")
    return selected


def sl2a_final_jobs(config, selected):
    jobs = []
    for condition, model in config["conditions"].items():
        lr = selected[condition]
        for image in config["images"]:
            params = dict(model)
            params.update({"lr": lr, "schedule": "decay0p1"})
            jobs.append(make_job(config, "sl2a", image, f"sl2a_final__{condition}__{image}", params))
    return jobs


def completed(job):
    out = Path(job["out"])
    try:
        summary = json.loads((out / "summary.json").read_text())
        return ((out / "returncode.txt").read_text().strip() == "0"
                and json.loads((out / "job.json").read_text()) == job
                and summary["params"] == job["params"]
                and summary["common"] == job["common"]
                and math.isfinite(summary["best_psnr"])
                and math.isfinite(summary["best_ssim"]))
    except (OSError, ValueError, KeyError):
        return False


def launch(job, gpu):
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    job_file = out / "job.json"
    job_file.write_text(json.dumps(job, indent=2) + "\n")
    log = (out / "stdout.log").open("w")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    process = subprocess.Popen(
        [sys.executable, "-m", "lsa.train", "--job", str(job_file)],
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
    )
    return process, log


def run_jobs(jobs, gpus, resume):
    queue = [job for job in jobs if not (resume and completed(job))]
    running = {}
    failures = []

    while queue or running:
        free = [gpu for gpu in gpus if gpu not in running]
        while queue and free:
            gpu = free.pop(0)
            job = queue.pop(0)
            process, log = launch(job, gpu)
            running[gpu] = (job, process, log)
            print(f"START gpu={gpu} {job['name']}", flush=True)

        time.sleep(1)
        for gpu in list(running):
            job, process, log = running[gpu]
            rc = process.poll()
            if rc is None:
                continue
            log.close()
            Path(job["out"], "returncode.txt").write_text(f"{rc}\n")
            print(f"DONE  gpu={gpu} rc={rc} {job['name']}", flush=True)
            if rc != 0:
                failures.append(job["name"])
            del running[gpu]

    if failures:
        raise RuntimeError("failed jobs: " + ", ".join(failures))

