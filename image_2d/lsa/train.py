import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from lsa.common import load_image, psnr_01, render, save_prediction, set_seed, ssim_01, write_json, write_metrics
from lsa.models import LSAImageMLP
from third_party.finer import Finer, Siren
from third_party.sl2a import SL2A
from third_party.staf import STAF


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def optimizer_for_lsa(model, lr, act_lr_mult):
    activation = []
    other = []
    for name, parameter in model.named_parameters():
        if "coeffs" in name:
            activation.append(parameter)
        else:
            other.append(parameter)
    return torch.optim.Adam([
        {"params": other, "lr": lr},
        {"params": activation, "lr": lr * act_lr_mult},
    ])


def scheduler(optimizer, kind, steps):
    if kind == "constant":
        fn = lambda step: 1.0
    elif kind == "decay0p1":
        fn = lambda step: 0.1 ** min(step / steps, 1.0)
    elif kind == "decay0p05":
        fn = lambda step: 0.05 ** min(step / steps, 1.0)
    else:
        raise ValueError(kind)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def build_model(method, params):
    if method == "lsa":
        return LSAImageMLP(
            width=params["width"],
            hidden_layers=params["hidden_layers"],
            harmonics=params["harmonics"],
            residual=True,
        )
    if method == "finer":
        return Finer(
            in_features=2,
            hidden_features=params["width"],
            hidden_layers=params["hidden_layers"],
            out_features=3,
            first_omega_0=params["w0"],
            hidden_omega_0=params["w0"],
            first_bias_scale=params["first_bias_scale"],
        )
    if method == "siren":
        return Siren(
            in_features=2,
            hidden_features=params["width"],
            hidden_layers=params["hidden_layers"],
            out_features=3,
            first_omega_0=params["w0"],
            hidden_omega_0=params["w0"],
        )
    if method == "staf":
        return STAF(
            in_features=2,
            hidden_features=params["width"],
            hidden_layers=params["hidden_layers"],
            out_features=3,
            outermost_linear=True,
            first_omega_0=params["first_omega"],
            hidden_omega_0=params["hidden_omega"],
            tau=params["tau"],
            skip_conn=True,
        )
    if method == "sl2a":
        return SL2A(
            in_features=2,
            hidden_features=params["width"],
            hidden_layers=params["hidden_layers"],
            out_features=3,
            deg=params["degree"],
            outermost_linear=True,
            nonlinearity="relu",
            rank=params["rank"],
            init_method="xavier_uniform",
            linear_init_type="kaiming_uniform",
        )
    raise ValueError(method)


def run(job):
    method = job["method"]
    params = job["params"]
    common = job["common"]
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)

    deterministic = method == "sl2a"
    set_seed(common["seed"], deterministic=deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coords, target01, target11, height, width = load_image(job["image"], device)
    model = build_model(method, params).to(device)

    expected_params = params.get("parameters")
    observed_params = parameter_count(model)
    if expected_params is not None and observed_params != expected_params:
        raise RuntimeError(f"parameter count: {observed_params} != {expected_params}")

    if method == "lsa":
        optimizer = optimizer_for_lsa(model, params["lr"], params["act_lr_mult"])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    schedule = scheduler(optimizer, params.get("schedule", "decay0p1"), common["steps"])
    count = coords.shape[0]
    best = {"psnr": -1.0, "ssim": -1.0, "step": -1}
    rows = []
    started = time.time()

    for step in range(1, common["steps"] + 1):
        indices = torch.randint(0, count, (min(common["batch_size"], count),), device=device)
        prediction = model(coords[indices])
        loss = F.mse_loss(prediction, target11[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()

        evaluate = step % common["eval_every"] == 0 or step == common["steps"]
        if method != "staf":
            evaluate = evaluate or step == 1

        if evaluate:
            prediction01 = render(model, coords, height, width, common["eval_chunk"])
            psnr = psnr_01(target01, prediction01)
            ssim = ssim_01(target01, prediction01)
            row = {
                "step": step,
                "psnr": psnr,
                "ssim": ssim,
                "train_loss": float(loss.item()),
                "elapsed_sec": time.time() - started,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            if psnr > best["psnr"]:
                best = {"psnr": psnr, "ssim": ssim, "step": step}
                if common.get("save_best_image", False):
                    save_prediction(prediction01, out / "best.png")

    summary = {
        "image": Path(job["image"]).stem,
        "method": method,
        "params": params,
        "common": common,
        "best_psnr": best["psnr"],
        "best_ssim": best["ssim"],
        "best_step": best["step"],
        "num_params": observed_params,
    }
    write_metrics(out / "metrics.csv", rows)
    write_json(out / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text())
    run(job)


if __name__ == "__main__":
    main()
