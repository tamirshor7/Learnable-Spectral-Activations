from __future__ import annotations

import argparse
import os
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

try:
    from scipy.signal import resample_poly
except Exception:
    resample_poly = None


ROOT = Path(__file__).resolve().parents[1]


def find_sl2a_dir() -> Path:
    for name in ["SL2A-INR-main", "SL2A-INR", "SL2A", "sl2a"]:
        d = ROOT / name
        if (d / "SL2A_INR.py").exists() and (d / "ChebyKANLayer.py").exists():
            return d
    raise RuntimeError("Could not find SL2A baseline code.")


SL2A_DIR = find_sl2a_dir()
sys.path.insert(0, str(SL2A_DIR))
from SL2A_INR import SL2A  # noqa: E402


def sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))


def get_sample_id(row: pd.Series, fallback: int) -> str:
    for key in ["sample_id", "clip_id", "id", "utt_id", "name"]:
        if key in row and pd.notna(row[key]):
            return str(row[key])
    for key in ["wav_path", "path", "audio_path", "filename", "file"]:
        if key in row and pd.notna(row[key]):
            return Path(str(row[key])).stem
    return f"idx{fallback:03d}"


def load_audio(row: pd.Series, sample_rate: int, n_samples: int, normalization: str) -> torch.Tensor:
    path = None
    for key in ["wav_path", "path", "audio_path", "filename", "file"]:
        if key in row and pd.notna(row[key]):
            path = Path(str(row[key]))
            break
    if path is None:
        raise RuntimeError(f"No audio path column found. Available columns: {list(row.index)}")

    if not path.is_absolute():
        p2 = ROOT / path
        if p2.exists():
            path = p2

    y, sr = sf.read(str(path), always_2d=False)
    y = np.asarray(y)

    if y.ndim == 2:
        y = y.mean(axis=1)

    y = y.astype(np.float32)

    if int(sr) != int(sample_rate):
        if resample_poly is None:
            raise RuntimeError(f"Need scipy.signal.resample_poly for sr={sr} -> {sample_rate}.")
        import math as _math
        g = _math.gcd(int(sr), int(sample_rate))
        up = int(sample_rate) // g
        down = int(sr) // g
        y = resample_poly(y, up, down).astype(np.float32)

    if len(y) < n_samples:
        y = np.pad(y, (0, n_samples - len(y)))
    else:
        y = y[:n_samples]

    if normalization == "zero_mean_unit_peak":
        y = y - float(np.mean(y))
        peak = float(np.max(np.abs(y)))
        if peak > 0:
            y = y / peak
    elif normalization == "unit_peak":
        peak = float(np.max(np.abs(y)))
        if peak > 0:
            y = y / peak
    elif normalization == "none":
        pass
    else:
        raise ValueError(normalization)

    return torch.from_numpy(y).float().view(-1, 1)


def psnr_from_mse(mse: float) -> float:
    return float(-10.0 * math.log10(max(float(mse), 1e-30)))


def count_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def train_one(y_cpu: torch.Tensor, args) -> dict:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    y = y_cpu.to(device)
    n = y.shape[0]
    t = torch.linspace(-1.0, 1.0, n, device=device).view(-1, 1)

    model = SL2A(
        in_features=1,
        out_features=1,
        hidden_layers=args.hidden_layers,
        hidden_features=args.hidden_features,
        deg=args.deg,
        rank=args.rank,
        nonlinearity=args.nonlinearity,
        init_method=args.init_method,
        linear_init_type=args.linear_init_type,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda step: args.scheduler_b ** min(step / max(args.max_steps, 1), 1.0),
        )

    best_psnr = -1e30
    best_mse = None
    best_step = None

    t0 = time.time()

    for step in range(args.max_steps + 1):
        if args.batch_size and args.batch_size > 0 and args.batch_size < n:
            idx = torch.randint(0, n, (args.batch_size,), device=device)
            tb = t[idx]
            yb = y[idx]
        else:
            tb = t
            yb = y

        pred = model(tb)
        loss = (pred - yb).pow(2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if scheduler is not None:
            scheduler.step()

        if step % args.log_every == 0 or step == args.max_steps:
            with torch.no_grad():
                pred_full = model(t)
                mse = float((pred_full - y).pow(2).mean().detach().cpu())
                cur_psnr = psnr_from_mse(mse)

            if cur_psnr > best_psnr:
                best_psnr = cur_psnr
                best_mse = mse
                best_step = step

            print(f"step={step} psnr={cur_psnr:.6f} best={best_psnr:.6f}", flush=True)

    wall = time.time() - t0

    return {
        "psnr": best_psnr,
        "mse": best_mse,
        "best_step": best_step,
        "wall_seconds": wall,
        "params": count_params(model),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nsynth", "librispeech"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--group", required=True)

    ap.add_argument("--sample_rate", type=int, default=48000)
    ap.add_argument("--n_samples", type=int, default=48000)
    ap.add_argument("--normalization", default="zero_mean_unit_peak", choices=["zero_mean_unit_peak", "unit_peak", "none"])

    ap.add_argument("--hidden_features", type=int, default=256)
    ap.add_argument("--hidden_layers", type=int, default=3)
    ap.add_argument("--deg", type=int, default=256)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--nonlinearity", type=str, default="relu")
    ap.add_argument("--init_method", type=str, default="xavier_uniform")
    ap.add_argument("--linear_init_type", type=str, default="kaiming_uniform")

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--scheduler", action="store_true")
    ap.add_argument("--scheduler_b", type=float, default=0.1)

    ap.add_argument("--batch_size", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    out_root = Path(os.environ.get("LSA_AUDIO_OUTPUT_ROOT", str(ROOT / "outputs"))) / "runs" / args.group
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, row in manifest.iterrows():
        sample_id = get_sample_id(row, idx)
        safe_id = sanitize(sample_id)
        run_dir = out_root / f"{args.group}_{safe_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        y = load_audio(row, args.sample_rate, args.n_samples, args.normalization)
        res = train_one(y, args)

        summary = {
            "dataset": args.dataset,
            "sample_id": sample_id,
            "group": args.group,
            "method": "sl2a",
            "model": "official_sl2a",
            "sample_rate": args.sample_rate,
            "n_samples": args.n_samples,
            "normalization": args.normalization,
            "hidden_features": args.hidden_features,
            "hidden_layers": args.hidden_layers,
            "deg": args.deg,
            "rank": args.rank,
            "nonlinearity": args.nonlinearity,
            "init_method": args.init_method,
            "linear_init_type": args.linear_init_type,
            "lr": args.lr,
            "scheduler": args.scheduler,
            "scheduler_b": args.scheduler_b if args.scheduler else None,
            "batch_size": args.batch_size,
            "max_steps": args.max_steps,
            "seed": args.seed,
            **res,
        }

        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
        rows.append(summary)
        print(f"DONE {sample_id} {res['psnr']:.6f} step {res['best_step']} params {res['params']}", flush=True)

    out_sum = Path(os.environ.get("LSA_AUDIO_OUTPUT_ROOT", str(ROOT / "outputs"))) / "summaries" / f"{args.group}.csv"
    out_sum.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_sum, index=False)
    print("WROTE", out_sum)


if __name__ == "__main__":
    main()
