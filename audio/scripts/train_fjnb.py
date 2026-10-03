import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

try:
    import soundfile as sf
except Exception as e:
    raise RuntimeError("This script requires soundfile. Install pysoundfile/soundfile in the active env.") from e


def add_fkan_path(root: Path):
    candidates = [
        root / "third_party" / "fKAN",
        root / "fKAN",
        root / "fkan",
    ]
    for c in candidates:
        if (c / "fkan" / "torch.py").exists():
            sys.path.insert(0, str(c))
            return c
    raise FileNotFoundError(
        "Could not find fKAN package. Expected one of: "
        + ", ".join(str(c) for c in candidates)
    )


def read_audio(path, sample_rate, n_samples):
    y, sr = sf.read(str(path), always_2d=True)
    y = y.astype(np.float32)
    y = y.mean(axis=1)

    if sr != sample_rate:
        try:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate).astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Need resampling {sr}->{sample_rate}, but librosa failed for {path}") from e

    if len(y) < n_samples:
        y = np.pad(y, (0, n_samples - len(y)))
    else:
        y = y[:n_samples]

    return y.astype(np.float32)


def get_wav_path(row):
    for key in ["wav_path", "path", "audio_path", "filepath", "file_path"]:
        if key in row and isinstance(row[key], str) and row[key]:
            return row[key]
    raise KeyError(f"No path column found in manifest row. Columns: {list(row.keys())}")


def get_sample_id(row, idx):
    for key in ["sample_id", "id", "clip_id", "name"]:
        if key in row and str(row[key]):
            return str(row[key])
    p = Path(get_wav_path(row))
    return p.stem if p.name else f"sample_{idx:04d}"


class FourierEncoding(nn.Module):
    def __init__(self, in_dim=1, num_bands=0, sigma=1.0, include_input=True, seed=0):
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_bands = int(num_bands)
        self.sigma = float(sigma)
        self.include_input = bool(include_input)

        if self.num_bands > 0:
            g = torch.Generator()
            g.manual_seed(int(seed))
            B = torch.randn(self.num_bands, self.in_dim, generator=g) * self.sigma
        else:
            B = torch.empty(0, self.in_dim)
        self.register_buffer("B", B)

    @property
    def out_dim(self):
        return (self.in_dim if self.include_input else 0) + 2 * self.num_bands

    def forward(self, x):
        outs = [x] if self.include_input else []
        if self.num_bands > 0:
            z = 2.0 * math.pi * (x @ self.B.t())
            outs += [torch.sin(z), torch.cos(z)]
        return torch.cat(outs, dim=-1)


class FKANINR(nn.Module):
    def __init__(self, fJNB, in_dim=1, out_dim=1, width=256, depth=4,
                 degree=3, num_bands=0, sigma=1.0, ff_seed=0):
        super().__init__()
        self.enc = FourierEncoding(
            in_dim=in_dim,
            num_bands=num_bands,
            sigma=sigma,
            include_input=True,
            seed=ff_seed,
        )
        layers = []
        d = self.enc.out_dim
        for i in range(depth):
            layers.append(nn.Linear(d if i == 0 else width, width))
            layers.append(fJNB(int(degree)))
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(width, out_dim)

    def forward(self, x):
        return self.head(self.net(self.enc(x)))


def psnr_unit(mse):
    return -10.0 * math.log10(max(float(mse), 1e-20))


def train_one(y_np, args, fJNB, sample_id, outdir):
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    n = len(y_np)
    x_np = np.linspace(-1.0, 1.0, n, dtype=np.float32)[:, None]
    x_full = torch.from_numpy(x_np).to(device)
    y_full = torch.from_numpy(y_np[:, None]).to(device)

    model = FKANINR(
        fJNB=fJNB,
        in_dim=1,
        out_dim=1,
        width=args.width,
        depth=args.depth,
        degree=args.degree,
        num_bands=args.num_bands,
        sigma=args.sigma,
        ff_seed=args.ff_seed,
    ).to(device).float()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = {
        "psnr": -1e9,
        "mse": None,
        "step": None,
    }

    t0 = time.time()
    for step in range(args.max_steps + 1):
        if args.batch_size and args.batch_size > 0 and args.batch_size < n:
            idx = torch.randint(0, n, (args.batch_size,), device=device)
            xb = x_full[idx]
            yb = y_full[idx]
        else:
            xb = x_full
            yb = y_full

        opt.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = (pred - yb).pow(2).mean()
        loss.backward()
        opt.step()

        if step % args.log_every == 0 or step == args.max_steps:
            with torch.no_grad():
                pred_full = model(x_full)
                mse = (pred_full - y_full).pow(2).mean().item()
                cur = psnr_unit(mse)
            if cur > best["psnr"]:
                best = {"psnr": cur, "mse": mse, "step": step}
            print(f"DONE_STEP {sample_id} step={step} psnr={cur:.6f} best={best['psnr']:.6f}", flush=True)

    wall = time.time() - t0

    outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "sample_id": sample_id,
        "method": "fkan",
        "degree": args.degree,
        "num_bands": args.num_bands,
        "sigma": args.sigma,
        "width": args.width,
        "depth": args.depth,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "best_step": best["step"],
        "psnr": best["psnr"],
        "mse": best["mse"],
        "wall_seconds": wall,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "outputs")
    p.add_argument("--dataset", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--group", required=True)
    p.add_argument("--sample_rate", type=int, default=48000)
    p.add_argument("--n_samples", type=int, default=48000)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--num_bands", type=int, default=0)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--ff_seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    fkan_root = add_fkan_path(root)
    from fkan.torch import FractionalJacobiNeuralBlock as fJNB

    print("USING_FKAN", fkan_root, flush=True)
    print("ARGS", vars(args), flush=True)

    df = pd.read_csv(args.manifest)
    rows = []

    for idx, row in df.iterrows():
        rd = row.to_dict()
        sid = get_sample_id(rd, idx)
        wav = Path(get_wav_path(rd))
        if not wav.is_absolute():
            wav = root / wav
        y = read_audio(wav, args.sample_rate, args.n_samples)

        safe_sid = "".join(c if c.isalnum() or c in "-_." else "_" for c in sid)
        outdir = args.output_root / "runs" / args.group / f"{args.group}_{safe_sid}"
        print(f"START {idx} {sid} wav={wav}", flush=True)
        summary = train_one(y, args, fJNB, sid, outdir)
        rows.append(summary)
        print(f"DONE {sid} {summary['psnr']:.6f} step {summary['best_step']}", flush=True)

    out_summary = args.output_root / "summaries" / f"{args.group}.csv"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_summary, index=False)
    print("WROTE", out_summary, flush=True)


if __name__ == "__main__":
    main()
