from pathlib import Path
import sys, json, math, argparse, time, os
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
import math
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "FINER"))

try:
    from models import Finer
except Exception as e:
    Finer = None
    FINER_IMPORT_ERROR = repr(e)
else:
    FINER_IMPORT_ERROR = None


class FourierEncoding(nn.Module):
    def __init__(self, num_bands=0, sigma=1.0, include_input=True):
        super().__init__()
        self.num_bands = int(num_bands)
        self.sigma = float(sigma)
        self.include_input = include_input
        B = torch.randn(self.num_bands, 1) * self.sigma
        self.register_buffer("B", B)

    @property
    def out_dim(self):
        return (1 if self.include_input else 0) + 2 * self.B.shape[0]

    def forward(self, x):
        outs = [x] if self.include_input else []
        if self.B.shape[0] > 0:
            z = 2.0 * math.pi * (x @ self.B.t())
            outs += [torch.sin(z), torch.cos(z)]
        return torch.cat(outs, dim=-1)


class FourierEncodingND(nn.Module):
    def __init__(self, in_dim=2, num_bands=0, sigma=1.0, include_input=True):
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_bands = int(num_bands)
        self.sigma = float(sigma)
        self.include_input = include_input
        if self.num_bands > 0:
            B = torch.randn(self.num_bands, self.in_dim) * self.sigma
        else:
            B = torch.empty(0, self.in_dim)
        self.register_buffer("B", B)

    @property
    def out_dim(self):
        return (self.in_dim if self.include_input else 0) + 2 * self.B.shape[0]

    def forward(self, x):
        outs = [x] if self.include_input else []
        if self.B.shape[0] > 0:
            z = 2.0 * math.pi * (x @ self.B.t())
            outs += [torch.sin(z), torch.cos(z)]
        return torch.cat(outs, dim=-1)


class ResidualFourierAct(nn.Module):
    def __init__(self, K=16, init_std=0.01):
        super().__init__()
        self.coeffs = nn.Parameter(torch.empty(K))
        nn.init.normal_(self.coeffs, mean=0.0, std=init_std)

    def forward(self, u):
        y = u
        for k in range(self.coeffs.numel()):
            y = y + self.coeffs[k] * torch.sin(2.0 * math.pi * (k + 1) * u)
        return y


class FFAct16Audio(nn.Module):
    def __init__(self, width=256, depth=4, num_bands=0, sigma=1.0, K=16, omega_init=30.0):
        super().__init__()
        self.enc = FourierEncoding(num_bands=num_bands, sigma=sigma, include_input=True)
        layers = []
        in_dim = self.enc.out_dim
        for i in range(depth):
            lin = nn.Linear(in_dim if i == 0 else width, width)
            layers.append(lin)
            layers.append(ResidualFourierAct(K))
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(width, 1)
        self.finerstyle_init(omega_init)

    def finerstyle_init(self, omega_init=30.0):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    fan = m.weight.shape[1]
                    bound = math.sqrt(6.0 / fan) / omega_init
                    m.weight.uniform_(-bound, bound)
                    m.bias.uniform_(-bound, bound)

            first = None
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    first = m
                    break
            if first is not None:
                fan = first.weight.shape[1]
                first.weight.uniform_(-1.0 / fan, 1.0 / fan)
                first.bias.uniform_(-1.0 / fan, 1.0 / fan)

    def forward(self, x):
        h = self.enc(x)
        h = self.net(h)
        return self.head(h)


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                bound = math.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class FFSirenAudio(nn.Module):
    def __init__(self, width=256, depth=4, num_bands=0, sigma=1.0, w0=30.0):
        super().__init__()
        self.enc = FourierEncoding(num_bands=num_bands, sigma=sigma, include_input=True)
        layers = [SineLayer(self.enc.out_dim, width, is_first=True, omega_0=w0)]
        for _ in range(depth - 1):
            layers.append(SineLayer(width, width, is_first=False, omega_0=w0))
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(width, 1)
        with torch.no_grad():
            bound = math.sqrt(6 / width) / w0
            self.head.weight.uniform_(-bound, bound)
            self.head.bias.uniform_(-bound, bound)

    def forward(self, x):
        h = self.enc(x)
        h = self.net(h)
        return self.head(h)


class FFFINERAudio(nn.Module):
    def __init__(self, width=256, depth=4, num_bands=0, sigma=1.0, w0=30.0, fbs=20.0):
        super().__init__()
        if Finer is None:
            raise RuntimeError(f"Could not import FINER/models.py: {FINER_IMPORT_ERROR}")
        self.enc = FourierEncoding(num_bands=num_bands, sigma=sigma, include_input=True)
        self.net = Finer(
            in_features=self.enc.out_dim,
            hidden_features=width,
            hidden_layers=depth - 1,
            out_features=1,
            first_omega_0=w0,
            hidden_omega_0=w0,
            first_bias_scale=fbs,
        )

    def forward(self, x):
        h = self.enc(x)
        out = self.net(h)
        if isinstance(out, dict):
            out = out.get("model_out", out.get("output"))
        return out


class FFFINERGeneric(nn.Module):
    def __init__(self, in_dim=2, out_dim=3, width=256, depth=4, num_bands=0, sigma=1.0, w0=30.0, fbs=0.7071):
        super().__init__()
        if Finer is None:
            raise RuntimeError(f"Could not import FINER/models.py: {FINER_IMPORT_ERROR}")
        self.enc = FourierEncodingND(in_dim=in_dim, num_bands=num_bands, sigma=sigma, include_input=True)
        self.net = Finer(
            in_features=self.enc.out_dim,
            hidden_features=width,
            hidden_layers=depth - 1,
            out_features=out_dim,
            first_omega_0=w0,
            hidden_omega_0=w0,
            first_bias_scale=fbs,
        )

    def forward(self, x):
        h = self.enc(x)
        out = self.net(h)
        if isinstance(out, dict):
            out = out.get("model_out", out.get("output"))
        return out


def infer_id(row, idx):
    for k in ["sample_id", "id", "stem", "name"]:
        if k in row and pd.notna(row[k]):
            return str(row[k])
    for k in ["wav_path", "path", "audio_path", "file"]:
        if k in row and pd.notna(row[k]):
            return Path(str(row[k])).stem
    return f"sample_{idx:04d}"


def get_wav(row):
    for k in ["wav_path", "path", "audio_path", "file"]:
        if k in row and pd.notna(row[k]):
            return str(row[k])
    raise KeyError(f"No wav path column in {list(row.index)}")



def resolve_audio_file_for_release(path):
    p = Path(str(path))
    if p.is_absolute():
        return str(p)

    roots = []
    for key in ["LSA_AUDIO_DATA_ROOT", "LSA_AUDIO_SOURCE_ROOT"]:
        val = os.environ.get(key)
        if val:
            roots.append(Path(val))

    roots.extend([
        Path.cwd(),
        ROOT,
    ])

    for root in roots:
        cand = root / p
        if cand.exists():
            return str(cand)

    return str(p)

def load_audio(path, sample_rate=16000, n_samples=16000):
    path = resolve_audio_file_for_release(path)
    y, sr = sf.read(path)
    if int(sr) != int(sample_rate):
        # Audit version: resample source audio so we can test a denser coordinate grid.
        g = math.gcd(int(sr), int(sample_rate))
        up = int(sample_rate) // g
        down = int(sr) // g
        y = resample_poly(y, up, down).astype(np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    y = y[:n_samples] if len(y) >= n_samples else np.pad(y, (0, n_samples - len(y)))

    # Unified official regime for this sweep.
    y = y - y.mean()
    y = y / (np.max(np.abs(y)) + 1e-8)

    x = np.linspace(-1.0, 1.0, n_samples, dtype=np.float32)[:, None]
    return x, y[:, None]


def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    return -10.0 * np.log10(mse + 1e-12)


def make_model(args):
    if args.method in ("act16", "lsa"):
        return FFAct16Audio(
            width=args.width,
            depth=args.depth,
            num_bands=args.num_bands,
            sigma=args.sigma,
            K=args.K,
            omega_init=args.w0,
        )
    if args.method == "finer":
        if hasattr(args, "in_dim") or hasattr(args, "out_dim"):
            return FFFINERGeneric(
                in_dim=getattr(args, "in_dim", 1),
                out_dim=getattr(args, "out_dim", 1),
                width=args.width,
                depth=args.depth,
                num_bands=args.num_bands,
                sigma=args.sigma,
                w0=args.w0,
                fbs=args.fbs,
            )
        return FFFINERAudio(
            width=args.width,
            depth=args.depth,
            num_bands=args.num_bands,
            sigma=args.sigma,
            w0=args.w0,
            fbs=args.fbs,
        )
    if args.method == "siren":
        return FFSirenAudio(
            width=args.width,
            depth=args.depth,
            num_bands=args.num_bands,
            sigma=args.sigma,
            w0=args.w0,
        )
    raise ValueError(args.method)



def freeze_act_coeffs_v1(model, args):
    """Freeze only LSA harmonic coefficient tensors.

    In our LSA audio model, the trainable activation coefficients are the
    small K-length parameter vectors. Linear-layer weights/biases have much
    larger shapes, and Fourier-feature matrices are buffers.
    """
    frozen = []
    K = int(getattr(args, "K", -1))
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if K > 0 and int(param.numel()) == K:
            param.requires_grad_(False)
            with torch.no_grad():
                flat = param.detach().float().flatten().cpu()
                frozen.append({
                    "name": name,
                    "shape": tuple(param.shape),
                    "numel": int(param.numel()),
                    "mean": float(flat.mean()),
                    "std": float(flat.std(unbiased=False)),
                })

    if getattr(args, "method", None) == "act16" and not frozen:
        raise RuntimeError(
            "freeze_act_coeffs requested, but no K-sized activation coefficient "
            "parameters were found. Inspect model.named_parameters()."
        )

    print("=== FROZEN_LSA_ACT_COEFFS ===", flush=True)
    for r in frozen:
        print(
            f"{r['name']} shape={r['shape']} numel={r['numel']} "
            f"mean={r['mean']:.6g} std={r['std']:.6g}",
            flush=True,
        )
    print(f"FROZEN_LSA_ACT_COEFF_COUNT={len(frozen)}", flush=True)
    print("=== END_FROZEN_LSA_ACT_COEFFS ===", flush=True)


def train_one(wav, args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x_np, y_np = load_audio(wav, args.sample_rate, args.n_samples)
    x = torch.from_numpy(x_np).to(args.device)
    y = torch.from_numpy(y_np).to(args.device)

    model = make_model(args).to(args.device)

    if args.method in ("act16", "lsa"):
        act_params, other_params = [], []
        for name, p in model.named_parameters():
            if "coeffs" in name:
                act_params.append(p)
            else:
                other_params.append(p)
        opt = torch.optim.Adam([
            {"params": other_params, "lr": args.lr},
            {"params": act_params, "lr": args.lr * args.act_lr_mult},
        ])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=args.lr_decay)

    best = {"psnr": -1e9, "step": -1, "state_dict": None}
    t0 = time.time()

    coeff0 = None
    if args.method in ("act16", "lsa"):
        coeffs = [p.detach().cpu().flatten() for n, p in model.named_parameters() if "coeffs" in n]
        coeff0 = torch.cat(coeffs) if coeffs else None

    convergence_rows = []

    for step in range(args.max_steps):
        if getattr(args, "batch_size", 0) and args.batch_size > 0 and args.batch_size < x.shape[0]:
            idx = torch.randint(0, x.shape[0], (args.batch_size,), device=x.device)
            xb = x[idx]
            yb = y[idx]
        else:
            xb = x
            yb = y

        pred_batch = model(xb)
        loss = ((pred_batch - yb) ** 2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (step + 1) % args.decay_every == 0:
            sched.step()

        if step % args.log_every == 0 or step == args.max_steps - 1:
            with torch.no_grad():
                pred_full = model(x)
                cur = psnr(pred_full.detach(), y)
                convergence_rows.append({
                    "step": int(step),
                    "psnr": float(cur),
                    "loss": float(loss.detach().cpu().item()),
                })
            if cur > best["psnr"]:
                best = {
                    "psnr": float(cur),
                    "step": int(step),
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                }

    coeff_delta = None
    if args.method == "act16" and coeff0 is not None:
        coeffs = [p.detach().cpu().flatten() for n, p in model.named_parameters() if "coeffs" in n]
        coeff1 = torch.cat(coeffs)
        coeff_delta = float(torch.norm(coeff1 - coeff0))

    return best, time.time() - t0, coeff_delta, convergence_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nsynth", "librispeech"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--method", required=True, choices=["lsa", "act16", "finer", "siren"])
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--n_samples", type=int, default=16000)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--num_bands", type=int, required=True)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--w0", type=float, default=30.0)
    ap.add_argument("--fbs", type=float, default=20.0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--act_lr_mult", type=float, default=1.0)
    ap.add_argument("--lr_decay", type=float, default=0.2)
    ap.add_argument("--decay_every", type=int, default=5000)
    ap.add_argument("--max_steps", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=0, help="Random coordinate minibatch size; 0 means full coordinate grid.")
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--freeze_act_coeffs", action="store_true", help="Freeze LSA harmonic activation coefficients a_k at initialization.")
    ap.add_argument("--limit", type=int, default=0, help="Optional number of manifest rows to run; 0 means all.")
    ap.add_argument("--eval_chunk", type=int, default=0, help="Accepted for compatibility; historical runner evaluates full grid.")
    ap.add_argument("--force", action="store_true", help="Accepted for compatibility; historical runner overwrites outputs.")
    args = ap.parse_args()
    if args.method == "lsa":
        args.method = "act16"

    out_runs = ROOT / "outputs/runs" / args.group
    out_runs.mkdir(parents=True, exist_ok=True)
    out_sum = ROOT / "outputs/summaries"
    out_sum.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest)
    if getattr(args, "limit", 0) and args.limit > 0:
        df = df.head(args.limit).copy()
    rows = []

    for idx, row in df.iterrows():
        sid = infer_id(row, idx)
        run_dir = out_runs / f"{args.group}_{sid}"
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_dir / "summary.json"

        if summary_path.exists():
            d = json.loads(summary_path.read_text())
            rows.append(d)
            print("SKIP", sid, d["psnr"], flush=True)
            continue

        best, wall, coeff_delta, convergence_rows = train_one(get_wav(row), args)

        d = {
            "status": "ok",
            "dataset": args.dataset,
            "sample_id": sid,
            "method": args.method,
            "model": {
                "act16": "FF+Act16" if args.num_bands > 0 else "Act16 no FF",
                "finer": "FF+FINER" if args.num_bands > 0 else "FINER no FF",
                "siren": "FF+SIREN" if args.num_bands > 0 else "SIREN no FF",
            }[args.method],
            "unified_sweep": True,
            "target_regime": "zero_mean_unit_peak_linear_output",
            "coordinate_regime": "linspace(-1,1,n_samples)",
            "sample_rate": args.sample_rate,
            "n_samples": args.n_samples,
            "duration_sec": args.n_samples / args.sample_rate,
            "num_bands": args.num_bands,
            "sigma": None if args.num_bands == 0 else args.sigma,
            "K": args.K if args.method in ("act16", "lsa") else None,
            "w0": args.w0,
            "fbs": args.fbs if args.method == "finer" else None,
            "lr": args.lr,
            "act_lr_mult": args.act_lr_mult if args.method in ("act16", "lsa") else None,
            "lr_decay": args.lr_decay,
            "decay_every": args.decay_every,
            "width": args.width,
            "depth": args.depth,
            "best_step": best["step"],
            "psnr": best["psnr"],
            "act_coeff_delta_l2": coeff_delta,
            "train_wall_clock_sec": wall,
        }

        summary_path.write_text(json.dumps(d, indent=2))
        if convergence_rows:
            pd.DataFrame(convergence_rows).to_csv(run_dir / "convergence.csv", index=False)
        torch.save({"model": best["state_dict"], "meta": d}, run_dir / "checkpoint_best.pt")
        rows.append(d)
        print("DONE", sid, d["psnr"], "step", d["best_step"], flush=True)

    out_csv = out_sum / f"{args.group}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("WROTE", out_csv, flush=True)


if __name__ == "__main__":
    main()
