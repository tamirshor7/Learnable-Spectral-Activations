import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim_metric


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_image(path, device):
    image = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    height, width = image.shape[:2]
    target01 = torch.from_numpy(image).to(device)
    target11 = target01 * 2.0 - 1.0
    ys = torch.linspace(-1, 1, height, device=device)
    xs = torch.linspace(-1, 1, width, device=device)
    y, x = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([x, y], dim=-1).reshape(-1, 2)
    return coords, target01, target11.reshape(-1, 3), height, width


def psnr_01(target, prediction):
    mse = torch.mean((target - prediction) ** 2).item()
    return -10.0 * math.log10(max(mse, 1e-12))


def ssim_01(target, prediction):
    return float(ssim_metric(
        target.detach().cpu().numpy(),
        prediction.detach().cpu().numpy(),
        channel_axis=2,
        data_range=1.0,
    ))


@torch.no_grad()
def render(model, coords, height, width, chunk):
    model.eval()
    outputs = []
    for start in range(0, coords.shape[0], chunk):
        outputs.append(model(coords[start:start + chunk]).detach())
    model.train()
    pred11 = torch.cat(outputs, dim=0).reshape(height, width, 3)
    return (pred11 / 2.0 + 0.5).clamp(0, 1)


def save_prediction(prediction, path):
    array = prediction.detach().cpu().numpy().astype(np.float32)
    Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8)).save(path)


def write_metrics(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")
