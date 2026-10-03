# Learnable Spectral Activations: 2D image reproduction

This repository reproduces the Kodak image-fitting experiments from the main paper. It contains only the code needed for the frozen 2D/image results.

Included experiments:

- residual LSA vs. FINER and SIREN on Kodak
- per-image LSA vs. STAF
- LSA vs. full and parameter-matched SL2A

Poisson reconstruction, harmonic-count ablations, WIRE, MFN, fJNB, theory diagnostics, and appendix-only experiments are intentionally excluded.

## Frozen protocol

All image fits use raw coordinates in `[-1, 1]^2`, RGB targets in `[-1, 1]`, width 256, four nonlinear layers, Adam, seed 0, 5,000 updates, minibatches of 65,536 pixels, and full-image evaluation every 100 updates.

LSA uses

`phi(u) = u + sum_k a_k sin(2*pi*k*u)`

with `K=32` and coefficient initialization scale `1e-3`. The globally selected LSA configuration is `lr=5e-4`, `act_lr_mult=2`. LSA has 199,043 trainable parameters: 198,915 linear parameters and 128 activation coefficients.

The frozen reference environment was Python 3.10.19, PyTorch 2.10.0+cu130, CUDA 13.0, cuDNN 9.15.1, NumPy 2.2.6, Pillow 12.0.0, scikit-image 0.25.2, and PyYAML 6.0.3 on NVIDIA RTX A6000 GPUs with driver 580.173.02.

## Setup

Create the environment:

```bash
conda create -n lsa-image python=3.10.19 -y
conda activate lsa-image
python -m pip install --upgrade pip
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

The exact frozen torch build is `2.10.0+cu130`. Other CUDA/PyTorch stacks can give small numerical differences.

## Kodak data

Download and verify the 24 Kodak PNGs:

```bash
python scripts/download_kodak.py --out data/kodak
```

The files are downloaded from the Kodak Lossless True Color Image Suite and verified against the SHA-256 hashes in `reference/kodak_sha256.json`.

Expected layout:

```text
data/kodak/
  kodim01.png
  ...
  kodim24.png
```

## 1. LSA, FINER, and SIREN

Run the full frozen core experiment. On an 8-GPU machine:

```bash
python scripts/run.py \
  --config configs/kodak_core.yaml \
  --gpus 0,1,2,3,4,5,6,7
```

One GPU is also valid:

```bash
python scripts/run.py --config configs/kodak_core.yaml --gpus 0
```

Aggregate the results:

```bash
python scripts/summarize.py core --config configs/kodak_core.yaml
```

Frozen reference means over all 24 Kodak images:

| Method | Mean PSNR |
| --- | ---: |
| LSA, globally selected | 38.1102 |
| FINER | 37.3173 |
| SIREN | 34.9900 |

The six calibration images are `kodim02`, `kodim05`, `kodim07`, `kodim10`, `kodim13`, and `kodim23`. The three LSA candidates are listed explicitly in `configs/kodak_core.yaml`. The selected candidate is the one with the highest calibration mean PSNR.

Per-image selection over the same three residual-LSA candidates gives 38.3085 dB over all 24 images.

## 2. LSA vs. STAF

The STAF protocol evaluates 56 configurations per image: the original 48-config grid plus the eight high-learning-rate boundary configurations.

```bash
python scripts/run.py \
  --config configs/kodak_staf.yaml \
  --gpus 0,1,2,3,4,5,6,7

python scripts/summarize.py staf \
  --config configs/kodak_staf.yaml \
  --core-output outputs/kodak_core
```

Frozen reference results:

| Method | Candidates/image | Mean PSNR |
| --- | ---: | ---: |
| residual LSA | 3 | 38.3085 |
| STAF | 56 | 35.4251 |

Residual LSA wins 24/24 images in PSNR and 24/24 in SSIM. The mean paired PSNR difference is 2.8834 dB.

## 3. LSA vs. SL2A

The SL2A experiment evaluates the official architecture in two settings:

- full: degree 256, rank 128, 330,243 parameters
- parameter matched: degree 192, rank 64, 199,171 parameters

For each setting, learning rate is selected from `{5e-4, 1e-3, 2e-3, 4e-3, 8e-3}` on the same six calibration images, then frozen for the 24-image run.

```bash
python scripts/run.py \
  --config configs/kodak_sl2a.yaml \
  --gpus 0,1,2,3,4,5,6,7

python scripts/summarize.py sl2a \
  --config configs/kodak_sl2a.yaml \
  --core-output outputs/kodak_core
```

Both SL2A settings select `lr=4e-3`.

Frozen 18-image holdout results:

| Method | Parameters | Mean PSNR |
| --- | ---: | ---: |
| residual LSA | 199,043 | 38.0450 |
| SL2A full | 330,243 | 38.2565 |
| SL2A parameter matched | 199,171 | 35.9098 |

LSA is 0.2115 dB below full SL2A on average and wins 8/18 images. Against the nearly parameter-matched SL2A model, LSA gains 2.1352 dB and wins 18/18 images.

## Reference check

After all three experiments finish:

```bash
python scripts/check_reference.py
```

The default tolerance is 0.02 dB. Frozen per-image values are in `reference/`.

## Repository structure

```text
configs/       frozen experiment grids
lsa/           LSA model, training loop, metrics, and I/O
scripts/       data download, experiment runner, aggregation, reference check
third_party/   minimal FINER/SIREN, STAF, and SL2A model definitions
reference/     frozen hashes and numerical reference results
```

Third-party source provenance and commit hashes are recorded in `THIRD_PARTY.md`.
