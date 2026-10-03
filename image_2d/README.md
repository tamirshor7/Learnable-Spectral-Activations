# 2D images

LSA, FINER, SIREN, STAF, and SL2A on the 24 Kodak images.

## Data

From the repository root:

```bash
cd image_2d
python scripts/download_kodak.py
```

Images are saved to `data/kodak/` and checked against the supplied SHA-256 hashes. Existing images can be selected with `--data-root /path/to/kodak` when running an experiment.

## Run

Run the core comparison first, then either baseline comparison:

```bash
python run_core.py --gpus 0
python run_staf.py --gpus 0
python run_sl2a.py --gpus 0
```

Use `--gpus 0,1` to distribute fits across two GPUs. Each script writes its logs, per-image results, and summary to `outputs/kodak_core`, `outputs/kodak_staf`, or `outputs/kodak_sl2a`. Completed fits are skipped. Use `--no-resume` to recompute them.

`configs/` contains the experiment settings. Core evaluates three LSA configurations alongside FINER and SIREN. STAF evaluates 56 configurations per image. Each SL2A variant selects its learning rate on six calibration images before fitting all 24 images. Summary files include the 18-image holdout results.

## Results

Compare all results, or check the core experiment:

```bash
python check_results.py
python check_results.py --suite core
```

For a custom data location, pass the same `--data-root` used for training. The check compares per-image PSNR and SSIM with `reference/`, using tolerances of 0.02 dB and 0.002. It also checks the selected LSA configuration and SL2A learning rates.
