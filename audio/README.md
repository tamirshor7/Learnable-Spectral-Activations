# Audio

LSA, FINER, SIREN, fJNB, and SL2A on 10 NSynth clips and 40 LibriSpeech clips. The clip lists are in `manifests/`.

## Data

From the repository root:

```bash
cd audio
python download_data.py
```

This downloads LibriSpeech dev-clean and NSynth validation. If you already have the datasets, place or symlink them at:

```text
data/librispeech_raw/LibriSpeech/dev-clean/
data/nsynth-valid/audio/
```

## Run

Run the methods separately:

```bash
GPU=0 bash run_lsa.sh
GPU=0 bash run_finer.sh
GPU=0 bash run_siren.sh
GPU=0 bash run_fjnb.sh
GPU=0 bash run_sl2a.sh
```

Each script runs both datasets. fJNB includes raw coordinates and Fourier features. SL2A includes all four degree/rank settings. Change `GPU` to select a device. Logs, per-clip results, and dataset summaries are written to `outputs/`. Running a script again recomputes that method's fits.

## Results

Compare all results, or check one method:

```bash
python check_results.py
python check_results.py --method lsa
```

The check requires the exact clip set and mean PSNR within 0.01 dB of `reference/results.csv`.

The released settings use 48,000 samples per clip. Core methods and SL2A use zero-mean, unit-peak targets. fJNB uses decoded amplitudes. Core methods perform 5,000 updates, and fJNB/SL2A perform 5,001. The scripts retain their original preprocessing and evaluation intervals.
