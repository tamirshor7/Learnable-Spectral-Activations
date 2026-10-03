# Learnable Spectral Activations: main-paper audio reproduction

Private audio-only release for reproducing the main-paper audio results.

Included experiments: LSA, FINER, SIREN, fJNB/fKAN, SL2A.

STAF and WIRE are not included because they are supplementary comparisons, not main-paper audio baselines.

## Setup

Install CUDA PyTorch for your machine, then run:

```bash
pip install -r requirements.txt
```

## Data

From the repository root:

```bash
bash scripts/download_data.sh
```

Expected layout:

```text
data/librispeech_raw/LibriSpeech/dev-clean/...
data/nsynth-valid/audio/...
```

## Reproduce

Run all main-paper audio experiments:

```bash
GPU=0 PY=python bash scripts/reproduce_main_audio.sh
python scripts/audit_main_audio.py
```

Expected values:

```text
results/expected/main_paper_audio_expected.csv
```

The audit passes when it prints:

```text
MAIN_PAPER_AUDIO_STATUS=PASS
```
