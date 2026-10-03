# Learnable Spectral Activations

Code for the audio and 2D image-fitting experiments in *Learnable Spectral Activations*.

## Setup

From the repository root:

```bash
conda env create -f environment.yml
conda activate lsa
```

## Experiments

| Experiment | Instructions |
| --- | --- |
| Audio fitting on LibriSpeech and NSynth | [audio](audio/README.md) |
| 2D image fitting on Kodak | [image_2d](image_2d/README.md) |

Each directory contains its own data preparation, training scripts, and reference results. Both use the same environment.

## Acknowledgments

We use the authors' implementations of FINER, STAF, SL2A, and fKAN. See [THIRD_PARTY.md](THIRD_PARTY.md).
