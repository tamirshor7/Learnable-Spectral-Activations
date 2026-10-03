# Learnable Spectral Activations: unified audio and image reproduction

This release combines the two supplied authoritative repositories behind one environment, dataset layout, launcher, and audit interface. The original numerical implementations remain in `experiments/audio` and `experiments/images`. The repository references define the reproduction targets. Paper revisions are tracked separately in `docs/PAPER_REVISION_NOTES.md`.

**Status: locally validated integration candidate. Full GPU reference reproduction is pending.** See `docs/VALIDATION.md` for actual checks performed and `docs/AUDIT_FINDINGS.md` for inherited protocol differences and remaining release metadata.

## Scope

| Suite | Included fits | Count |
| --- | --- | ---: |
| Audio | LSA, FINER, SIREN, raw/FF16 fJNB, four SL2A settings, on fixed 10-clip NSynth and 40-clip LibriSpeech manifests | 450 |
| Image core | Three LSA candidates, FINER, SIREN, all 24 Kodak images | 120 |
| Image STAF | 56 candidates per Kodak image | 1,344 |
| Image SL2A | Two architectures, five learning rates on six calibration images, then 24 fits per selected architecture | 108 |
| Total | Separate signal/image fits | 2,022 |

The supplied repositories cover selected fitting results. They omit audio bandwidth sweeps, off-grid interpolation, Poisson reconstruction, NeRF, NAF, frozen-coefficient and harmonic-count ablations, and several appendix baselines. This package makes the same scope commitment.

## One environment

The primary target is Linux x86_64, Python 3.10.19, PyTorch 2.10.0+cu130, and an NVIDIA GPU/driver compatible with that build. The image source reference used RTX A6000 and driver 580.173.02. The audio source did not specify historical package versions. Audio package pins are a tested integration candidate pending numerical comparison with the historical audio environment.

```bash
REPO=/absolute/path/to/lsa-unified
cd "$REPO"
conda env create -f environment.yml
conda activate lsa-unified
python reproduce.py doctor > environment-observed.json
python reproduce.py verify-source
python -m unittest discover -s tests -v
python tools/smoke.py
```

`environment.yml` installs the complete version-pinned CUDA requirements in `environments/requirements-cu130.lock.txt`. Its dependency graph was resolved here. CUDA installation and execution must still be checked on the target server. The CPU counterpart is `environment-cpu.yml` and is intended for functional tests. Neither environment claims bitwise equivalence across GPU architectures.

Run installation from the repository root so relative requirement paths resolve. Do not substitute a different torch/CUDA stack and call it the validated reference environment. Record such a run as a separate environment experiment. Keep the historical environments until the migration passes.

## Data

All commands accept an absolute `--data-root`. Expected structure:

```text
data/
  kodak/kodim01.png ... kodim24.png
  librispeech_raw/LibriSpeech/dev-clean/<speaker>/<chapter>/*.flac
  nsynth-valid/audio/*.wav
```

Existing dataset directories can be symlinked into this layout. The 50 audio clip identities and manifest order are preserved. The loaders receive generated manifests with absolute paths, so all audio methods use the same resolved files.

```bash
python tools/download_data.py --suite images --data-root /absolute/path/to/data
python tools/download_data.py --suite audio --data-root /absolute/path/to/data
python reproduce.py check-data --data-root /absolute/path/to/data
```

The audio command downloads the full official validation archives and needs sufficient disk space. Datasets are not included or relicensed. Kodak files are checked against the source repository SHA-256 values. The audio preflight checks source format and identities, and records local hashes. Historical audio hashes were not supplied, so this inventory does not prove equality to the original server's audio bytes.

## Run in stages

`--gpus` contains physical GPU ordinals. Unset an inherited `CUDA_VISIBLE_DEVICES` before launch. The runner starts one job per listed GPU and preserves every training seed, candidate and checkpoint-selection rule.

```bash
DATA=/absolute/path/to/data
OUT=/absolute/path/to/outputs/reference-v1
python reproduce.py plan
python reproduce.py run --suite image-core --gpus 7 --data-root "$DATA" --output-root "$OUT"
python reproduce.py run --suite audio --gpus 7,6,5 --data-root "$DATA" --output-root "$OUT"
python reproduce.py run --suite image-staf --gpus 7,6,5 --data-root "$DATA" --output-root "$OUT"
python reproduce.py run --suite image-sl2a --gpus 7,6,5 --data-root "$DATA" --output-root "$OUT"
python reproduce.py audit --data-root "$DATA" --output-root "$OUT"
```

The example assumes an eight-GPU server and available GPUs. For one GPU use `--gpus 0`. To run everything sequentially by suite, use `--suite all`. `run` automatically audits the requested suite. STAF and SL2A require verified image-core outputs in the same output root. SL2A completes calibration before selecting learning rates and scheduling the final fits.

For a visible background launch:

```bash
PY="$(command -v python)" SUITE=all GPUS=7,6,5 DATA_ROOT="$DATA" OUTPUT_ROOT="$OUT" bash tools/launch_background.sh
```

The launcher prints PID, LOG, RC and DONE paths. Per-job command, PID, GPU, log, exit status and completion receipts are also saved. No session is closed. An interrupted or failed job stays available for diagnosis.

## Verification and resumption

Successful jobs are reused only when code, data, environment, job specification, output hashes and receipts agree. A partial or changed job directory causes an error. Move that job directory and receipt aside after reviewing the log, or use a fresh output root. Audio is resumed at group granularity, images at individual-fit granularity. Training checkpoints are not resumed mid-trajectory.

Audit checks include exact cohort membership, required job completion, finite metrics, and calibration selection. Audio uses the inherited 0.01 dB tolerance on each of 18 dataset means. Images use 0.02 dB on per-image PSNR and 0.002 on per-image SSIM, plus selected LSA candidate and SL2A learning rates. The SSIM and per-image gates are added diagnostics. Cross-hardware tolerance adequacy remains to be established. A failure is investigated, never repaired by changing reference numbers automatically.

Synthetic tests live under `outputs/smoke` and never enter reference audits. Historical source audit reports live under `provenance/historical_audio` and never count as newly reproduced results.

To return compact evidence:

```bash
python tools/collect_evidence.py --output-root "$OUT" --out /absolute/path/to/lsa-validation-evidence.zip
```

The archive contains reports, manifests, job metadata, summaries and bounded log tails. It excludes datasets and checkpoints. Check paths/environment metadata before sharing outside your research team.

## Layout

```text
reproduce.py          shared command-line interface
environment.yml       unified CUDA environment
configs/              authoritative audio command specifications
experiments/audio/    audio models, loaders, runners, manifests and references
experiments/images/   image models, loaders, configs and references
tools/                execution, data checks, tests and evidence collection
tests/                orchestration and integrity regression tests
provenance/           source hashes, exact I/O patches and historical evidence
validation/           evidence from tests actually run during integration
docs/                 protocol, environment, release audit and server runbook
```

Use the top-level interface for release validation. Nested READMEs and launchers are retained as source history and do not implement the unified safeguards. Original training/model code is intentionally separate by domain to preserve initialization order, floating-point operation order, and protocol differences. Third-party attribution and outstanding source metadata are recorded in `THIRD_PARTY.md`.
