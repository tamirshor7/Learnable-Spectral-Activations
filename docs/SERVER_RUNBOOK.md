# Server acceptance runbook

## Preparation and provenance

Choose a server with an available RTX A6000 if possible, matching the documented image reference. Keep both historical environments and datasets. The image source records Python 3.10.19, torch 2.10.0+cu130, cuDNN 9.15.1, NumPy 2.2.6, Pillow 12.0.0, scikit-image 0.25.2 and PyYAML 6.0.3. The audio environment remains to be recovered.

Run the following with the historical audio interpreter, replacing the absolute paths. This only reads that environment and its data/results.

```bash
REPO=/absolute/path/to/lsa-unified
OLD_PY=/absolute/path/to/historical/audio/environment/bin/python
AUDIO_DATA=/absolute/path/to/historical/audio/repository/data
HISTORICAL_RUNS=/absolute/path/to/historical/audio/outputs/runs
EVIDENCE=/absolute/path/to/lsa-historical-evidence
mkdir -p "$EVIDENCE"
"$OLD_PY" "$REPO/tools/capture_environment.py" --out "$EVIDENCE/audio-environment.json"
"$OLD_PY" "$REPO/tools/fingerprint_audio.py" --data-root "$AUDIO_DATA" --out "$EVIDENCE/audio-preprocessing.json"
"$OLD_PY" "$REPO/tools/export_audio_evidence.py" --runs-root "$HISTORICAL_RUNS" --out "$EVIDENCE/audio-per-clip-results.json"
python "$REPO/tools/collect_evidence.py" --output-root "$EVIDENCE" --out "$EVIDENCE/historical-evidence.zip"
```

An incomplete export remains useful and reports which groups lack exact identities. No missing historical values are fabricated. The fingerprint tool uses unchanged source loaders under the selected interpreter, so differences between historical and unified preprocessing can be localized before retraining. Original audio hashes were absent from the upload, so retain the historical fingerprint export as evidence.

## Environment and staged execution

Install the root `environment.yml` from the repository directory. The CUDA dependency graph was resolved in this integration, and the CPU environment was installed and executed. The Conda solver and CUDA installation must still succeed on the server. Run `doctor`, unit tests, the CPU smoke suite, and `check-data` before training. For a CPU-only smoke test on a GPU machine, invoke `CUDA_VISIBLE_DEVICES= python tools/smoke.py` so the inherited image runner takes its CPU path.

Use an absolute output root and dedicated available GPUs. On Romeo, the example GPU order is 7,6,5. Single-GPU operation uses 7. A single group occupies one GPU and processes its clips sequentially. Core image fits provide an early check against a documented source environment. Run the core suite first, then audio, then STAF and SL2A as shown in the root README. After initial acceptance, `--suite all` is the complete combined run.

The prescribed scripts retain full-grid audio training and evaluation. If a configuration runs out of memory, record the failure and move to suitable hardware. Reducing batch size, splitting full-grid evaluation, changing precision or vectorizing the harmonic loop changes numerical execution and requires separate validation. Runtime and peak-memory estimates are not supplied because these were not measured here.

## Evidence required for acceptance

| Check | Acceptance evidence |
| --- | --- |
| Environment | actual package versions, CUDA/cuDNN, GPU and driver, successful installation |
| Audio data | manifest identity and source hashes, ideally matched to historical server |
| Preprocessing | historical and unified target fingerprints for all 50 clips |
| Core images | all 120 verified fits, selected candidate, per-image PSNR/SSIM agreement |
| Audio | all 18 groups and 450 fits, exact identities, finite results, means within inherited 0.01 dB |
| STAF | all 1,344 fits and per-image selected results |
| SL2A | all 60 calibration fits, selected rates, 48 final fits, per-image agreement |
| Process integrity | zero return codes, matching receipts, unchanged source and data |
| Public release metadata | author-supplied project license/citation and complete baseline notices |

Run `tools/collect_evidence.py` after each suite or after a failure. It includes bounded log tails and numerical summaries. Compare failures at the per-item level before changing any tolerance. For audio, the initial aggregate gate is constrained by the references supplied in the upload. The historical per-clip export allows a stronger second comparison.

Completion of this runbook, with measured numerical results, supports a claim that the supplied repository experiments reproduce under the unified environment. CPU parity alone supports the narrower integration claim documented in this package.
