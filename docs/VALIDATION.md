# Validation status

Integration date: 2026-10-03. The attached repositories supply the authoritative numerical targets. No reference value was changed.

## Executed here

| Check | Result | Evidence |
| --- | --- | --- |
| Clean Python 3.10.19 CPU dependency installation | PASS | validation/cpu-environment-freeze.txt |
| Full CUDA 13.0 dependency graph resolution | PASS | environments/requirements-cu130.lock.txt |
| Source-file SHA-256 integrity | PASS | provenance/source_files.json |
| Audio numerical function/class AST equality to original scripts | PASS | unit test log |
| Exact transcription of all 18 source audio commands | PASS | unit test log |
| All 18 audio command variants, original versus unified | PASS, exact equality on synthetic reduced runs | validation/smoke_report.json |
| Core audio checkpoint tensors and convergence CSV parity | PASS for all six core groups | validation/smoke_report.json |
| Six image architecture/configuration smoke runs | PASS | validation/smoke_report.json |
| Real image worker launched through unified scheduler | PASS, including receipt and verified reuse | validation/unit-tests.log |
| Failure, identity, finite-value, partial-output, provenance and archive checks | PASS, 15 unit/integration tests in total | validation/unit-tests.log |
| Official Kodak download and source SHA-256 verification | PASS, all 24 images | validation/kodak-download.log and kodak-data-inventory.json |
| Python compilation and background-shell syntax | PASS | validation/status.json |
| Bounded evidence ZIP generation | PASS | validation/status.json |

Audio parity uses one deterministic synthetic 16 kHz PCM WAV, resampling through the actual loaders, 96 target samples, full source model sizes, batch size 16, and `max_steps=2`. Core performs two updates, fJNB/SL2A three. All original and unified per-clip summary values agree exactly after excluding wall-clock duration. For the six core groups, checkpoint state tensors and convergence CSV bytes also agree. This covers path routing and unchanged numerical execution on the tested CPU stack.

Image smoke uses a synthetic 16x16 RGB image, full architecture dimensions, two updates, batch size 16, and rendering in chunks of 64. Tested architectures are LSA, FINER, SIREN, STAF, SL2A full and SL2A parameter matched. Expected parameter counts are enforced by the original trainer where declared. The scheduler integration test separately launches the actual worker on an 8x8 image through the new queue and checks receipt-based resumption.

Unit tests also check calibration-only SL2A selection, exact candidate counts, unchanged audio functions/classes, cohort integrity, duplicate IDs, nonfinite values, changed outputs, failed subprocess return codes, partial-output refusal, safe tar extraction and reuse of verified outputs. Synthetic audit fixtures test bookkeeping and never count as reference reproduction.

The actual CPU run had no usable CUDA runtime. Package versions, platform and runtime information are preserved. The version-pinned CUDA graph includes PyTorch 2.10.0+cu130 and cuDNN 9.15.1.9 dependencies, but those GPU libraries were not executed here.

## Pending

- Conda environment creation and CUDA installation on the target server.
- Full 2,022-fit reproduction with the actual datasets.
- Agreement of all image per-item results and all 18 audio means with the supplied targets.
- Historical audio environment, source waveform hashes, and per-clip reference recovery.
- Cross-environment preprocessing fingerprint comparison.
- GPU runtime and peak-memory measurement.
- Author-supplied citation/license metadata and remaining third-party notices.

Current status is **integration verified on CPU, full numerical reproduction pending**. The bundled historical audio PASS report is inherited evidence and is not counted as a new validation result.
