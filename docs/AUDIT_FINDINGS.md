# Integration audit

## Preserved scientific content

The two supplied ZIPs are authoritative. Every source file is mapped to its release path with original and release SHA-256 values. All image source files are unchanged. Audio model, preprocessing, training, optimizer and metric functions are unchanged. Only three audio `main` entry points gain an environment-selected output directory, with one required `os` import. Exact patches and original scripts are included in `provenance/`.

The 18 frozen audio command lines were parsed from the source shell launcher. Tests compare every argument to that launcher. Image job enumeration and selection use the source modules directly. The original namespaces and subprocess boundaries prevent collisions between baseline implementations.

## Operational issues addressed

The source audio launcher continued after subprocess failures and did not summarize a failure status. The unified runner propagates failures, validates outputs and records job receipts. The source core audio script reused any existing summary, while the other audio runners overwrote results. The unified interface prevents launch into unverified populated directories and reuses only verified complete groups.

The source audio audit counted finite values without enforcing exact manifest identities and accepted aggregate CSV fallback. The unified audit checks the expected clip identities and individual summaries. The source image reference check could accept NaN because a comparison with NaN is false. The unified checks reject nonfinite metrics and compare available per-image references. Historical audio PASS reports are stored as historical evidence.

Source hashes, absolute resolved data paths, local data hashes, package versions, runtime information, command arguments and output hashes accompany new reference runs. No dataset, hyperparameter, metric, source reference value or floating-point model operation was changed to obtain agreement.

## Inherited limitations relevant to interpretation

1. Core/SL2A audio normalize targets, while fJNB retains decoded amplitude scale. Resamplers and cast order also differ. The release documents these differences. They affect the interpretation of a fully controlled cross-method comparison.
2. The fJNB and SL2A audio loops execute 5,001 updates under `max_steps=5000`. The original core executes 5,000. Evaluation grids in time and PSNR stabilization differ.
3. The source audio environment is unpinned. In particular, `librosa.resample` uses a version-dependent default. The candidate pins librosa 0.11.0 and soxr 0.5.0.post1, but only the historical environment can establish what produced the original fJNB targets. Official librosa 0.11 documentation identifies `soxr_hq` as its default: https://librosa.org/doc/0.11.0/generated/librosa.resample.html.
4. Audio source hashes and per-clip reference results were not supplied. The original expected CSV contains dataset means. A mean-level pass cannot establish per-clip numerical agreement. The bundled historical summaries supply group aggregates, not raw per-clip evidence.
5. The source core `--force` flag does not force recomputation, and `--freeze_act_coeffs` is parsed without invoking the freezing helper. Neither behavior is exposed as a new scientific option by the unified runner. Frozen-coefficient experiments remain outside this release.
6. Core audio checkpoint tensors use `detach().cpu()` without cloning. On CPU this can retain aliases to live parameters, so a saved checkpoint need not represent the best evaluated step. The official CUDA path copies tensors to CPU. The integration preserves this code, tests scalar results, and does not advertise CPU checkpoints for scientific use.
7. Image fitting saves summaries and trajectories, with optional quantized previews. It does not save full model checkpoints. fJNB/SL2A audio save metrics without model checkpoints. Exact re-rendering of selected outputs requires an explicitly scoped extension.
8. The image source includes upstream commit IDs and two license texts. The audio source lacks corresponding provenance/license metadata for its bundled baselines. A top-level project license and citation record were also absent. They must be supplied by the authors before public release. No license or bibliographic metadata was invented.

## Remaining acceptance work

Run the unified CUDA environment on the actual datasets, compare all individual image targets and audio group means, and return the compact evidence archive. Recover the historical audio environment and data hashes, then compare preprocessing outputs and per-clip PSNR. Preserve the supplied targets when investigating deviations. Any proposed scientific correction belongs in an explicit new protocol version.
