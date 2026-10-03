# Environment decisions

One runtime is technically feasible: the source trees use ordinary PyTorch, NumPy and Python scientific packages, with no conflicting declared dependency pins. The image repository supplies exact versions. The audio repository supplies unversioned torch, NumPy, pandas, scipy, soundfile and librosa requirements.

The unified candidate therefore retains the image pins and supplies explicit audio pins. The actual CPU integration environment was created using Python 3.10.19 and PyTorch 2.10.0+cpu. Its installed freeze is retained in `validation/cpu-environment-freeze.txt`. The CUDA dependency graph is separately resolved and pinned in `environments/requirements-cu130.lock.txt`. CUDA runtime execution, Conda environment creation, and agreement with historical audio metrics remain server validation tasks.

The environment files specify one environment for both domains. Separate CPU and CUDA files select the intended device build for that same codebase. This avoids mixing runtime libraries from the two original servers. The transitive CUDA lock was resolved for Linux x86_64 and is a version pin set, not a platform-independent or hash-verified package archive.

PyTorch's official previous-version instructions list CUDA 13.0 wheels for 2.10.0: https://pytorch.org/get-started/previous-versions/. The explicit build tag `2.10.0+cu130` and an additional official wheel index prevent silently selecting a different torch CUDA build. General dependencies remain available through PyPI. A compatible NVIDIA driver is required, with the original image reference driver recorded as 580.173.02.

The most consequential unknown is the historical audio preprocessing stack. librosa 0.11.0 defaults to `soxr_hq`, while earlier versions may use a different resampler. SoundFile/libsndfile, SciPy, NumPy and GPU kernels can also change numerical results. `tools/fingerprint_audio.py` records decoded source hashes and the three actual target arrays under each environment. Use those fingerprints to decide whether a mismatch begins before optimization.

No common seed helper, altered deterministic setting, mixed precision, TF32 toggle, new optimizer implementation or shared rewritten activation has been introduced. The source code chooses its original defaults. The environment and runtime flags are recorded so those defaults remain inspectable.
