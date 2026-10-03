# Authoritative protocols

## Image fitting

The 24 original Kodak files are loaded as RGB float32 in [0,1]. Training targets are mapped to [-1,1]. Coordinates are constructed on the device in [-1,1]^2 in horizontal-then-vertical order. Predictions are transformed to [0,1] and clipped for PSNR and SSIM. Evaluation uses full-resolution images. PSNR is `-10 log10(max(MSE, 1e-12))` on clipped float predictions, before PNG quantization. SSIM uses `skimage.metrics.structural_similarity(channel_axis=2, data_range=1)`.

Core models use width 256, four nonlinear layers, Adam, seed 0, 5,000 updates and 65,536 sampled pixels with replacement per update. The default exponential schedule reaches multiplier 0.1. Core/SL2A evaluate at update 1, every 100 updates and the last update. STAF evaluates every 100 updates and the last update. Best PSNR chooses a checkpoint, and reported SSIM comes from that same checkpoint.

LSA has K=32, coefficient standard deviation 0.001 and 199,043 trainable parameters, including 128 activation coefficients. Three candidates are evaluated: `(lr, activation multiplier) = (0.0005,2), (0.001,2), (0.001,1)`. Global selection maximizes mean PSNR on kodim02, 05, 07, 10, 13 and 23, with the source tie-breaking rule. Per-image selection maximizes each image's PSNR over the same three candidates. The 18-image holdout concerns hyperparameter selection across images, with all pixels fitted on each holdout image.

FINER and SIREN use learning rate 0.0005 and 198,915 parameters. FINER first bias scale is 0.70710678. STAF uses the source 48-configuration grid plus eight boundary configurations. SL2A uses `(degree,rank) = (256,128)` and `(192,64)`, with 330,243 and 199,171 parameters. Each architecture selects learning rate from 0.0005, 0.001, 0.002, 0.004 and 0.008 on the six calibration images. No holdout result participates in that selection. The source SL2A deterministic/cuDNN settings remain local to each worker.

## Audio fitting

Each group uses the fixed manifest order, resetting the random seed for every clip. Signals are resampled to 48 kHz before selecting the first 48,000 samples. Short signals are padded after resampling. These are fitting-grid evaluations, with best observed PSNR across the method-specific evaluation schedule.

| Property | Core LSA / FINER / SIREN | fJNB | SL2A |
| --- | --- | --- | --- |
| Source decode | soundfile default float64 | soundfile, cast float32 before mono mean | soundfile, mono mean then float32 |
| Resampler | scipy resample_poly, before mono averaging | librosa default resampler, after mono averaging | scipy resample_poly, after mono averaging |
| Normalization | subtract mean, divide by max absolute value + 1e-8 | decoded amplitude scale | subtract mean, divide by peak if nonzero |
| Coordinates | NumPy float32 linspace, then move to device | NumPy float32 linspace, then move to device | torch linspace directly on device |
| Updates | 5,000 | 5,001 | 5,001 |
| Evaluation after updates | 1, 251, ..., 4,751, 5,000 | 1, 101, ..., 5,001 | 1, 101, ..., 5,001 |
| Batch | full 48,000-sample grid | 4,096 sampled coordinates | full grid |
| PSNR stabilization | MSE + 1e-12 | max(MSE, 1e-20) | max(MSE, 1e-30) |
| Learning rate | LSA 0.002, FINER/SIREN 0.0005 | 0.001 | 0.001 |
| Schedule | multiply by 0.2 after update 5,000 | constant | LambdaLR to multiplier 0.1, clipped after step 5,000 |

LSA uses FF16, sigma 100, K=16, coefficient standard deviation 0.01 and activation learning-rate multiplier 3. FINER uses FF16, sigma 20, first-bias scale 20. SIREN uses raw coordinates and omega 30. Core Fourier features draw from the global torch RNG before constructing the network. fJNB Fourier features use a separate seeded CPU generator, so RNG consumption for subsequent network initialization differs. This behavior is preserved.

fJNB raw degree is 6 on NSynth and 2 on LibriSpeech. FF16 uses sigma 100 and degree 6 on both. SL2A runs all four combinations of degree 256/512 and rank 64/128, using the raw coordinate input and the supplied official architecture. Width and nonlinear depth are preserved from the source commands. The source loops report zero-based audio step indices.

These protocols differ in target normalization and optimization budget. The unified release reproduces those defined comparisons. A future common-normalization comparison would be a separately named experiment with its own targets. At high PSNR, the core epsilon changes the metric materially near its approximately 120 dB ceiling. It must remain part of the reported definition.
