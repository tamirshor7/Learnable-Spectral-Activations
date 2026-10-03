# Paper revision notes, separate from execution

The repositories are the source of truth. This checklist identifies changes to the attached draft and imposes no paper-derived target on the code. Values below are inherited reference values, pending fresh reproduction in the unified environment.

| Item | Repository-authoritative value or definition |
| --- | --- |
| Kodak global LSA, all 24 | 38.1102234273 dB |
| Kodak per-image-tuned LSA | 38.3084867767 dB |
| Kodak global LSA, 18-image holdout | 38.0450462474 dB |
| Kodak LSA trainable parameter count | 199,043, including 128 coefficients |
| LSA selected image optimizer | lr 0.0005, activation multiplier 2 |
| Kodak LSA minus per-image STAF | 2.8834331520 dB |
| Kodak LSA minus full SL2A, holdout | -0.2114951292 dB, 8/18 wins |
| Kodak LSA minus matched SL2A, holdout | 2.1352281635 dB, 18/18 wins |
| Image training target | RGB transformed to [-1,1], evaluated after mapping to [0,1] and clipping |
| Image PSNR stabilization | max(MSE, 1e-12) |
| Core audio targets | zero mean, unit peak with denominator epsilon 1e-8 |
| SL2A audio targets | zero mean, exact unit peak for nonzero signals |
| fJNB audio targets | decoded amplitude scale |
| Audio update counts | core 5,000, fJNB/SL2A 5,001 |
| Audio best checkpoint terminology | best evaluated fitting-grid PSNR, with method-specific intervals and stabilization |

Audio LSA source means are 86.718373 dB on NSynth and 90.779016 dB on LibriSpeech. These are the supplied release targets, distinct from later experiments in conversation history. All 18 group targets remain unchanged in the source CSV.

Associated standard deviations, paired intervals, wins and tables should be recomputed from the authoritative per-item outputs wherever available. This integration does not edit the manuscript or infer replacement statistics from rounded means. Results outside the two supplied repositories require their own authoritative artifacts.
