"""Fit fJNB with raw coordinates or Fourier features."""
import argparse
from scripts.experiment import add_arguments, prepare_data, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", choices=["raw", "ff16", "both"], default="both",
                        help="Raw coordinates or 16 Fourier bands (default: both).")
    add_arguments(parser)
    args = parser.parse_args()
    manifests = prepare_data(args)
    inputs = ["raw", "ff16"] if args.input == "both" else [args.input]
    for encoding in inputs:
        for dataset, manifest in manifests.items():
            degree = 2 if dataset == "librispeech" and encoding == "raw" else 6
            suffix = f"noff_q{degree}" if encoding == "raw" else "ff16_s100_q6"
            parameters = dict(dataset=dataset, manifest=manifest,
                              group=f"{dataset}_fkan_{suffix}_lr1em3_bs4096_48k_v1",
                              sample_rate=48000, n_samples=48000, width=256, depth=4,
                              degree=degree, num_bands=0 if encoding == "raw" else 16,
                              sigma=1 if encoding == "raw" else 100,
                              lr=0.001, batch_size=4096, max_steps=5000, log_every=100, seed=0)
            if encoding == "ff16":
                parameters["ff_seed"] = 0
            run("train_fjnb.py", parameters, args)


if __name__ == "__main__":
    main()
