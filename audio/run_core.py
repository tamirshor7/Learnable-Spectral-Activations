"""Fit LSA, FINER or SIREN using the released audio settings."""
import argparse
from scripts.experiment import add_arguments, prepare_data, run

SETTINGS = {
    "lsa": dict(suffix="act16_ff16_sigma100_K16_lr2em3_am3p0_48k_full_v1",
                num_bands=16, sigma=100, K=16, lr=0.002, act_lr_mult=3.0),
    "finer": dict(suffix="ff16_finer_48k_sigma20_v1",
                  num_bands=16, sigma=20, fbs=20, lr=0.0005),
    "siren": dict(suffix="siren_noff_48k_audit_v1",
                  num_bands=0, sigma=1, w0=30, lr=0.0005),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=list(SETTINGS), required=True)
    add_arguments(parser)
    args = parser.parse_args()
    for dataset, manifest in prepare_data(args).items():
        settings = SETTINGS[args.method].copy()
        group = dataset + "_" + settings.pop("suffix")
        parameters = dict(dataset=dataset, manifest=manifest, group=group, method=args.method,
                          sample_rate=48000, n_samples=48000, width=256, depth=4,
                          lr_decay=0.2, decay_every=5000, max_steps=5000,
                          batch_size=0, log_every=250, seed=0, force=True, **settings)
        run("train_core.py", parameters, args)


if __name__ == "__main__":
    main()
