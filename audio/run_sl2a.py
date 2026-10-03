"""Fit the released SL2A degree and rank comparisons."""
import argparse
from scripts.experiment import add_arguments, prepare_data, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", choices=["256", "512", "all"], default="all")
    parser.add_argument("--rank", choices=["64", "128", "all"], default="all")
    add_arguments(parser)
    args = parser.parse_args()
    degrees = [512, 256] if args.degree == "all" else [int(args.degree)]
    ranks = [128, 64] if args.rank == "all" else [int(args.rank)]
    for dataset, manifest in prepare_data(args).items():
        for degree in degrees:
            for rank in ranks:
                parameters = dict(dataset=dataset, manifest=manifest,
                                  group=f"{dataset}_sl2a_deg{degree}_rank{rank}_lr1em3_sched_48k_v1",
                                  sample_rate=48000, n_samples=48000, hidden_features=256,
                                  hidden_layers=3, deg=degree, rank=rank, lr=0.001,
                                  scheduler=True, scheduler_b=0.1, batch_size=0,
                                  max_steps=5000, log_every=100, seed=0)
                run("train_sl2a.py", parameters, args)


if __name__ == "__main__":
    main()
