"""Generate data under one or both of the paper's flexible-arm controllers."""

import argparse
from pathlib import Path

import numpy as np

from flexible_arm import FlexibleArm, generate_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("stable", "unstable", "both"), default="both")
    parser.add_argument("--plant", choices=("nonlinear", "linear"), default="nonlinear")
    parser.add_argument("--trajectories", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200, help="number of 0.05 s feedback periods")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initial-scale", type=float, default=0.01,
                        help="standard deviation of initial joint angles and rates")
    parser.add_argument("--output", type=Path, default=Path("data"), help="output directory")
    args = parser.parse_args()

    arm = FlexibleArm()
    names = ("stable", "unstable") if args.controller == "both" else (args.controller,)
    numerical_failure = False
    for name in names:
        try:
            data = generate_data(name, plant=args.plant, trajectories=args.trajectories,
                                 steps=args.steps, seed=args.seed, initial_scale=args.initial_scale,
                                 arm=arm)
        except ValueError as error:
            parser.error(str(error))
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / f"{name}.npz"
        np.savez_compressed(path, **data, seed=args.seed, initial_scale=args.initial_scale)
        values, counts = np.unique(data["status"], return_counts=True)
        summary = ", ".join(f"{n} {value}" for value, n in zip(values, counts))
        print(f"{path}: {summary}")
        numerical_failure |= bool(np.any(data["status"] == "numerical_failure"))
    return int(numerical_failure)


if __name__ == "__main__":
    raise SystemExit(main())
