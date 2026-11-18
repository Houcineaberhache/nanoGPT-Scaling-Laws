
"""Launch the six nanoGPT scaling-law experiments."""

import argparse
import yaml

from src.training.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=["E1", "E2", "E3", "E4", "E5", "E6"],
        default=None,
        help="Run only one experiment.",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=None,
        help="Override iteration count for testing.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size for testing.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device: auto, cpu, or cuda.",
    )
    args = parser.parse_args()

    with open("configs/experiments.yaml", "r") as f:
        settings = yaml.safe_load(f)

    training = settings["training"]
    experiments = settings["experiments"]

    selected = (
        {args.only: experiments[args.only]}
        if args.only
        else experiments
    )

    for name, experiment in selected.items():
        print("\n" + "=" * 60)
        print(f"STARTING {name}: {experiment['description']}")
        print("=" * 60)

        train(
            config_path=experiment["config"],
            train_path=experiment["train"],
            val_path="data/processed/val.bin",
            output_dir=experiment["output"],
            max_iters=args.max_iters or training["max_iters"],
            batch_size=args.batch_size or training["batch_size"],
            learning_rate=training["learning_rate"],
            eval_interval=training["eval_interval"],
            eval_iters=training["eval_iters"],
            log_interval=training["log_interval"],
            save_interval=training["save_interval"],
            seed=training["seed"],
            device=args.device or training["device"],
        )

        print(f"\nFINISHED {name}")


if __name__ == "__main__":
    main()
