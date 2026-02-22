import argparse
from sigtsc.experiments.run_experiment import run_from_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="sigtsc")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    run_from_config(args.config)