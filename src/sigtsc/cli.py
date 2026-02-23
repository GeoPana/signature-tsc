import argparse
from sigtsc.utils.io import load_yaml
from sigtsc.experiments.run_experiment import run_from_config
from sigtsc.experiments.run_suite import run_suite_from_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="sigtsc")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file (single experiment or suite).",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if "suite" in cfg:
        run_suite_from_config(args.config)
    else:
        run_from_config(args.config)