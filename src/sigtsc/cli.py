import argparse

from sigtsc.utils.io import load_yaml
from sigtsc.experiments.run_experiment import run_from_config
from sigtsc.experiments.run_suite import run_suite_from_config
from sigtsc.experiments.aggregate_results import aggregate_results


def main() -> None:
    parser = argparse.ArgumentParser(prog="sigtsc")
    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Run a single experiment config or a suite config.")
    p_run.add_argument("--config", default="configs/default.yaml")

    p_agg = sub.add_parser("aggregate", help="Aggregate metrics.json into results/summary.csv")
    p_agg.add_argument("--results-root", default="results")
    p_agg.add_argument("--out", default="results/summary.csv")

    args = parser.parse_args()

    # Backward-compatible default: if no subcommand, treat as run --config <value>
    if args.cmd is None:
        cfg = load_yaml("configs/default.yaml")
        if "suite" in cfg:
            run_suite_from_config("configs/default.yaml")
        else:
            run_from_config("configs/default.yaml")
        return

    if args.cmd == "run":
        cfg = load_yaml(args.config)
        if "suite" in cfg:
            run_suite_from_config(args.config)
        else:
            run_from_config(args.config)

    elif args.cmd == "aggregate":
        out = aggregate_results(args.results_root, args.out)
        print(f"[sigtsc] wrote {out}")