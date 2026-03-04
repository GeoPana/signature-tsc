import argparse

from sigtsc.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(prog="sigtsc")
    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Run a single experiment config or a suite config.")
    p_run.add_argument("--config", default="configs/default.yaml")

    p_agg = sub.add_parser("aggregate", help="Aggregate metrics.json into results/*.csv")
    p_agg.add_argument("--results-root", default="results")
    p_agg.add_argument("--out-summary", default="results/summary.csv")
    p_agg.add_argument("--out-report", default="results/report.csv")
    p_agg.add_argument("--out-robustness", default="results/robustness.csv")
    p_agg.add_argument("--out-winners", default="results/robustness_winners.csv")

    p_plot = sub.add_parser("plot", help="Generate plots from aggregated CSVs.")
    p_plot.add_argument("--summary-csv", default="results/summary.csv")
    p_plot.add_argument("--report-csv", default="results/report.csv")
    p_plot.add_argument("--robustness-csv", default="results/robustness.csv")
    p_plot.add_argument("--out-dir", default="results/plots")
    p_plot.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Optional dataset filter (repeat flag), e.g. --dataset NATOPS --dataset CharacterTrajectories",
    )

    args = parser.parse_args()

    # Backward-compatible default: if no subcommand, run default config.
    if args.cmd is None:
        from sigtsc.experiments.run_experiment import run_from_config
        from sigtsc.experiments.run_suite import run_suite_from_config

        cfg = load_yaml("configs/default.yaml")
        if "suite" in cfg:
            run_suite_from_config("configs/default.yaml")
        else:
            run_from_config("configs/default.yaml")
        return

    if args.cmd == "run":
        from sigtsc.experiments.run_experiment import run_from_config
        from sigtsc.experiments.run_suite import run_suite_from_config

        cfg = load_yaml(args.config)
        if "suite" in cfg:
            run_suite_from_config(args.config)
        else:
            run_from_config(args.config)

    elif args.cmd == "aggregate":
        from sigtsc.experiments.aggregate_results import aggregate_results

        outs = aggregate_results(
            results_root=args.results_root,
            out_summary_csv=args.out_summary,
            out_report_csv=args.out_report,
            out_robustness_csv=args.out_robustness,
            out_robustness_winners_csv=args.out_winners,
        )
        print("[sigtsc] wrote:")
        for p in outs:
            print(f"[sigtsc] - {p}")

    elif args.cmd == "plot":
        from sigtsc.experiments.plot_results import generate_plots, print_plot_summary

        paths = generate_plots(
            summary_csv=args.summary_csv,
            report_csv=args.report_csv,
            robustness_csv=args.robustness_csv,
            out_dir=args.out_dir,
            datasets=args.dataset,
        )
        print_plot_summary(paths)
