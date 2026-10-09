"""Command-line interface for the reproducible project workflow."""

import argparse
from pathlib import Path

from .download import EXPERIMENTS, download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download", help="Download and checksum a measured experiment")
    fetch.add_argument("--experiment", choices=EXPERIMENTS, default="O4")
    fetch.add_argument("--data-dir", type=Path, default=Path("data"))
    for name in ("audit", "prepare"):
        command = commands.add_parser(name, help=f"{name.title()} measured HDF5 files")
        command.add_argument("--experiment", choices=EXPERIMENTS, default="O4")
        command.add_argument("--data-dir", type=Path, default=Path("data"))
        command.add_argument("--output", type=Path)
        if name == "prepare":
            command.add_argument("--wave", type=int, choices=(1, 2, 3), default=1)
            command.add_argument("--stride", type=int, default=8)
    measured = commands.add_parser("measured-baseline", help="Train and evaluate sparse WSE reconstruction")
    measured.add_argument("--dataset", type=Path, default=Path("data/processed/O4_wave1.npz"))
    measured.add_argument("--output", type=Path, default=Path("outputs/measured/O4_wave1"))
    measured.add_argument("--epochs", type=int, default=1500)
    measured.add_argument("--sensors", type=int, default=128)
    measured.add_argument("--seed", type=int, default=42)
    synthetic = commands.add_parser("synthetic-benchmark", help="Verify a shallow-water PINN on simulated data")
    synthetic.add_argument("--output", type=Path, default=Path("outputs/synthetic"))
    synthetic.add_argument("--epochs", type=int, default=1500)
    synthetic.add_argument("--seed", type=int, default=42)
    synthetic.add_argument("--physics-weight", type=float, default=0.01)
    transfer = commands.add_parser("measured-transfer", help="Evaluate a frozen model on a different measured wave")
    transfer.add_argument("--source", type=Path, default=Path("data/processed/O4_wave1.npz"))
    transfer.add_argument("--target", type=Path, default=Path("data/processed/O4_wave3.npz"))
    transfer.add_argument("--trained", type=Path, default=Path("outputs/measured/O4_wave1"))
    transfer.add_argument("--output", type=Path, default=Path("outputs/measured/O4_wave1_to_wave3"))
    report = commands.add_parser("report", help="Export completed runs as small portfolio artifacts")
    report.add_argument("--measured", type=Path, default=Path("outputs/measured/O4_wave1"))
    report.add_argument("--synthetic", type=Path, default=Path("outputs/synthetic"))
    report.add_argument("--transfer", type=Path, default=Path("outputs/measured/O4_wave1_to_wave3"))
    report.add_argument("--output", type=Path, default=Path("reports"))
    notebook = commands.add_parser("run-notebook", help="Execute the full showcase in the current Python environment")
    notebook.add_argument("--source", type=Path, default=Path("notebooks/flood_reconstruction_model_comparison.ipynb"))
    notebook.add_argument("--output", type=Path, default=Path("outputs/notebooks/verified.ipynb"))
    notebook.add_argument("--data-source", choices=("processed", "raw"), default="processed")
    ablation = commands.add_parser("sensor-ablation", help="Run paired sensor budgets using the notebook's model code")
    ablation.add_argument("--source", type=Path, default=Path("notebooks/flood_reconstruction_model_comparison.ipynb"))
    ablation.add_argument("--output", type=Path, default=Path("outputs/sensor_ablation"))
    ablation.add_argument("--reports", type=Path, default=Path("reports/sensor_ablation"))
    ablation.add_argument("--sensors", type=int, nargs="+", default=[16, 32, 64, 128])
    ablation.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ablation.add_argument("--steps", type=int, default=1500)
    args = parser.parse_args()
    if args.command == "download":
        print(download(args.experiment, args.data_dir))
    elif args.command == "audit":
        from .data import audit
        output = args.output or Path(f"reports/{args.experiment}_audit.json")
        report = audit(args.data_dir, args.experiment, output)
        print(report["waves"])
        print(output)
    elif args.command == "prepare":
        from .data import prepare
        output = args.output or args.data_dir / "processed" / f"{args.experiment}_wave{args.wave}.npz"
        print(prepare(args.data_dir, args.experiment, args.wave, args.stride, output))
        print(output)
    elif args.command == "measured-baseline":
        from .measured import run_measured
        report = run_measured(args.dataset, args.output, args.epochs, args.sensors, args.seed)
        print({key: report[key] for key in ("linear_interpolation", "supervised_mlp")})
        print(args.output)
    elif args.command == "synthetic-benchmark":
        from .synthetic import run_synthetic
        report = run_synthetic(args.output, args.epochs, args.seed, args.physics_weight)
        print(report["results"])
        print(args.output)
    elif args.command == "measured-transfer":
        from .measured import run_transfer
        report = run_transfer(args.source, args.target, args.trained, args.output)
        print({key: report[key] for key in ("frozen_source_interpolation", "frozen_source_mlp")})
        print(args.output)
    elif args.command == "report":
        from .report import summarize
        print(summarize(args.measured, args.synthetic, args.transfer, args.output))
    elif args.command == "run-notebook":
        from .notebook import run_notebook
        run_notebook(args.source, args.output, args.data_source)
    elif args.command == "sensor-ablation":
        from .ablation import run_ablation
        run_ablation(args.source, args.output, args.reports, args.sensors, args.seeds, args.steps)


if __name__ == "__main__":
    main()
