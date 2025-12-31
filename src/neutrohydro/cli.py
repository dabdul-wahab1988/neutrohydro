"""
Command-line interface for NeutroHydro.

Provides CLI entrypoints for running the pipeline
and inspecting results.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from neutrohydro.pipeline import NeutroHydroPipeline, PipelineConfig
from neutrohydro.quality_check import check_sanity


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        prog="neutrohydro",
        description="Neutralization-Displacement Geosystem (NDG) Framework",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run pipeline command
    run_parser = subparsers.add_parser(
        "run",
        help="Run the NeutroHydro pipeline on data",
    )
    run_parser.add_argument(
        "data_file",
        type=str,
        help="Path to CSV file with ion data",
    )
    run_parser.add_argument(
        "--target",
        "-t",
        type=str,
        required=True,
        help="Name of target column",
    )
    run_parser.add_argument(
        "--features",
        "-f",
        type=str,
        nargs="+",
        help="Names of feature columns (if not specified, uses all except target)",
    )
    run_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="neutrohydro_results",
        help="Output directory for results (default: neutrohydro_results)",
    )
    run_parser.add_argument(
        "--n-components",
        type=int,
        default=5,
        help="Number of PLS components (default: 5)",
    )
    run_parser.add_argument(
        "--log-transform",
        action="store_true",
        help="Apply log transform to predictors",
    )
    run_parser.add_argument(
        "--baseline",
        type=str,
        choices=["robust_pca", "low_rank", "hydrofacies"],
        default="robust_pca",
        help="Baseline type for NDG encoder (default: robust_pca)",
    )
    run_parser.add_argument(
        "--baseline-rank",
        type=int,
        default=2,
        help="Rank for low-rank baseline methods (default: 2)",
    )
    run_parser.add_argument(
        "--rho-i",
        type=float,
        default=1.0,
        help="Weight for Indeterminacy channel (default: 1.0)",
    )
    run_parser.add_argument(
        "--rho-f",
        type=float,
        default=1.0,
        help="Weight for Falsity channel (default: 1.0)",
    )
    run_parser.add_argument(
        "--lambda-f",
        type=float,
        default=1.0,
        help="Falsity weighting strength (default: 1.0)",
    )
    run_parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        help="Classification threshold for baseline/perturbation (default: 0.7)",
    )
    run_parser.add_argument(
        "--minerals",
        action="store_true",
        help="Run mineral stoichiometric inference",
    )
    run_parser.add_argument(
        "--groups",
        type=str,
        help="Column name for grouping samples (handles heterogeneity)",
    )
    run_parser.add_argument(
        "--validate-thermo",
        action="store_true",
        help="Enable thermodynamic validation (requires pH/Eh)",
    )
    run_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show information about the package",
    )

    return parser


def run_command(args) -> int:
    """Execute the run command."""
    # Load data
    data_path = Path(args.data_file)
    if not data_path.exists():
        print(f"Error: File not found: {data_path}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Loading data from {data_path}...")

    df = pd.read_csv(data_path)

    # Sanity Check
    if args.verbose:
        print("Running sanity checks...")
    sanity_report = check_sanity(df)
    
    if not sanity_report["valid"]:
        print("CRITICAL ERROR: Data failed sanity checks!", file=sys.stderr)
        for warning in sanity_report["warnings"]:
            print(f"  - {warning}", file=sys.stderr)
        print("Aborting to prevent meaningless results.", file=sys.stderr)
        return 1
        
    if sanity_report["warnings"]:
        print("WARNING: Potential data quality issues detected:", file=sys.stderr)
        for warning in sanity_report["warnings"]:
            print(f"  - {warning}", file=sys.stderr)
        print("Proceeding with caution...", file=sys.stderr)

    # Validate target column
    if args.target not in df.columns:
        print(f"Error: Target column '{args.target}' not found in data", file=sys.stderr)
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        return 1

    # Get feature columns
    if args.features:
        feature_names = args.features
        missing = [f for f in feature_names if f not in df.columns]
        if missing:
            print(f"Error: Feature columns not found: {missing}", file=sys.stderr)
            return 1
    else:
        # Use all numeric columns except target and special ones
        exclude = [args.target]
        if args.groups: exclude.append(args.groups)
        
        # Also exclude common ID columns
        id_cols = ["SampleID", "Code", "ID", "Name", "Date", "Time"]
        exclude.extend([c for c in df.columns if c in id_cols])
        
        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_names = [c for c in numeric_cols if c not in exclude]

    if args.verbose:
        print(f"Target: {args.target}")
        print(f"Features ({len(feature_names)}): {feature_names}")

    # Extract X and y
    X = df[feature_names].values
    y = df[args.target].values

    # Check for NaN
    if np.any(np.isnan(X)):
        print("Warning: NaN values found in features, filling with column medians")
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            if np.any(mask):
                X[mask, j] = np.nanmedian(X[:, j])

    if np.any(np.isnan(y)):
        print("Error: NaN values in target column", file=sys.stderr)
        return 1

    # Create config
    config = PipelineConfig(
        log_transform=args.log_transform,
        baseline_type=args.baseline,
        baseline_rank=args.baseline_rank,
        n_components=args.n_components,
        rho_I=args.rho_i,
        rho_F=args.rho_f,
        lambda_F=args.lambda_f,
        gamma=args.gamma,
        run_mineral_inference=args.minerals,
        run_thermodynamic_validation=args.validate_thermo,
    )

    if args.verbose:
        print("\nPipeline configuration:")
        print(f"  n_components: {config.n_components}")
        print(f"  baseline_type: {config.baseline_type}")
        print(f"  log_transform: {config.log_transform}")
        print(f"  rho_I: {config.rho_I}, rho_F: {config.rho_F}")
        print(f"  lambda_F: {config.lambda_F}")
        print(f"  gamma: {config.gamma}")

    # Extract groups if provided
    groups = None
    if args.groups:
        if args.groups not in df.columns:
            print(f"Error: Group column '{args.groups}' not found", file=sys.stderr)
            return 1
        # Convert to integer codes if string
        if df[args.groups].dtype == object:
            groups = pd.Categorical(df[args.groups]).codes
        else:
            groups = df[args.groups].values

    # Extract pH and Eh if available (for thermodynamic validation)
    pH = None
    Eh = None
    for col in df.columns:
        if col.lower() == "ph":
            pH = df[col].values
        elif col.lower() == "eh":
            Eh = df[col].values

    # Run pipeline
    if args.verbose:
        print("\nFitting pipeline...")

    pipeline = NeutroHydroPipeline(config)
    results = pipeline.fit(X, y, feature_names, groups=groups, pH=pH, Eh=Eh)

    if args.verbose:
        print(f"\nModel R2: {results.r2_train:.4f}")
        print(f"Components used: {results.model.components_.n_components}")

    # Save results
    output_dir = Path(args.output)
    if args.verbose:
        print(f"\nSaving results to {output_dir}/...")

    pipeline.save_results(output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("NEUTROHYDRO RESULTS SUMMARY")
    print("=" * 60)

    print(f"\nModel Performance:")
    print(f"  R2 (train): {results.r2_train:.4f}")
    print(f"  Components: {results.model.components_.n_components}")

    print(f"\nVariable Importance (VIP_agg):")
    for j, (name, vip) in enumerate(zip(feature_names, results.nvip.VIP_agg)):
        marker = "*" if vip >= 1.0 else " "
        print(f"  {marker} {name}: {vip:.3f}")

    print(f"\nBaseline vs Perturbation (pi_G):")
    for j, (name, pi_g, cls) in enumerate(zip(
        feature_names, results.nsr.pi_G, results.nsr.classification
    )):
        print(f"    {name}: {pi_g:.3f} ({cls})")

    print(f"\nSample-level Baseline Fraction (G):")
    G = results.sample_attribution.G
    print(f"  Mean: {np.mean(G):.3f}")
    print(f"  Std:  {np.std(G):.3f}")
    print(f"  Min:  {np.min(G):.3f}")
    print(f"  Max:  {np.max(G):.3f}")

    print(f"\nResults saved to: {output_dir.absolute()}")
    print("=" * 60)

    return 0


def info_command(args) -> int:
    """Execute the info command."""
    from neutrohydro import __version__

    print("NeutroHydro - Neutralization-Displacement Geosystem (NDG) Framework")
    print(f"Version: {__version__}")
    print()
    print("Components:")
    print("  - NDG Encoder: Neutrosophic Data Generator")
    print("  - PNPLS: Probabilistic Neutrosophic PLS regression")
    print("  - NVIP: Neutrosophic Variable Importance in Projection")
    print("  - NSR/pi_G: Baseline vs perturbation attribution")
    print("  - Mineral Inference: Stoichiometric inversion via NNLS")
    print()
    print("Usage:")
    print("  neutrohydro run data.csv --target TDS")
    print("  neutrohydro-pipeline (interactive mode)")
    print()
    print("For more information, see the documentation.")

    return 0


def main() -> int:
    """Main CLI entrypoint."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "run":
        return run_command(args)
    elif args.command == "info":
        return info_command(args)
    else:
        parser.print_help()
        return 1


def pipeline_main() -> int:
    """Interactive pipeline entrypoint."""
    print("=" * 60)
    print("NEUTROHYDRO INTERACTIVE PIPELINE")
    print("=" * 60)
    print()

    # Get data file
    data_file = input("Enter path to CSV data file: ").strip()
    if not data_file:
        print("Error: No file specified")
        return 1

    data_path = Path(data_file)
    if not data_path.exists():
        print(f"Error: File not found: {data_path}")
        return 1

    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")

    # Get target
    print()
    target = input("Enter target column name: ").strip()
    if target not in df.columns:
        print(f"Error: Column '{target}' not found")
        return 1

    # Get features
    print()
    features_input = input("Enter feature columns (comma-separated, or 'all' for all except target): ").strip()
    if features_input.lower() == "all":
        feature_names = [c for c in df.columns if c != target]
    else:
        feature_names = [f.strip() for f in features_input.split(",")]

    print(f"\nUsing {len(feature_names)} features: {feature_names}")

    # Configuration
    print("\n--- Configuration ---")

    n_components = input("Number of PLS components [5]: ").strip()
    n_components = int(n_components) if n_components else 5

    log_transform = input("Apply log transform to predictors? (y/n) [n]: ").strip().lower()
    log_transform = log_transform == "y"

    baseline = input("Baseline type (median/low_rank) [median]: ").strip()
    baseline = baseline if baseline else "median"

    # Extract data
    X = df[feature_names].values
    y = df[target].values

    # Handle NaN
    if np.any(np.isnan(X)):
        print("\nWarning: NaN values found, filling with column medians")
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            if np.any(mask):
                X[mask, j] = np.nanmedian(X[:, j])

    # Create and run pipeline
    print("\n--- Running Pipeline ---")

    config = PipelineConfig(
        n_components=n_components,
        log_transform=log_transform,
        baseline_type=baseline,
    )

    pipeline = NeutroHydroPipeline(config)
    results = pipeline.fit(X, y, feature_names)

    print(f"\nModel R2: {results.r2_train:.4f}")

    # Display results
    print("\n--- Variable Importance (VIP_agg) ---")
    for name, vip in zip(feature_names, results.nvip.VIP_agg):
        bar = "*" * int(vip * 10)
        print(f"  {name:15s}: {vip:.3f} {bar}")

    print("\n--- Attribution (pi_G) ---")
    for name, pi_g, cls in zip(feature_names, results.nsr.pi_G, results.nsr.classification):
        print(f"  {name:15s}: {pi_g:.3f} [{cls}]")

    print("\n--- Sample Baseline Fraction ---")
    G = results.sample_attribution.G
    print(f"  Mean G: {np.mean(G):.3f}")
    print(f"  Range: [{np.min(G):.3f}, {np.max(G):.3f}]")

    # Save
    print()
    save = input("Save results? (y/n) [y]: ").strip().lower()
    if save != "n":
        output_dir = input("Output directory [neutrohydro_results]: ").strip()
        output_dir = output_dir if output_dir else "neutrohydro_results"
        pipeline.save_results(output_dir)
        print(f"\nResults saved to {output_dir}/")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
