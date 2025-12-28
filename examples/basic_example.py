"""
Basic example of using the NeutroHydro package.

This script demonstrates the complete workflow:
1. Generate synthetic groundwater data
2. Fit the PNPLS model
3. Compute NVIP variable importance
4. Calculate NSR/pi_G attribution
5. Analyze sample-level baseline fractions
"""

import numpy as np
import pandas as pd

# Import NeutroHydro components
from neutrohydro import (
    NeutroHydroPipeline,
    Preprocessor,
    NDGEncoder,
    PNPLS,
    compute_nvip,
    compute_nsr,
    compute_sample_baseline_fraction,
)
from neutrohydro.pipeline import PipelineConfig


def generate_synthetic_data(n_samples=100, seed=42):
    """
    Generate synthetic groundwater-like ion data.

    Returns:
        X: Ion concentrations (mg/L)
        y: Log TDS (target)
        ion_names: Names of ions
    """
    rng = np.random.default_rng(seed)

    # Ion names
    ion_names = ["Ca2+", "Mg2+", "Na+", "K+", "HCO3-", "Cl-", "SO42-"]
    n_ions = len(ion_names)

    # Generate correlated ion data
    # Base concentrations (log-normal to simulate real groundwater)
    base_means = np.array([50, 20, 30, 5, 200, 40, 30])  # Typical mg/L
    base_stds = np.array([30, 15, 25, 3, 100, 35, 25])

    # Generate with some correlation structure
    X = np.zeros((n_samples, n_ions))
    for j in range(n_ions):
        X[:, j] = np.abs(rng.normal(base_means[j], base_stds[j], n_samples))

    # Add correlation: Na-Cl, Ca-HCO3, Ca-SO4
    X[:, 2] = 0.5 * X[:, 2] + 0.5 * X[:, 5]  # Na correlates with Cl
    X[:, 0] = 0.4 * X[:, 0] + 0.3 * X[:, 4] + 0.3 * X[:, 6]  # Ca correlates with HCO3, SO4

    # Target: log TDS (approximately sum of ions with some noise)
    tds = X.sum(axis=1) + rng.normal(0, 20, n_samples)
    y = np.log(np.maximum(tds, 1))  # Log transform

    return X, y, ion_names


def main():
    """Run the complete NeutroHydro workflow."""

    print("=" * 70)
    print("NEUTROHYDRO: Neutrosophic Chemometrics for Groundwater Analysis")
    print("=" * 70)

    # Generate synthetic data
    print("\n1. Generating synthetic groundwater data...")
    X, y, ion_names = generate_synthetic_data(n_samples=100)
    print(f"   Generated {X.shape[0]} samples with {X.shape[1]} ions")
    print(f"   Ions: {ion_names}")

    # Option A: Use the unified pipeline (recommended)
    print("\n2. Running NeutroHydro Pipeline...")
    print("-" * 50)

    config = PipelineConfig(
        n_components=5,
        log_transform=False,  # Data is already in reasonable scale
        baseline_type="median",
        rho_I=1.0,
        rho_F=1.0,
        lambda_F=1.0,
        gamma=0.7,
    )

    pipeline = NeutroHydroPipeline(config)
    results = pipeline.fit(X, y, feature_names=ion_names)

    # Print model performance
    print(f"\n   Model R² (train): {results.r2_train:.4f}")
    print(f"   Components used: {results.model.components_.n_components}")

    # Print NVIP results
    print("\n3. Variable Importance (NVIP):")
    print("-" * 50)
    print(f"   {'Ion':<10} {'VIP_T':>8} {'VIP_I':>8} {'VIP_F':>8} {'VIP_agg':>10}")
    print("   " + "-" * 46)

    for j, ion in enumerate(ion_names):
        vip_t = results.nvip.VIP_T[j]
        vip_i = results.nvip.VIP_I[j]
        vip_f = results.nvip.VIP_F[j]
        vip_agg = results.nvip.VIP_agg[j]
        marker = "*" if vip_agg >= 1.0 else " "
        print(f"   {ion:<10} {vip_t:>8.3f} {vip_i:>8.3f} {vip_f:>8.3f} {vip_agg:>9.3f}{marker}")

    print("\n   (* = VIP >= 1, important variable)")

    # Verify L2 decomposition
    from neutrohydro.nvip import verify_l2_decomposition
    l2_valid = verify_l2_decomposition(results.nvip)
    print(f"\n   L2 decomposition theorem verified: {l2_valid}")

    # Print NSR/pi_G attribution
    print("\n4. Baseline vs Perturbation Attribution (NSR/pi_G):")
    print("-" * 50)
    print(f"   {'Ion':<10} {'NSR':>10} {'pi_G':>8} {'Classification':<15}")
    print("   " + "-" * 43)

    for j, ion in enumerate(ion_names):
        nsr = results.nsr.NSR[j]
        pi_g = results.nsr.pi_G[j]
        cls = results.nsr.classification[j]
        print(f"   {ion:<10} {nsr:>10.3f} {pi_g:>8.3f} {cls:<15}")

    # Interpretation
    n_baseline = sum(results.nsr.classification == "baseline")
    n_perturbation = sum(results.nsr.classification == "perturbation")
    n_mixed = sum(results.nsr.classification == "mixed")
    print(f"\n   Classification summary (gamma={results.nsr.gamma}):")
    print(f"     Baseline-dominant: {n_baseline}")
    print(f"     Perturbation-dominant: {n_perturbation}")
    print(f"     Mixed: {n_mixed}")

    # Print sample-level G_i
    print("\n5. Sample-level Baseline Fraction (G_i):")
    print("-" * 50)
    G = results.sample_attribution.G
    print(f"   Mean G:   {np.mean(G):.3f}")
    print(f"   Std G:    {np.std(G):.3f}")
    print(f"   Min G:    {np.min(G):.3f}")
    print(f"   Max G:    {np.max(G):.3f}")
    print(f"   Median G: {np.median(G):.3f}")

    # Fraction of samples that are baseline-dominated
    frac_baseline = np.mean(G >= config.gamma)
    print(f"\n   Samples with G >= {config.gamma}: {frac_baseline*100:.1f}%")

    # Export results
    print("\n6. Exporting Results:")
    print("-" * 50)

    dfs = pipeline.to_dataframes()
    print("   Available DataFrames:")
    for name, df in dfs.items():
        print(f"     {name}: {len(df)} rows, {len(df.columns)} columns")

    # Show first few rows of NVIP results
    print("\n   NVIP DataFrame (first 5 rows):")
    print(dfs['nvip'].to_string(index=False))

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)

    return results


def step_by_step_example():
    """
    Alternative: Step-by-step usage of individual components.

    This shows how to use each module separately for more control.
    """
    print("\n" + "=" * 70)
    print("STEP-BY-STEP EXAMPLE (Individual Components)")
    print("=" * 70)

    # Generate data
    X, y, ion_names = generate_synthetic_data(n_samples=100)

    # Step 1: Preprocessing
    print("\nStep 1: Preprocessing...")
    preprocessor = Preprocessor(log_transform=False)
    X_std, y_std = preprocessor.fit_transform(X, y, feature_names=ion_names)
    print(f"   Standardized X shape: {X_std.shape}")

    # Step 2: NDG Encoding
    print("\nStep 2: NDG Encoding (T, I, F triplets)...")
    encoder = NDGEncoder(baseline_type="median", falsity_map="exponential")
    triplets = encoder.fit_transform(X_std)
    print(f"   Truth (T) shape: {triplets.T.shape}")
    print(f"   Indeterminacy (I) range: [{triplets.I.min():.3f}, {triplets.I.max():.3f}]")
    print(f"   Falsity (F) range: [{triplets.F.min():.3f}, {triplets.F.max():.3f}]")

    # Step 3: PNPLS Model
    print("\nStep 3: PNPLS Regression...")
    model = PNPLS(n_components=5, rho_I=1.0, rho_F=1.0, lambda_F=1.0)
    model.fit(triplets, y_std)
    r2 = model.score(triplets, y_std)
    print(f"   Model fitted with {model.components_.n_components} components")
    print(f"   R² = {r2:.4f}")

    # Step 4: NVIP
    print("\nStep 4: Computing NVIP...")
    nvip = compute_nvip(model)
    print(f"   VIP_agg range: [{nvip.VIP_agg.min():.3f}, {nvip.VIP_agg.max():.3f}]")

    # Step 5: NSR / pi_G
    print("\nStep 5: Computing NSR and pi_G...")
    nsr = compute_nsr(nvip, gamma=0.7)
    print(f"   pi_G range: [{nsr.pi_G.min():.3f}, {nsr.pi_G.max():.3f}]")

    # Step 6: Sample attribution
    print("\nStep 6: Computing sample-level G_i...")
    sample_attr = compute_sample_baseline_fraction(model, triplets, nsr)
    print(f"   G range: [{sample_attr.G.min():.3f}, {sample_attr.G.max():.3f}]")

    print("\nStep-by-step workflow complete!")

    return model, nvip, nsr, sample_attr


if __name__ == "__main__":
    # Run main example
    results = main()

    # Optionally run step-by-step example
    # step_by_step_example()
