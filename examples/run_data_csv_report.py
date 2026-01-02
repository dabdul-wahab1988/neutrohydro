"""Reproducible analysis + plotting for the bundled `data.csv`.

This script is designed to be consistent with the public NeutroHydro API.
It:
- loads `data.csv`
- runs the NeutroHydro pipeline using TDS as the target
- computes ionic balance diagnostics
- generates standard visualization figures (Gibbs, ILR, correlation, VIP)
- writes a LaTeX snippet with computed summary stats for inclusion in docs

Run (from repo root):
  python examples/run_data_csv_report.py

Notes:
- Uses a headless Matplotlib backend so it works on CI/servers.
- Enables mineral inversion (NNLS) without PHREEQC validation by default.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Headless plotting
os.environ.setdefault("MPLBACKEND", "Agg")

from neutrohydro import NeutroHydroPipeline
from neutrohydro.pipeline import PipelineConfig
from neutrohydro import visualization as vz


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data.csv"
OUTPUT_DIR = REPO_ROOT / "figures" / "data_csv_report"
RESULTS_DIR = REPO_ROOT / "neutrohydro_results" / "data_csv_report"
TEX_SNIPPET_PATH = RESULTS_DIR / "summary.tex"
JSON_SUMMARY_PATH = RESULTS_DIR / "summary.json"


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Canonical columns for NeutroHydro hydrochem plots (mg/L in data.csv)
    ions_mg = ["Ca", "Mg", "Na", "K", "HCO3", "Cl", "SO4", "NO3", "F"]
    required_cols = ["TDS", *ions_mg]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"data.csv is missing required columns: {missing}. Found: {list(df.columns)}"
        )

    # Convert to meq/L for modeling + mineral inversion
    df_meq_base = vz.mg_to_meq(df)
    df_meq = vz.calculate_ion_sums(df_meq_base)

    ions_meq_cols = [
        "Ca_meq",
        "Mg_meq",
        "Na_meq",
        "K_meq",
        "HCO3_meq",
        "Cl_meq",
        "SO4_meq",
        "NO3_meq",
        "F_meq",
    ]
    missing_meq = [c for c in ions_meq_cols if c not in df_meq.columns]
    if missing_meq:
        raise ValueError(
            f"mg_to_meq() did not produce required meq columns: {missing_meq}. "
            f"Found: {list(df_meq.columns)}"
        )

    # Charged ion names required by the mineral library
    feature_names = ["Ca2+", "Mg2+", "Na+", "K+", "HCO3-", "Cl-", "SO42-", "NO3-", "F-"]

    # Pipeline run (model TDS from major ions in meq/L)
    X = df_meq[ions_meq_cols].to_numpy(dtype=float)
    y = df["TDS"]

    config = PipelineConfig(
        baseline_type="robust_pca",
        baseline_rank=2,
        n_components=5,
        # enable mineral inversion; keep PHREEQC thermo validation off for portability
        run_mineral_inference=True,
        run_thermodynamic_validation=False,
    )

    pipeline = NeutroHydroPipeline(config=config)
    results = pipeline.fit(
        X=X,
        y=y,
        feature_names=feature_names,
        c_meq=X,
    )

    # Ionic balance diagnostics
    mean_ibe = _safe_float(df_meq["Ion_Balance_Error"].mean())
    median_ibe = _safe_float(df_meq["Ion_Balance_Error"].median())
    frac_ok = _safe_float((df_meq["Ion_Balance_Error"].abs() < 10).mean())

    # Generate plots
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vz.plot_gibbs(df, df_meq, output_path=str(OUTPUT_DIR / "gibbs_plot.png"))
    vz.plot_ilr_classification(df, df_meq, output_path=str(OUTPUT_DIR / "ilr_classification.png"))
    vz.plot_correlation_matrix(df, output_path=str(OUTPUT_DIR / "correlation_matrix.png"))
    vz.plot_vip_decomposition(results.nvip, output_path=str(OUTPUT_DIR / "vip_decomposition.png"))

    if results.mineral_result is not None:
        vz.plot_mineral_fractions(
            results.mineral_result,
            output_path=str(OUTPUT_DIR / "mineral_fractions.png"),
        )

    # Registry-composed plots (for documentation examples)
    vz.create_figure(
        "[g1|g2]",
        data={"df": df, "df_meq": df_meq},
        output_path=str(OUTPUT_DIR / "registry_gibbs_dual.png"),
    )
    vz.create_figure(
        "[h1]",
        data={"df": df, "df_meq": df_meq},
        output_path=str(OUTPUT_DIR / "registry_correlation.png"),
    )
    vz.create_figure(
        "[p1][p2]",
        data={"nvip_result": results.nvip},
        output_path=str(OUTPUT_DIR / "registry_model_combined.png"),
    )

    # Summary JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": "data.csv",
        "n_samples": int(len(df)),
        "ions": ions_mg,
        "target": "TDS",
        "pipeline": {
            "baseline_type": config.baseline_type,
            "baseline_rank": config.baseline_rank,
            "n_components": config.n_components,
            "r2_train": _safe_float(getattr(results, "r2_train", float("nan"))),
        },
        "tds": {
            "min": _safe_float(df["TDS"].min()),
            "median": _safe_float(df["TDS"].median()),
            "max": _safe_float(df["TDS"].max()),
        },
        "ionic_balance": {
            "mean_error_percent": mean_ibe,
            "median_error_percent": median_ibe,
            "fraction_within_10_percent": frac_ok,
        },
        "figures": {
            "gibbs_plot": str(OUTPUT_DIR / "gibbs_plot.png"),
            "ilr_classification": str(OUTPUT_DIR / "ilr_classification.png"),
            "correlation_matrix": str(OUTPUT_DIR / "correlation_matrix.png"),
            "vip_decomposition": str(OUTPUT_DIR / "vip_decomposition.png"),
            "mineral_fractions": str(OUTPUT_DIR / "mineral_fractions.png"),
            "registry_gibbs_dual": str(OUTPUT_DIR / "registry_gibbs_dual.png"),
            "registry_correlation": str(OUTPUT_DIR / "registry_correlation.png"),
            "registry_model_combined": str(OUTPUT_DIR / "registry_model_combined.png"),
        },
    }

    JSON_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # LaTeX snippet (paths relative to docs/document.tex)
    # docs/ -> repo root is ..
    def rel_to_docs(path: Path) -> str:
        return path.as_posix().replace(REPO_ROOT.as_posix() + "/", "../")

    tex = "\n".join(
        [
            "% Auto-generated by examples/run_data_csv_report.py",
            "\\newcommand{\\DataCsvNSamples}{%d}" % int(len(df)),
            "\\newcommand{\\DataCsvRtwo}{%.4f}" % _safe_float(getattr(results, "r2_train", float("nan"))),
            "\\newcommand{\\DataCsvTdsMin}{%.2f}" % _safe_float(df["TDS"].min()),
            "\\newcommand{\\DataCsvTdsMedian}{%.2f}" % _safe_float(df["TDS"].median()),
            "\\newcommand{\\DataCsvTdsMax}{%.2f}" % _safe_float(df["TDS"].max()),
            "\\newcommand{\\DataCsvMeanIbe}{%.2f}" % mean_ibe,
            "\\newcommand{\\DataCsvMedianIbe}{%.2f}" % median_ibe,
            "\\newcommand{\\DataCsvFracIbeOk}{%.2f}" % frac_ok,
            "\\newcommand{\\DataCsvFigGibbs}{%s}" % rel_to_docs(OUTPUT_DIR / "gibbs_plot.png"),
            "\\newcommand{\\DataCsvFigIlr}{%s}" % rel_to_docs(OUTPUT_DIR / "ilr_classification.png"),
            "\\newcommand{\\DataCsvFigCorr}{%s}" % rel_to_docs(OUTPUT_DIR / "correlation_matrix.png"),
            "\\newcommand{\\DataCsvFigVip}{%s}" % rel_to_docs(OUTPUT_DIR / "vip_decomposition.png"),
            "\\newcommand{\\DataCsvFigMinerals}{%s}" % rel_to_docs(OUTPUT_DIR / "mineral_fractions.png"),
            "\\newcommand{\\DataCsvFigRegistryGibbs}{%s}" % rel_to_docs(OUTPUT_DIR / "registry_gibbs_dual.png"),
            "\\newcommand{\\DataCsvFigRegistryCorr}{%s}" % rel_to_docs(OUTPUT_DIR / "registry_correlation.png"),
            "\\newcommand{\\DataCsvFigRegistryModel}{%s}" % rel_to_docs(OUTPUT_DIR / "registry_model_combined.png"),
            "",
        ]
    )

    TEX_SNIPPET_PATH.write_text(tex, encoding="utf-8")

    print("[OK] Wrote figures to:", OUTPUT_DIR)
    print("[OK] Wrote summary to:", JSON_SUMMARY_PATH)
    print("[OK] Wrote LaTeX snippet to:", TEX_SNIPPET_PATH)


if __name__ == "__main__":
    main()
