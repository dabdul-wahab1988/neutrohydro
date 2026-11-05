"""
Jacobian-orthogonal encoding: a worked example on data.csv.

Shows, on the bundled dataset,
1. how entangled the legacy T/I/F channels are (Jacobian entanglement index),
2. the orthogonal encoding: selected ranks, exact R^2 split, exact pi_G,
3. process alignment: which mineral / end-member directions lie in the
   baseline subspace and which in the perturbation subspace,
4. scenario analysis: predicted log-TDS response to +1 meq/L of a process,
5. mineral identifiability (aliased minerals) and Jacobian error bars.

Run from the repository root:
    python examples/jacobian_orthogonality_demo.py
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from neutrohydro import NeutroHydroPipeline
from neutrohydro.pipeline import PipelineConfig
from neutrohydro.minerals import MineralInverter, STANDARD_IONS
from neutrohydro.orthogonal import (
    jacobian_entanglement,
    subspace_alignment,
    directions_to_standardized,
    scenario_response,
)

ROOT = Path(__file__).resolve().parents[1]

# mg/L -> meq/L (equivalent weights, g/eq)
EQ_WEIGHT = {
    "Ca": 20.04, "Mg": 12.15, "Na": 22.99, "K": 39.10, "HCO3": 61.02,
    "Cl": 35.45, "SO4": 48.03, "NO3": 62.00, "F": 19.00,
}
COLUMNS = ["Ca", "Mg", "Na", "K", "HCO3", "Cl", "SO4", "NO3", "F"]
IONS = STANDARD_IONS[:9]  # same order: Ca2+, Mg2+, Na+, K+, HCO3-, Cl-, SO42-, NO3-, F-

# A few end-member directions (meq/L per unit), beyond the mineral library
END_MEMBERS = {
    # Standard seawater composition, meq/L
    "Seawater": {"Ca2+": 20.6, "Mg2+": 107.0, "Na+": 469.0, "K+": 10.2,
                 "HCO3-": 2.3, "Cl-": 546.0, "SO42-": 56.5},
    # Nitrogen fertiliser / wastewater style input
    "NH4NO3-fertiliser (nitrified)": {"NO3-": 1.0, "Ca2+": 0.5, "Mg2+": 0.5},
}


def stoich_vector(stoich: dict) -> np.ndarray:
    return np.array([stoich.get(ion, 0.0) for ion in IONS], dtype=float)


def main() -> None:
    df = pd.read_csv(ROOT / "data.csv", encoding="utf-8-sig")
    c_meq = np.column_stack([df[c].to_numpy(float) / EQ_WEIGHT[c] for c in COLUMNS])
    y = np.log10(df["TDS"].to_numpy(float))

    print(f"{len(df)} samples, ions: {', '.join(IONS)}; target: log10(TDS)\n")

    # --- 1. legacy encoding --------------------------------------------
    legacy = NeutroHydroPipeline(PipelineConfig(log_transform=True, n_components=3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_l = legacy.fit(c_meq, y, feature_names=IONS)
        X_std, _ = res_l.preprocessor.transform(c_meq)
        ent_l = jacobian_entanglement(res_l.encoder, X_std, model=res_l.model)
    print("LEGACY (robust_pca rank 2)")
    print(f"  R2 = {res_l.r2_train:.3f}")
    print("  Jacobian entanglement of the model's channels (0 = disentangled):",
          {k: round(ent_l[k], 3) for k in ("T-I", "T-F", "I-F")})

    # --- 2. orthogonal encoding ----------------------------------------
    ortho_pipe = NeutroHydroPipeline(PipelineConfig(
        log_transform=True, n_components=3,
        baseline_type="orthogonal", baseline_rank=None,
    ))
    res_o = ortho_pipe.fit(c_meq, y, feature_names=IONS)
    enc, at = res_o.encoder, res_o.orthogonal
    b = enc.basis_
    ent_o = jacobian_entanglement(enc, X_std)
    print("\nORTHOGONAL")
    print(f"  ranks: T={b.rank_T}, I={b.rank_I}, F={len(IONS) - b.rank_T - b.rank_I}"
          f"  (bootstrap rank counts: {b.rank_selection['bootstrap_ranks'].tolist()})")
    print(f"  R2 = {res_o.r2_train:.3f}; weighted R2 = {at.R2:.3f} = "
          f"T {at.R2_channels[0]:.3f} + I {at.R2_channels[1]:.3f} + F {at.R2_channels[2]:.3f}")
    print("  Jacobian entanglement:", {k: (None if np.isnan(ent_o[k]) else round(ent_o[k], 6))
                                       for k in ("T-I", "T-F", "I-F")})
    print("  exactness checks:", {k: f"{v:.1e}" for k, v in at.checks.items()})
    print("\n  Per-ion exact attribution (shares of predictive energy):")
    print(at.ions_dataframe()[["ion", "pi_T", "pi_I", "pi_F"]].round(3).to_string(index=False))

    # --- 3. process alignment -------------------------------------------
    inverter = MineralInverter(ion_order=IONS)
    directions = {name: inverter.A[:, k] for k, name in enumerate(inverter.mineral_names)}
    directions.update({name: stoich_vector(v) for name, v in END_MEMBERS.items()})
    D_std = directions_to_standardized(np.array(list(directions.values())), res_o.preprocessor)
    align = subspace_alignment(enc, D_std, names=list(directions.keys()))
    print("\n  Process alignment at the median composition: share of each process")
    print("  direction in T/I/F (sums to 1) and enrichment over a random direction")
    print(f"  (chance levels T {b.rank_T / len(IONS):.2f}, I {b.rank_I / len(IONS):.2f}, "
          f"F {(len(IONS) - b.rank_T - b.rank_I) / len(IONS):.2f}):")
    print(align.round(2).sort_values("enrich_T", ascending=False).to_string(index=False))

    # --- 4. scenario ------------------------------------------------------
    s_y = res_o.preprocessor.params_.s_y
    names = ["Halite", "Seawater", "NH4NO3-fertiliser (nitrified)"]
    unit = []
    for nme in names:
        v = directions[nme]
        unit.append(v / v.sum())  # +1 meq/L total of dissolved ions
    sc = scenario_response(enc, res_o.model, directions_to_standardized(np.array(unit), res_o.preprocessor), s_y)
    print("\n  Scenario: +1 meq/L of a process at the median composition")
    for k, nme in enumerate(names):
        bt, bi, bf = sc["by_channel"][k]
        print(f"    {nme:32s} d log10(TDS) = {sc['response'][k]:+.4f}"
              f"  (T {bt:+.4f}, I {bi:+.4f}, F {bf:+.4f})")

    # --- 5. mineral identifiability and error bars ----------------------
    rep = inverter.identifiability()
    print("\nMINERAL INVERSION")
    print(f"  {len(inverter.mineral_names)} minerals on {len(IONS)} ions; rank {rep['rank']};"
          f" underdetermined: {rep['underdetermined']}")
    print(f"  alias groups (inseparable by these ions): {rep['alias_groups']}")
    inv = inverter.invert(c_meq, rel_error=0.05)
    i = int(np.argmax(c_meq.sum(axis=1)))
    sup = np.where(inv.s[i] > 0)[0]
    print(f"  most mineralised sample ({df.iloc[i, 0]}), 5% analytical error:")
    for k in sup:
        print(f"    {inverter.mineral_names[k]:14s} {inv.s[i, k]:8.3f} +/- {inv.s_std[i, k]:.3f}")
    print(f"    support condition number {inv.support_condition[i]:.1f}, "
          f"orthogonality defect {inv.support_defect[i]:.2f}")


if __name__ == "__main__":
    main()
