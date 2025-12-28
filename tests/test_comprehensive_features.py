
import pytest
import numpy as np
import pandas as pd
from neutrohydro.minerals import (
    MineralInverter, 
    calculate_simpson_ratio, 
    STANDARD_MINERALS, 
    REDOX_PHASES,
    STANDARD_IONS
)
from neutrohydro.quality_check import assess_water_quality, add_quality_flags

def test_simpson_ratio_classification():
    """Test Simpson's Ratio (Standard and Inverse) for different water types."""
    
    # Case 1: Fresh Water
    # Cl = 1.0, HCO3 = 4.0 -> Ratio = 0.25 (< 0.5) -> Fresh
    # Inverse = 4.0 / 1.0 = 4.0 (> 1) -> Recharge
    c_fresh = np.array([[1.0, 4.0]]) # Cl, HCO3
    r_std, r_inv = calculate_simpson_ratio(c_fresh, ["Cl-", "HCO3-"])
    
    assert r_std[0] < 0.5, f"Expected Fresh (<0.5), got {r_std[0]}"
    assert r_inv[0] > 1.0, f"Expected Recharge (>1.0), got {r_inv[0]}"

    # Case 2: Seawater Intrusion
    # Cl = 20.0, HCO3 = 1.0 -> Ratio = 20.0 (> 15.5) -> Extremely Saline
    # Inverse = 1.0 / 20.0 = 0.05 (< 0.5) -> Intrusion
    c_sea = np.array([[20.0, 1.0]])
    r_std, r_inv = calculate_simpson_ratio(c_sea, ["Cl-", "HCO3-"])
    
    assert r_std[0] > 15.5, f"Expected Extremely Saline (>15.5), got {r_std[0]}"
    assert r_inv[0] < 0.5, f"Expected Intrusion (<0.5), got {r_inv[0]}"

def test_who_quality_flags():
    """Test WHO Quality Assessment logic."""
    
    # Case 1: Saline Intrusion (High Na, High Cl)
    row_saline = {"Na": 300.0, "Cl": 400.0, "TDS": 1200.0} # Exceeds 200, 250, 1000
    res = assess_water_quality(row_saline)
    
    assert "TDS" in res["Exceedances"]
    assert "Na" in res["Exceedances"]
    assert "Cl" in res["Exceedances"]
    assert "Saline Intrusion/Brine" in res["Inferred_Sources"]

    # Case 2: Anthropogenic Nitrate
    row_agri = {"NO3": 60.0, "K": 10.0} # Exceeds 50
    res = assess_water_quality(row_agri)
    
    assert "NO3" in res["Exceedances"]
    assert "Anthropogenic (Agri/Sewage)" in res["Inferred_Sources"]

def test_redox_inversion():
    """
    Test if the inverter correctly uses a Sink phase to explain missing mass.
    Scenario: High K, Zero NO3.
    Only K source is Niter (KNO3).
    Solver must use Niter to fit K, and Sink_Denitrification to remove NO3.
    """
    
    # 1. Setup Custom Library: Niter + Denitrification only (to isolate effect)
    custom_minerals = {
        "Niter": STANDARD_MINERALS["Niter"],
        "Sink_Denitrification": REDOX_PHASES["Sink_Denitrification"]
    }
    
    # 2. Input Data: 1.0 meq/L K+, 0.0 meq/L NO3-
    # We need to provide all standard ions to match the default ion_order, or specify ion_order
    ion_order = ["K+", "NO3-", "HCO3-"]
    c_input = np.array([[1.0, 0.0, 0.0]]) # K=1, NO3=0, HCO3=0 (initially)
    
    # 3. Invert
    inverter = MineralInverter(minerals=custom_minerals, ion_order=ion_order)
    result = inverter.invert(c_input)
    
    # 4. Check Results
    # Expected: Niter = 1.0 (provides 1 K, 1 NO3)
    #           Sink = 1.0 (removes 1 NO3, adds 1 HCO3)
    #           Resulting HCO3 should be 1.0 (residual if not fitted, or fitted if we had a sink for it)
    #           Wait, Sink adds HCO3. So modeled HCO3 = 1.0. Observed = 0.0.
    #           So residual HCO3 = 0 - 1 = -1.
    
    idx_niter = list(custom_minerals.keys()).index("Niter")
    idx_sink = list(custom_minerals.keys()).index("Sink_Denitrification")
    
    niter_contrib = result.s[0, idx_niter]
    sink_contrib = result.s[0, idx_sink]
    
    print(f"\nRedox Test Results:")
    print(f"Niter Contribution: {niter_contrib:.4f} (Expected ~1.0)")
    print(f"Sink Contribution:  {sink_contrib:.4f} (Expected ~1.0)")
    
    # NNLS distributes error, so we might not get exactly 1.0/1.0 if other ions (HCO3) are affected.
    # But Sink MUST be active to reduce the NO3 error.
    assert sink_contrib > 0.1, "Sink_Denitrification should be active to explain missing Nitrate"

def test_hybrid_constraints():
    """Test if Quality Flags override Gibbs constraints."""
    
    # Setup: Rock Dominance water (Low Cl/HCO3, Low Na/Ca)
    # But with some Na and SO4 that *could* be Mirabilite (Na2SO4).
    # Gibbs Constraint: Should BAN Mirabilite in Rock Dominance.
    # Quality Override: If we flag "Gypsum" (implies evaporites), it should ALLOW Mirabilite.
    
    ion_order = STANDARD_IONS
    # Indices: Ca(0), Mg(1), Na(2), K(3), HCO3(4), Cl(5), SO4(6)
    c_rock = np.zeros((1, len(ion_order)))
    c_rock[0, 0] = 10.0 # Ca (High -> Low Na/Ca ratio)
    c_rock[0, 4] = 10.0 # HCO3 (High -> Low Cl/HCO3 ratio)
    c_rock[0, 2] = 2.0  # Na
    c_rock[0, 6] = 2.0  # SO4
    
    # 1. Test Constraint (Ban Mirabilite)
    inverter = MineralInverter()
    res_constrained = inverter.invert(
        c_rock, 
        use_gibbs_constraints=True, 
        quality_flags=[{"Inferred_Sources": "Natural/Safe"}]
    )
    
    idx_mirabilite = list(STANDARD_MINERALS.keys()).index("Mirabilite")
    mirabilite_constrained = res_constrained.s[0, idx_mirabilite]
    
    print(f"\nHybrid Constraint Test:")
    print(f"Mirabilite (Rock Dominance): {mirabilite_constrained:.4f}")
    assert mirabilite_constrained == 0.0, "Mirabilite should be banned in Rock Dominance"
    
    # 2. Test Override (Allow Mirabilite)
    # We force the flag "Gypsum" which enables sulfates
    res_override = inverter.invert(
        c_rock, 
        use_gibbs_constraints=True, 
        quality_flags=[{"Inferred_Sources": "Gypsum"}]
    )
    
    mirabilite_override = res_override.s[0, idx_mirabilite]
    print(f"Mirabilite (Override 'Gypsum'): {mirabilite_override:.4f}")
    
    # With override, it should be allowed. 
    # Since Na and SO4 are present (2.0 each), Mirabilite (Na2SO4) is a perfect fit.
    assert mirabilite_override > 0.0, "Mirabilite should be allowed when 'Gypsum' source is flagged"

if __name__ == "__main__":
    # Manual run for quick feedback
    try:
        test_simpson_ratio_classification()
        print("Simpson Ratio: PASS")
    except AssertionError as e:
        print(f"Simpson Ratio: FAIL - {e}")

    try:
        test_who_quality_flags()
        print("WHO Quality: PASS")
    except AssertionError as e:
        print(f"WHO Quality: FAIL - {e}")

    try:
        test_redox_inversion()
        print("Redox Inversion: PASS")
    except AssertionError as e:
        print(f"Redox Inversion: FAIL - {e}")
        
    try:
        test_hybrid_constraints()
        print("Hybrid Constraints: PASS")
    except AssertionError as e:
        print(f"Hybrid Constraints: FAIL - {e}")
