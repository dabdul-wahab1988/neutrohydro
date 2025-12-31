"""
Tests for quality_check module.
Verifies the "Sanity Check" logic for geological validity.
"""

import pytest
import pandas as pd
import numpy as np
from neutrohydro.quality_check import (
    check_data_completeness,
    calculate_cbe,
    check_sanity,
    assess_water_quality,
    REQUIRED_IONS
)

class TestQualityCheck:
    
    @pytest.fixture
    def good_data(self):
        """Create a balanced, complete dataset."""
        # Adjusted for better Charge Balance (< 5%)
        return pd.DataFrame({
            "Ca": [40.0, 45.0],      # ~2.0 meq
            "Mg": [24.0, 25.0],      # ~2.0 meq
            "Na": [23.0, 20.0],      # ~1.0 meq
            "K": [39.0, 40.0],       # ~1.0 meq
            # Sum Cations ~ 6.0 meq
            
            "HCO3": [170.0, 175.0],  # ~2.8 meq
            "Cl": [35.5, 36.0],      # ~1.0 meq
            "SO4": [96.0, 95.0],     # ~2.0 meq
            "NO3": [10.0, 5.0],      # ~0.16 meq
            # Sum Anions ~ 5.96 meq
            
            "pH": [7.2, 7.5],
            "TDS": [400.0, 450.0]
        })

    def test_completeness_check_pass(self, good_data):
        """Test that complete data passes."""
        missing = check_data_completeness(good_data.columns.tolist())
        assert len(missing) == 0

    def test_completeness_check_fail(self, good_data):
        """Test that missing ions are detected."""
        bad_data = good_data.drop(columns=["HCO3", "Mg"])
        missing = check_data_completeness(bad_data.columns.tolist())
        assert "HCO3" in missing
        assert "Mg" in missing
        assert len(missing) == 2

    def test_cbe_calculation_balanced(self):
        """Test CBE for a perfectly balanced water."""
        # 1 meq/L of Na (23 mg) and 1 meq/L of Cl (35.5 mg)
        row = {"Na": 22.99, "Cl": 35.45}
        cbe = calculate_cbe(row)
        assert abs(cbe) < 0.1 # Should be near 0

    def test_cbe_calculation_imbalanced(self):
        """Test CBE for imbalanced water (High Error)."""
        # Only Cations: 1 meq/L Na, 0 Anions
        row = {"Na": 22.99, "Cl": 0.0}
        cbe = calculate_cbe(row)
        # (1 - 0) / (1 + 0) * 100 = 100%
        assert cbe > 99.0

    def test_extreme_contamination(self, good_data):
        """Test detection of 'Swamp Effect' / Extreme Contamination."""
        # Add a sample with extreme Cl
        extreme_row = good_data.iloc[0].copy()
        extreme_row["Cl"] = 15000.0 # > 10,000 limit
        
        df = pd.concat([good_data, pd.DataFrame([extreme_row])], ignore_index=True)
        report = check_sanity(df)
        
        assert report["extreme_samples"] == 1
        assert any("extreme contamination" in w for w in report["warnings"])

    def test_sanity_report_valid(self, good_data):
        """Test full sanity report on good data."""
        report = check_sanity(good_data)
        assert report["valid"] is True
        assert report["high_cbe_count"] == 0
        assert report["extreme_samples"] == 0
        assert len(report["warnings"]) == 0

    def test_sanity_report_invalid(self, good_data):
        """Test full sanity report on invalid data."""
        bad_data = good_data.drop(columns=["Ca"])
        report = check_sanity(bad_data)
        
        assert report["valid"] is False
        assert "Ca" in report["missing_ions"]
        assert any("Missing critical ions" in w for w in report["warnings"])

    def test_who_assessment(self):
        """Test WHO guideline assessment."""
        row = {
            "Na": 250.0, # Exceeds 200
            "NO3": 60.0, # Exceeds 50
            "pH": 6.0    # Below 6.5
        }
        res = assess_water_quality(row)
        
        assert "Na" in res["Exceedances"]
        assert "NO3" in res["Exceedances"]
        assert "pH (Acidic)" in res["Exceedances"]
        assert "Anthropogenic (Agri/Sewage)" in res["Inferred_Sources"]
