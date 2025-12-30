"""
Tests for thermodynamic speciation module.
"""
import pytest
import numpy as np
import os
from neutrohydro.speciation import ThermodynamicValidator, PHREEQC_BACKEND

def test_backend_detection():
    """Verify backend is identified (or auto-installed)."""
    assert PHREEQC_BACKEND == 'phreeqpython'

def test_validator_init():
    """Test initialization of validator with bundled database."""
    validator = ThermodynamicValidator()
    # Check if a database path was set (even if phreeqpython default)
    assert validator.pp is not None

def test_si_calculation_mock():
    """Test SI calculation logic with mock data."""
    # This might fail if phreeqpython is not actually functional in this environment
    # but let's try a simple case.
    validator = ThermodynamicValidator()
    
    # 1 sample, Ca, Mg, HCO3, Cl
    c = np.array([[2.0, 1.0, 2.0, 1.0]]) 
    ion_names = ["Ca2+", "Mg2+", "HCO3-", "Cl-"]
    pH = np.array([7.5])
    
    si = validator.calculate_si(c, ion_names, pH)
    
    assert "Calcite" in si
    assert isinstance(si["Calcite"], np.ndarray)
    assert len(si["Calcite"]) == 1

def test_validate_dissolution():
    """Test plausibility filtering logic."""
    validator = ThermodynamicValidator()
    si_dict = {
        "Calcite": np.array([2.0, -1.0]),  # Supersaturated, Undersaturated
        "Halite": np.array([-5.0, -5.0])   # Undersaturated
    }
    mineral_names = ["Calcite", "Halite"]
    
    plausible = validator.validate_dissolution(si_dict, mineral_names, si_threshold=0.5)
    
    # Sample 0: Calcite (SI=2.0 > 0.5) -> False
    assert plausible[0, 0] == False
    # Sample 1: Calcite (SI=-1.0 < 0.5) -> True
    assert plausible[1, 0] == True
    # Halite always True
    assert plausible[0, 1] == True
    assert plausible[1, 1] == True
