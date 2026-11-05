"""
Tests for thermodynamic speciation module.
"""
import json
import subprocess
import sys
import textwrap

import pytest
import numpy as np
import os
from neutrohydro import speciation
from neutrohydro.speciation import ThermodynamicValidator, PHREEQC_BACKEND

@pytest.fixture
def validator():
    """ThermodynamicValidator backed by phreeqpython (skips if not installed)."""
    pytest.importorskip("phreeqpython")
    return ThermodynamicValidator()

def test_backend_detection():
    """Verify backend is identified when phreeqpython is installed."""
    pytest.importorskip("phreeqpython")
    assert PHREEQC_BACKEND == 'phreeqpython'

def test_validator_init(validator):
    """Test initialization of validator with bundled database."""
    # Check if a database path was set (even if phreeqpython default)
    assert validator.pp is not None

def test_si_calculation_mock(validator):
    """Test SI calculation logic with mock data."""
    # This might fail if phreeqpython is not actually functional in this environment
    # but let's try a simple case.
    
    # 1 sample, Ca, Mg, HCO3, Cl
    c = np.array([[2.0, 1.0, 2.0, 1.0]]) 
    ion_names = ["Ca2+", "Mg2+", "HCO3-", "Cl-"]
    pH = np.array([7.5])
    
    si = validator.calculate_si(c, ion_names, pH)
    
    assert "Calcite" in si
    assert isinstance(si["Calcite"], np.ndarray)
    assert len(si["Calcite"]) == 1

def test_validate_dissolution(validator):
    """Test plausibility filtering logic."""
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

def test_validator_without_phreeqpython_names_extra(monkeypatch):
    """Without phreeqpython, the validator raises an ImportError naming the extra."""
    monkeypatch.setattr(speciation, "phreeqpython", None)
    monkeypatch.setattr(speciation, "PHREEQC_BACKEND", None)
    with pytest.raises(ImportError, match=r"neutrohydro\[thermo\]"):
        ThermodynamicValidator()

# Runs in a fresh interpreter, where import-time side effects are observable.
_IMPORT_WITHOUT_PHREEQPYTHON = textwrap.dedent("""
    import json, logging, sys

    sys.modules["phreeqpython"] = None  # behave as if phreeqpython is not installed

    pip_calls = []

    def block_pip(event, args):
        if event in ("subprocess.Popen", "os.system"):
            cmd = str(args[1] if event == "subprocess.Popen" else args[0])
            if "pip" in cmd:
                pip_calls.append(cmd)
                raise RuntimeError("pip must not run during import")

    sys.addaudithook(block_pip)
    root = logging.getLogger()
    root_before = (list(root.handlers), root.level)

    import neutrohydro
    from neutrohydro import speciation

    print(json.dumps({
        "backend": speciation.PHREEQC_BACKEND,
        "pip_calls": pip_calls,
        "root_logger_changed": (list(root.handlers), root.level) != root_before,
    }))
""")

def test_import_without_phreeqpython_has_no_side_effects():
    """Importing without phreeqpython must not run pip or configure the root logger."""
    proc = subprocess.run(
        [sys.executable, "-c", _IMPORT_WITHOUT_PHREEQPYTHON],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout.splitlines()[-1]) == {
        "backend": None,
        "pip_calls": [],
        "root_logger_changed": False,
    }
