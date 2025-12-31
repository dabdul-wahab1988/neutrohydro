import pytest
import pandas as pd
import numpy as np
from neutrohydro.interpretation import GeochemicalInterpreter, Insight
from neutrohydro.pipeline import PipelineResults
from neutrohydro.minerals import MineralInversionResult

# Mock PipelineResults
@pytest.fixture
def mock_results():
    n_samples = 5
    
    # Mock Mineral Result
    mineral_res = MineralInversionResult(
        s=np.zeros((n_samples, 1)),
        residuals=np.zeros((n_samples, 1)),
        residual_norms=np.zeros(n_samples),
        plausible=np.zeros((n_samples, 1), dtype=bool),
        mineral_fractions=np.zeros((n_samples, 1)),
        indices={
            "CAI_1": np.array([-0.5, 0.5, -0.2, 0.1, 0.0]), # Mix of Exchange types
            "CAI_2": np.zeros(n_samples)
        },
        saturation_indices={
            "Fluorite": np.array([1.0, 1.0, -1.0, 0.0, 0.5]) # Supersat, Supersat, Undersat, Equil, Supersat
        },
        mineral_names=["Fluorite"]
    )
    
    # Mock PipelineResults (minimal needed for interpreter)
    results = PipelineResults(
        preprocessor=None,
        encoder=None,
        model=None,
        triplets=None,
        nvip=None,
        nsr=None,
        sample_attribution=None,
        y_pred=np.zeros(n_samples),
        y_pred_original=np.zeros(n_samples),
        r2_train=0.9,
        mineral_result=mineral_res
    )
    return results

def test_interpreter_initialization(mock_results):
    interpreter = GeochemicalInterpreter(mock_results)
    assert interpreter.results == mock_results
    assert interpreter.geology == "Unknown/Mixed" # Default

def test_geology_analysis(mock_results):
    # Test Granitic
    interpreter = GeochemicalInterpreter(mock_results, geology="Granitic/Gneissic (High F potential)")
    report = interpreter.interpret()
    
    geo_insights = [i for i in report.insights if i.category == "Geology"]
    assert len(geo_insights) == 1
    assert "Silicate Weathering" in geo_insights[0].message
    assert "Biotite" in geo_insights[0].message

    # Test Basaltic
    interpreter = GeochemicalInterpreter(mock_results, geology="Basaltic/Volcanic (Hard Rock)")
    report = interpreter.interpret()
    geo_insights = [i for i in report.insights if i.category == "Geology"]
    assert "ferromagnesian" in geo_insights[0].message

    # Test Sedimentary Carbonate
    interpreter = GeochemicalInterpreter(mock_results, geology="Sedimentary - Carbonate")
    report = interpreter.interpret()
    geo_insights = [i for i in report.insights if i.category == "Geology"]
    assert "Carbonate Dissolution" in geo_insights[0].message

def test_fluoride_mechanism_detection(mock_results):
    interpreter = GeochemicalInterpreter(mock_results)
    report = interpreter.interpret()
    
    df = report.fluoride_source_summary
    assert not df.empty
    assert len(df) == 5
    
    # Sample 0: SI=1.0 (Supersat), CAI=-0.5 (Ion Exchange)
    # Expect: Silicate Weathering + Ca-Removal
    row0 = df.iloc[0]
    assert row0["Status"] == "Supersaturated (Not Fluorite Dissolution)"
    assert row0["Likely_Mechanism"] == "Silicate Weathering + Ca-Removal (Ion Exchange)"
    
    # Sample 1: SI=1.0 (Supersat), CAI=0.5 (Reverse Exchange/Desorption)
    # Expect: Desorption...
    row1 = df.iloc[1]
    assert row1["Likely_Mechanism"] == "Desorption / Silicate Weathering / Anthropogenic"
    
    # Sample 2: SI=-1.0 (Undersat)
    # Expect: Fluorite Dissolution Possible
    row2 = df.iloc[2]
    assert row2["Status"] == "Undersaturated"
    assert row2["Likely_Mechanism"] == "Fluorite Dissolution Possible"

def test_insight_generation(mock_results):
    interpreter = GeochemicalInterpreter(mock_results)
    report = interpreter.interpret()
    
    # Check if Source insight is generated for supersaturated samples
    source_insights = [i for i in report.insights if i.category == "Source"]
    assert len(source_insights) >= 1
    assert "Fluorite is supersaturated" in source_insights[0].message

def test_missing_mineral_results():
    # Test graceful handling if mineral_result is None
    empty_results = PipelineResults(
        preprocessor=None, encoder=None, model=None, triplets=None, 
        nvip=None, nsr=None, sample_attribution=None, 
        y_pred=np.zeros(5), y_pred_original=np.zeros(5), r2_train=0.9,
        mineral_result=None
    )
    interpreter = GeochemicalInterpreter(empty_results)
    report = interpreter.interpret()
    assert report.fluoride_source_summary.empty
    assert len(report.insights) == 0 # Or whatever default behavior is
