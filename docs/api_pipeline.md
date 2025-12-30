# Pipeline API

## NeutroHydroPipeline

The `NeutroHydroPipeline` class orchestrates the entire workflow, from data loading to result generation.

### Class Reference

```python
class NeutroHydroPipeline:
    def __init__(self, target_ions: list[str], ...):
        """
        Initialize the pipeline.
        
        Args:
            target_ions: List of ions to model (e.g., ['Ca', 'Mg', ...])
        """
        ...

    def fit(self, X, y, feature_names=None, groups=None, c_meq=None, pH=None, Eh=None, temp=25.0):
        """
        Fit the internal models and optionally perform mineral inference.
        
        Args:
            X: Predictor matrix (samples x features)
            y: Target variable
            feature_names: Optional list of feature names
            groups: Optional grouping variable for cross-validation
            c_meq: Ion concentrations in meq/L for mineral inference
            pH: pH values for thermodynamic validation
            Eh: Redox potential (mV) for thermodynamic validation
            temp: Temperature (Celsius) for thermodynamic validation
        """
        ...

    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Run the full analysis on the provided data.
        
        Returns:
            dict: A dictionary containing:
                - 'vip_scores': Variable Importance
                - 'mineral_fractions': Mineral inversion results
                - 'quality_flags': WHO assessment
                - 'indices': Hydrogeochemical indices
        """
        ...
```
