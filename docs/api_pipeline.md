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

    def fit(self, df: pd.DataFrame):
        """
        Fit the internal models (Scaler, Encoder, PNPLS) to the data.
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
