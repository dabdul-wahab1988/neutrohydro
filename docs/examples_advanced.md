# Advanced Workflows

## Custom Mineral Libraries

You can define custom minerals for the inversion engine:

```python
from neutrohydro.minerals import MineralInverter

custom_minerals = {
    "MyMineral": {
        "formula": "X2Y",
        "stoichiometry": {"X": 2.0, "Y": 1.0},
        "description": "A custom phase"
    }
}

inverter = MineralInverter(minerals=custom_minerals)
```

## Handling Redox Processes

To explicitly model redox sinks (like Denitrification), include the `REDOX_PHASES` in your mineral library.

```python
from neutrohydro.minerals import STANDARD_MINERALS, REDOX_PHASES

combined_minerals = {**STANDARD_MINERALS, **REDOX_PHASES}
inverter = MineralInverter(minerals=combined_minerals)
```
