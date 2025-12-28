# Water Quality Assessment

**Module**: `neutrohydro.quality_check`

## Overview

The Quality Assessment module provides an automated system for evaluating groundwater samples against **WHO (World Health Organization)** drinking water guidelines. Beyond simple compliance checking, it implements an **intelligent source inference** engine that interprets combinations of exceedances to suggest potential pollution origins.

## Features

1.  **WHO Compliance Check**: Automatically flags parameters exceeding standard limits.
2.  **Source Inference**: Uses hydrogeochemical logic to infer the likely cause of contamination (e.g., Saline Intrusion vs. Anthropogenic Pollution).
3.  **Integration**: Can be used as a standalone tool or to provide **context-aware constraints** for the Mineral Inversion model.

## Mathematical Logic

### 1. Thresholds

The module uses standard WHO guideline values (mg/L):

| Parameter | Limit |
| :--- | :--- |
| **TDS** | 1000 |
| **pH** | 6.5 - 8.5 |
| **Na** | 200 |
| **Cl** | 250 |
| **SO4** | 250 |
| **NO3** | 50 |
| **F** | 1.5 |
| **Heavy Metals** | Various (e.g., Pb 0.01, As 0.01) |

### 2. Source Inference Rules

The module applies a set of heuristic rules to infer sources based on specific combinations of exceedances:

#### 2.1 Saline Intrusion
*   **Trigger**: High Chloride ($Cl > 250$) **AND** High Sodium ($Na > 200$).
*   **Inference**: "Saline Intrusion/Brine".
*   **Implication**: Suggests seawater mixing or deep brine upwelling.

#### 2.2 Anthropogenic Pollution
*   **Trigger**: High Nitrate ($NO_3 > 50$).
*   **Inference**: "Anthropogenic (Agri/Sewage)".
*   **Implication**: Surface contamination from fertilizers or wastewater.

#### 2.3 Industrial/Mining
*   **Trigger**: High Sulfate ($SO_4 > 250$) **WITHOUT** High Calcium (which would suggest Gypsum).
*   **Inference**: "Industrial/Mining".
*   **Implication**: Acid mine drainage or industrial effluent.

#### 2.4 Geogenic (Rock-Water Interaction)
*   **Trigger**: High Fluoride ($F > 1.5$) or High Calcium/Sulfate (Gypsum).
*   **Inference**: "Geogenic (Rock-Water)".
*   **Implication**: Natural weathering of specific mineral formations.

## Usage

### Standalone Assessment

```python
import pandas as pd
from neutrohydro.quality_check import add_quality_flags

# Load Data
df = pd.read_csv("data.csv")

# Run Assessment
df_report = add_quality_flags(df)

# View Results
print(df_report[['Code', 'Exceedances', 'Inferred_Sources']])
```

### Integration with Mineral Inversion

The inferred sources can be passed to the `MineralInverter` to override standard constraints. For example, if "Saline Intrusion" is detected, the inverter will force **Halite** to be considered plausible, even if other indices (like Gibbs) suggest otherwise.

```python
# 1. Get Quality Flags
quality_flags = df_report.to_dict('records')

# 2. Run Inversion with Flags
result = inverter.invert(
    c=concentrations,
    quality_flags=quality_flags
)
```
