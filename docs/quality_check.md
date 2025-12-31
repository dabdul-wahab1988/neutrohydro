# Water Quality Assessment

**Module**: `neutrohydro.quality_check`

## Overview

The Quality Assessment module provides two levels of validation:
1. **Data Sanity Checks**: Strict validation of chemical plausibility (Charge Balance, Completeness).
2. **Water Quality Assessment**: Evaluation against WHO guidelines and source inference.

## 1. Data Sanity Checks (Critical Validity)

Before any analysis, NeutroHydro enforces strict geological validity to prevent "garbage-in, garbage-out".

### 1.1 Charge Balance Error (CBE)
Ensures electroneutrality. Samples with **CBE > 15%** are flagged as unreliable.
$$ CBE (\%) = \frac{\sum z \cdot m_{cations} - \sum z \cdot m_{anions}}{\sum z \cdot m_{cations} + \sum z \cdot m_{anions}} \times 100 $$

### 1.2 Completeness Check
Verifies that all major ions (Ca, Mg, Na, K, HCO3, Cl, SO4) are present. Missing major ions make geochemical modeling impossible.

### 1.3 Extreme Value Detection
Flags physically impossible or extreme values (e.g., Cl > 10,000 mg/L without corresponding Na) that suggest measurement error or extreme brine conditions requiring specialized handling.

## 2. Water Quality Assessment (WHO)

Evaluates samples against **WHO (World Health Organization)** drinking water guidelines.

### 2.1 Thresholds

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

* **Trigger**: High Chloride ($Cl > 250$) **AND** High Sodium ($Na > 200$).
* **Inference**: "Saline Intrusion/Brine".
* **Implication**: Suggests seawater mixing or deep brine upwelling.

#### 2.2 Anthropogenic Pollution

* **Trigger**: High Nitrate ($NO_3 > 50$).
* **Inference**: "Anthropogenic (Agri/Sewage)".
* **Implication**: Surface contamination from fertilizers or wastewater.

#### 2.3 Industrial/Mining

* **Trigger**: High Sulfate ($SO_4 > 250$) **WITHOUT** High Calcium (which would suggest Gypsum).
* **Inference**: "Industrial/Mining".
* **Implication**: Acid mine drainage or industrial effluent.

#### 2.4 Geogenic (Rock-Water Interaction)

* **Trigger**: High Fluoride ($F > 1.5$) or High Calcium/Sulfate (Gypsum).
* **Inference**: "Geogenic (Rock-Water)".
* **Implication**: Natural weathering of specific mineral formations.

### 3. Supporting Indices for Inference

While exceedances trigger the inference, the module and `MineralInverter` use secondary indices to confirm the process:

| Process | Primary Index | Secondary Evidence |
| :--- | :--- | :--- |
| **Saline Intrusion** | Simpson Ratio (SR) > 15.5 | BEX < 0, Gibbs Anion > 0.8 |
| **Freshening/Recharge** | Freshening Ratio (FR) > 1 | BEX > 0, Gibbs Anion < 0.2 |
| **Ion Exchange** | CAI-1/CAI-2 | BEX ≠ 0 |

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
