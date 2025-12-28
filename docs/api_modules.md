# Core Modules API

## `neutrohydro.encoder`

Handles the Neutrosophic Data Transformation.

- **`NDGEncoder`**: Transforms raw concentrations into Truth (T), Indeterminacy (I), and Falsity (F) components.

## `neutrohydro.minerals`

Handles Stoichiometric Inversion.

- **`MineralInverter`**: Performs weighted NNLS inversion to estimate mineral contributions.
- **`calculate_simpson_ratio`**: Computes Standard and Inverse Simpson's Ratios.

## `neutrohydro.quality_check`

Handles Water Quality Assessment.

- **`assess_water_quality`**: Checks samples against WHO guidelines.
- **`add_quality_flags`**: Adds quality columns to a DataFrame.

## `neutrohydro.nvip`

Handles Variable Importance.

- **`calculate_nvip`**: Computes Neutrosophic Variable Importance in Projection.
