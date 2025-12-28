# Installation Guide

## Prerequisites

- Python 3.9 or higher
- pip (Python package installer)

## Installation from Source

NeutroHydro is currently available as a source distribution. To install it, clone the repository and install using `pip`.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/neutrohydro.git
    cd neutrohydro
    ```

2.  **Create a virtual environment (Recommended):**

    ```bash
    # Windows
    python -m venv .venv
    .venv\Scripts\Activate.ps1

    # Linux/macOS
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the package:**

    ```bash
    pip install .
    ```

    For development (including testing and documentation tools):

    ```bash
    pip install -e .[dev]
    ```

## Verifying Installation

To verify that NeutroHydro is installed correctly, you can run the following command in your terminal:

```bash
python -c "import neutrohydro; print(neutrohydro.__version__)"
```

If installed correctly, this should print the version number without errors.

## Dependencies

The core dependencies are automatically installed:
- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `matplotlib` (for plotting)
- `seaborn` (for advanced visualization)
