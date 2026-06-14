# Machine Learning-Based Surrogate Modelling for Aerodynamic Performance Prediction

This project builds **surrogate models** that predict aerodynamic coefficients from flight conditions, using a small CFD dataset instead of running expensive simulations for every point in the flight envelope.

All code runs **fully offline** after Python packages are installed — no internet, cloud APIs, or TensorFlow/PyTorch required.

---

## Problem & Objective

High-fidelity CFD simulations are slow and costly when many flight conditions must be analysed. This pipeline:

1. Loads CFD results (Mach, AoA, Altitude → CL, CD, CM)
2. Trains and compares **13 ML regressors** with hyperparameter tuning
3. Evaluates models with **RMSE**, **MAE**, and **R²**
4. Generates flight-envelope contour maps and report-ready plots

---

## Dataset

**Inputs (flight conditions)**

| Variable | Description |
|----------|-------------|
| `Mach` | Mach number |
| `AoA_deg` | Angle of attack (degrees) |
| `Altitude_m` | Altitude (metres) |

**Outputs (aerodynamic coefficients)**

| Variable | Description |
|----------|-------------|
| `CL` | Lift coefficient |
| `CD` | Drag coefficient |
| `CM` | Moment coefficient |

Data file: `data/aero_data.csv` (17 CFD samples)

---

## Models Compared

| Category | Algorithms |
|----------|------------|
| Surrogate / probabilistic | Gaussian Process (Kriging) |
| Tree-based | Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGradient Boosting, AdaBoost, XGBoost |
| Instance / kernel | k-Nearest Neighbors, Support Vector Regression |
| Linear | Ridge Regression, Elastic Net |
| Neural network | Artificial Neural Network (sklearn MLPRegressor) |

Hyperparameters are tuned with **3-fold cross-validation** (GridSearchCV).

---

## Requirements

- Python 3.10+
- Packages listed in `requirements.txt`:

```
numpy, pandas, scikit-learn, scipy, xgboost, matplotlib
```

Install once (if packages are not already on your machine):

```powershell
pip install -r requirements.txt
```

> **Offline use:** After installation, no network connection is needed. Do not run `pip install` on an offline PC unless packages are already installed.

---

## How to Run

### Full pipeline (train + evaluate + plots)

Double-click **`run.bat`**, or from the project folder:

```powershell
python main.py
```

This will:

- Train all 13 models
- Save the best model to `models/best_model.pkl`
- Write metrics to `results/model_comparison.json` and `results/model_comparison_table.csv`
- Generate all plots in `results/plots/`

### Regenerate plots only (no retraining)

```powershell
python scripts/generate_plots.py
```

Use this when results and `best_model.pkl` already exist and you only need updated figures.

---

## Project Structure

```
Aero project/
├── main.py                 # Entry point — train and visualize
├── run.bat                 # One-click run (Windows)
├── requirements.txt        # Python dependencies
├── README.md
├── data/
│   └── aero_data.csv       # CFD training data
├── models/
│   └── best_model.pkl      # Saved best surrogate model (pickle)
├── results/
│   ├── model_comparison.json
│   ├── model_comparison_table.csv
│   └── plots/              # Generated figures
├── scripts/
│   └── generate_plots.py   # Replot without retraining
└── src/
    ├── config.py           # Paths and constants
    ├── data_loader.py      # Load, split, scale data
    ├── metrics.py          # RMSE, MAE, R²
    ├── model_io.py         # Save/load models (pickle)
    ├── models.py           # Model definitions and tuning
    ├── offline.py          # Offline runtime settings
    ├── train.py            # Training pipeline
    └── visualize.py        # All plot generation
```

---

## Outputs

### Metrics

| File | Contents |
|------|----------|
| `results/model_comparison.json` | Full validation & test metrics per model and coefficient |
| `results/model_comparison_table.csv` | Ranked summary table (Excel-friendly) |

### Plots (`results/plots/`)

| Plot | Description |
|------|-------------|
| `model_comparison.png` | Overall RMSE and R² ranking |
| `metrics_heatmap.png` | Per-coefficient RMSE and R² heatmap |
| `per_target_metrics.png` | CL / CD / CM bar charts for top models |
| `predicted_vs_actual_*.png` | Parity plot for the best model |
| `residuals_*.png` | Residual analysis |
| `error_histogram_*.png` | Error distribution |
| `input_space_scatter.png` | CFD data in input space |
| `output_correlation.png` | CL–CD–CM correlation matrix |
| `envelope_with_data_alt_*.png` | Contour maps with CFD points overlaid |
| `3d_surface_alt_5000m.png` | 3D aerodynamic surfaces |

---

## Best Model (current results)

Based on validation RMSE, **Gaussian Process (Kriging)** performs best on this small dataset:

| Metric | Validation | Test |
|--------|------------|------|
| RMSE | 0.039 | 0.032 |
| R² | 0.90 | 0.99 |

GPR is well suited to low-data surrogate modelling. Tree boosters (XGBoost, HistGradient Boosting) tend to overfit with only 17 samples.

---

## Methodology Summary

1. **Data preparation** — StandardScaler on inputs and outputs; train / validation / test split
2. **Model development** — 13 algorithms with GridSearchCV
3. **Validation** — RMSE, MAE, R² on held-out validation and test sets
4. **Flight envelope prediction** — Contour and 3D surface plots across Mach–AoA space

---

## License

Academic / project use.
