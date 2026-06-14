"""Offline runtime settings — no network access required at run time."""

import os

# Non-interactive plot backend (no GUI, no font downloads).
os.environ.setdefault("MPLBACKEND", "Agg")

# Block accidental network use from common libraries.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

OFFLINE_LIBRARIES = (
    "numpy - numerical arrays",
    "pandas - tabular data",
    "scikit-learn - GPR, Random Forest, MLP (ANN), preprocessing, metrics",
    "xgboost - gradient boosting regressor",
    "matplotlib - contour / envelope plots",
    "scipy - scientific routines (sklearn dependency)",
    "pickle (stdlib) - model persistence",
)
