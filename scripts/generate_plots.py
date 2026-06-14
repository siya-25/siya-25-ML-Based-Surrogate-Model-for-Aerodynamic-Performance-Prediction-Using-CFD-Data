"""Regenerate all plots from saved models — no retraining, no internet."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import offline  # noqa: F401
from visualize import generate_plots_from_saved

if __name__ == "__main__":
    generate_plots_from_saved()
