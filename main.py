import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import offline  # noqa: F401 — enforce offline settings before other imports

from train import run_training
from visualize import generate_all_visualizations


def main():
    best_name, best_artifact, summary, data, df = run_training()
    generate_all_visualizations(best_name, best_artifact, summary, data, df)

    print("\nModel ranking (validation RMSE):")
    ranked = sorted(summary, key=lambda x: x["validation"]["overall"]["RMSE"])
    for i, entry in enumerate(ranked, 1):
        rmse = entry["validation"]["overall"]["RMSE"]
        r2 = entry["validation"]["overall"]["R2"]
        print(f"  {i}. {entry['model']}: RMSE={rmse:.5f}, R²={r2:.4f}")


if __name__ == "__main__":
    main()
