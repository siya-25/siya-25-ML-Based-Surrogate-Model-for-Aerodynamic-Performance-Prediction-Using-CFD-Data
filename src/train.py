import json
from pathlib import Path

import numpy as np

from config import MODELS_DIR, OUTPUT_TARGETS, PLOTS_DIR, RESULTS_DIR
from data_loader import load_data, split_and_scale
from metrics import evaluate_predictions, format_metrics
from model_io import save_model
from models import build_models, train_with_tuning
from visualize import plot_model_comparison


def run_training():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, df = load_data()
    data = split_and_scale(X, y)

    summary = []
    best_model_name = None
    best_val_rmse = float("inf")
    best_artifact = None

    print("=" * 60)
    print("Aerodynamic Surrogate Model Training")
    print("=" * 60)

    for name, spec in build_models().items():
        print(f"\nTraining: {name}")
        model, best_params, cv_rmse = train_with_tuning(
            name, spec, data["X_train"], data["y_train"]
        )
        print(f"  Best CV RMSE (scaled): {cv_rmse:.5f}")
        print(f"  Best params: {best_params}")

        y_val_pred_s = model.predict(data["X_val"])
        y_val_pred = data["y_scaler"].inverse_transform(y_val_pred_s)
        y_val_raw = data["y_scaler"].inverse_transform(data["y_val"])

        val_metrics = evaluate_predictions(y_val_raw, y_val_pred, OUTPUT_TARGETS)
        print("  Validation metrics:")
        print(format_metrics(val_metrics))

        y_test_pred_s = model.predict(data["X_test"])
        y_test_pred = data["y_scaler"].inverse_transform(y_test_pred_s)
        test_metrics = evaluate_predictions(data["y_test_raw"], y_test_pred, OUTPUT_TARGETS)

        artifact = {
            "model": model,
            "x_scaler": data["x_scaler"],
            "y_scaler": data["y_scaler"],
            "best_params": best_params,
        }

        entry = {
            "model": name,
            "best_params": best_params,
            "cv_rmse_scaled": cv_rmse,
            "validation": val_metrics,
            "test": test_metrics,
        }
        summary.append(entry)

        val_rmse = val_metrics["overall"]["RMSE"]
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_name = name
            best_artifact = artifact

    save_model(best_artifact, MODELS_DIR / "best_model.pkl")

    results_path = RESULTS_DIR / "model_comparison.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nGenerating model comparison chart...")
    plot_model_comparison(summary)

    print("\n" + "=" * 60)
    print(f"Best model (lowest validation RMSE): {best_model_name}")
    print(f"Results saved to: {results_path}")
    print("=" * 60)

    return best_model_name, best_artifact, summary, data, df


if __name__ == "__main__":
    run_training()
