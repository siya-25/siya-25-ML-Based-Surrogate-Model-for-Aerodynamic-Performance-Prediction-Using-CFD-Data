import offline  # noqa: F401 — enforce offline settings before matplotlib

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from config import (
    INPUT_FEATURES,
    MODELS_DIR,
    OUTPUT_TARGETS,
    PLOTS_DIR,
    RESULTS_DIR,
)
from data_loader import load_data, split_and_scale
from model_io import load_model

TARGET_LABELS = {
    "CL": "Lift Coefficient (CL)",
    "CD": "Drag Coefficient (CD)",
    "CM": "Moment Coefficient (CM)",
}


def _save_fig(fig, filename):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path.name}")
    return out_path


def model_to_filename(name):
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


def predict_artifact(artifact, X_raw):
    X_scaled = artifact["x_scaler"].transform(X_raw)
    y_scaled = artifact["model"].predict(X_scaled)
    return artifact["y_scaler"].inverse_transform(y_scaled)


def _envelope_grid(altitude_m, n=50):
    mach_grid = np.linspace(0.1, 0.9, n)
    aoa_grid = np.linspace(-5, 20, n)
    mach_mesh, aoa_mesh = np.meshgrid(mach_grid, aoa_grid)
    n_points = mach_mesh.size
    X_grid = np.column_stack(
        [mach_mesh.ravel(), aoa_mesh.ravel(), np.full(n_points, altitude_m)]
    )
    return mach_mesh, aoa_mesh, X_grid


def plot_envelope_with_data(model_artifact, df, altitude_m=5000, tolerance_m=1500):
    mach_mesh, aoa_mesh, X_grid = _envelope_grid(altitude_m)
    y_pred = predict_artifact(model_artifact, X_grid)

    mask = np.abs(df["Altitude_m"].values - altitude_m) <= tolerance_m
    df_slice = df[mask]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, idx, target in zip(axes, range(3), OUTPUT_TARGETS):
        Z = y_pred[:, idx].reshape(mach_mesh.shape)
        cf = ax.contourf(mach_mesh, aoa_mesh, Z, levels=20, cmap="viridis", alpha=0.85)
        ax.contour(mach_mesh, aoa_mesh, Z, levels=8, colors="k", linewidths=0.3, alpha=0.35)
        if len(df_slice):
            sc = ax.scatter(
                df_slice["Mach"],
                df_slice["AoA_deg"],
                c=df_slice[target],
                s=80,
                edgecolors="white",
                linewidths=1.2,
                cmap="plasma",
                zorder=5,
            )
            plt.colorbar(sc, ax=ax, label=f"CFD {target}")
        plt.colorbar(cf, ax=ax, label=f"Predicted {target}")
        ax.set_xlabel("Mach")
        ax.set_ylabel("AoA (deg)")
        ax.set_title(TARGET_LABELS[target])

    fig.suptitle(
        f"Predicted Envelope + CFD Data (altitude ~ {altitude_m} m, ±{tolerance_m} m)",
        fontsize=12,
    )
    fig.tight_layout()
    return _save_fig(fig, f"envelope_with_data_alt_{altitude_m}m.png")


def plot_3d_surfaces(model_artifact, altitude_m=5000):
    mach_mesh, aoa_mesh, X_grid = _envelope_grid(altitude_m, n=35)
    y_pred = predict_artifact(model_artifact, X_grid)

    fig = plt.figure(figsize=(16, 4.5))
    for idx, target in enumerate(OUTPUT_TARGETS):
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
        Z = y_pred[:, idx].reshape(mach_mesh.shape)
        surf = ax.plot_surface(
            mach_mesh, aoa_mesh, Z, cmap="viridis", edgecolor="none", alpha=0.9
        )
        fig.colorbar(surf, ax=ax, shrink=0.6, label=target)
        ax.set_xlabel("Mach")
        ax.set_ylabel("AoA (deg)")
        ax.set_zlabel(target)
        ax.set_title(TARGET_LABELS[target])

    fig.suptitle(f"3D Aerodynamic Surfaces at Altitude = {altitude_m} m", fontsize=13)
    fig.tight_layout()
    return _save_fig(fig, f"3d_surface_alt_{altitude_m}m.png")


def plot_predicted_vs_actual(y_true, y_pred, model_name):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, idx, target in zip(axes, range(3), OUTPUT_TARGETS):
        ax.scatter(y_true[:, idx], y_pred[:, idx], alpha=0.85, s=60, edgecolors="k", linewidths=0.4)
        lo = min(y_true[:, idx].min(), y_pred[:, idx].min())
        hi = max(y_true[:, idx].max(), y_pred[:, idx].max())
        pad = (hi - lo) * 0.08 or 0.01
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r--", linewidth=1.2, label="Ideal")
        ax.set_xlabel(f"Actual {target}")
        ax.set_ylabel(f"Predicted {target}")
        ax.set_title(TARGET_LABELS[target])
        ax.legend(loc="upper left", fontsize=8)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(f"Predicted vs Actual — {model_name} (validation + test)", fontsize=13)
    fig.tight_layout()
    safe = model_to_filename(model_name)
    return _save_fig(fig, f"predicted_vs_actual_{safe}.png")


def plot_residuals(y_true, y_pred, model_name):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, idx, target in zip(axes, range(3), OUTPUT_TARGETS):
        residuals = y_true[:, idx] - y_pred[:, idx]
        ax.scatter(y_pred[:, idx], residuals, alpha=0.85, s=60, edgecolors="k", linewidths=0.4)
        ax.axhline(0, color="r", linestyle="--", linewidth=1.2)
        ax.set_xlabel(f"Predicted {target}")
        ax.set_ylabel(f"Residual ({target})")
        ax.set_title(TARGET_LABELS[target])

    fig.suptitle(f"Residual Plot — {model_name}", fontsize=13)
    fig.tight_layout()
    safe = model_to_filename(model_name)
    return _save_fig(fig, f"residuals_{safe}.png")


def plot_error_histogram(y_true, y_pred, model_name):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, idx, target in zip(axes, range(3), OUTPUT_TARGETS):
        errors = np.abs(y_true[:, idx] - y_pred[:, idx])
        ax.hist(errors, bins=8, color="#4C72B0", edgecolor="white", alpha=0.9)
        ax.set_xlabel(f"Absolute error in {target}")
        ax.set_ylabel("Count")
        ax.set_title(TARGET_LABELS[target])

    fig.suptitle(f"Prediction Error Distribution — {model_name}", fontsize=13)
    fig.tight_layout()
    safe = model_to_filename(model_name)
    return _save_fig(fig, f"error_histogram_{safe}.png")


def plot_input_space(df):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    pairs = [
        ("Mach", "AoA_deg", "CL"),
        ("Mach", "Altitude_m", "CL"),
        ("AoA_deg", "Altitude_m", "CD"),
        ("Mach", "AoA_deg", "CM"),
    ]
    for ax, (x_col, y_col, c_col) in zip(axes.ravel(), pairs):
        sc = ax.scatter(
            df[x_col], df[y_col], c=df[c_col], s=90, cmap="coolwarm", edgecolors="k", linewidths=0.4
        )
        plt.colorbar(sc, ax=ax, label=c_col)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{c_col} over {x_col} vs {y_col}")

    fig.suptitle("CFD Training Data — Input Space Exploration", fontsize=13)
    fig.tight_layout()
    return _save_fig(fig, "input_space_scatter.png")


def plot_output_correlation(df):
    corr = df[OUTPUT_TARGETS].corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(OUTPUT_TARGETS)
    ax.set_yticklabels(OUTPUT_TARGETS)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", color="black")
    plt.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Correlation Between Aerodynamic Coefficients")
    fig.tight_layout()
    return _save_fig(fig, "output_correlation.png")


def save_comparison_table(summary):
    rows = []
    for entry in summary:
        val = entry["validation"]["overall"]
        test = entry["test"]["overall"]
        rows.append(
            {
                "Model": entry["model"],
                "Val_RMSE": val["RMSE"],
                "Val_MAE": val["MAE"],
                "Val_R2": val["R2"],
                "Test_RMSE": test["RMSE"],
                "Test_MAE": test["MAE"],
                "Test_R2": test["R2"],
                "CV_RMSE_scaled": entry["cv_rmse_scaled"],
            }
        )

    df = pd.DataFrame(rows).sort_values("Val_RMSE").reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "model_comparison_table.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"Comparison table saved: {csv_path}")
    return df


def plot_model_comparison(summary):
    df = save_comparison_table(summary)

    models = df["Model"].tolist()
    val_rmse = df["Val_RMSE"].tolist()
    test_rmse = df["Test_RMSE"].tolist()
    val_r2 = df["Val_R2"].tolist()

    y_pos = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(models) * 0.45)))

    ax_rmse = axes[0]
    bar_h = 0.35
    ax_rmse.barh(y_pos - bar_h / 2, val_rmse, bar_h, label="Validation", color="#4C72B0")
    ax_rmse.barh(y_pos + bar_h / 2, test_rmse, bar_h, label="Test", color="#DD8452")
    ax_rmse.set_yticks(y_pos)
    ax_rmse.set_yticklabels(models, fontsize=9)
    ax_rmse.invert_yaxis()
    ax_rmse.set_xlabel("RMSE (lower is better)")
    ax_rmse.set_title("Model Comparison — RMSE")
    ax_rmse.legend(loc="lower right")

    ax_r2 = axes[1]
    colors = plt.cm.RdYlGn(np.clip(val_r2, 0, 1))
    ax_r2.barh(y_pos, val_r2, color=colors)
    ax_r2.set_yticks(y_pos)
    ax_r2.set_yticklabels(models, fontsize=9)
    ax_r2.invert_yaxis()
    ax_r2.set_xlim(0, 1.05)
    ax_r2.set_xlabel("R² score (higher is better)")
    ax_r2.set_title("Model Comparison — Validation R²")
    ax_r2.axvline(x=0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    fig.suptitle(f"Aerodynamic Surrogate Model Comparison ({len(models)} models)", fontsize=13, y=1.01)
    fig.tight_layout()
    return _save_fig(fig, "model_comparison.png")


def plot_metrics_heatmap(summary):
    ranked = sorted(summary, key=lambda x: x["validation"]["overall"]["RMSE"])
    models = [e["model"] for e in ranked]

    rmse_matrix = np.array(
        [[e["validation"][t]["RMSE"] for t in OUTPUT_TARGETS] for e in ranked]
    )
    r2_matrix = np.array(
        [[e["validation"][t]["R2"] for t in OUTPUT_TARGETS] for e in ranked]
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, max(5, len(models) * 0.35)))

    for ax, matrix, title, cmap, fmt in [
        (axes[0], rmse_matrix, "Validation RMSE by Target", "YlOrRd", ".4f"),
        (axes[1], r2_matrix, "Validation R² by Target", "RdYlGn", ".3f"),
    ]:
        im = ax.imshow(matrix, aspect="auto", cmap=cmap)
        ax.set_xticks(range(3))
        ax.set_xticklabels(OUTPUT_TARGETS)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=8)
        for i in range(len(models)):
            for j in range(3):
                ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", fontsize=7)
        plt.colorbar(im, ax=ax)
        ax.set_title(title)

    fig.suptitle("Per-Coefficient Model Performance Heatmap", fontsize=13)
    fig.tight_layout()
    return _save_fig(fig, "metrics_heatmap.png")


def plot_per_target_bars(summary, top_n=8):
    ranked = sorted(summary, key=lambda x: x["validation"]["overall"]["RMSE"])[:top_n]
    models = [e["model"] for e in ranked]
    x = np.arange(len(models))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, ylabel, title in [
        (axes[0], "RMSE", "RMSE (lower is better)", "RMSE by Aerodynamic Coefficient"),
        (axes[1], "R2", "R² (higher is better)", "R² by Aerodynamic Coefficient"),
    ]:
        for i, target in enumerate(OUTPUT_TARGETS):
            vals = [e["validation"][target][metric] for e in ranked]
            offset = (i - 1) * width
            ax.bar(x + offset, vals, width, label=target)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    fig.suptitle(f"Top {top_n} Models — Per-Target Validation Metrics", fontsize=13)
    fig.tight_layout()
    return _save_fig(fig, "per_target_metrics.png")


def generate_all_visualizations(best_name, best_artifact, summary, data, df):
    print("\nGenerating visualizations...")

    X_eval = np.vstack([data["X_val_raw"], data["X_test_raw"]])
    y_eval = np.vstack([data["y_val_raw"], data["y_test_raw"]])
    y_pred = predict_artifact(best_artifact, X_eval)

    plot_predicted_vs_actual(y_eval, y_pred, best_name)
    plot_residuals(y_eval, y_pred, best_name)
    plot_error_histogram(y_eval, y_pred, best_name)
    plot_metrics_heatmap(summary)
    plot_per_target_bars(summary)
    plot_input_space(df)
    plot_output_correlation(df)

    for alt in [0, 5000, 10000]:
        plot_envelope_with_data(best_artifact, df, altitude_m=alt)
    plot_3d_surfaces(best_artifact, altitude_m=5000)

    print(f"All plots saved to: {PLOTS_DIR}")


def generate_plots_from_saved():
    """Regenerate plots from saved results without retraining."""
    results_path = RESULTS_DIR / "model_comparison.json"
    if not results_path.exists():
        raise FileNotFoundError("Run main.py first to train models and save results.")

    with open(results_path, encoding="utf-8") as f:
        summary = json.load(f)

    best_artifact = load_model(MODELS_DIR / "best_model.pkl")
    best_name = min(summary, key=lambda x: x["validation"]["overall"]["RMSE"])["model"]

    X, y, df = load_data()
    data = split_and_scale(X, y)

    plot_model_comparison(summary)
    generate_all_visualizations(best_name, best_artifact, summary, data, df)
