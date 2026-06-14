import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_predictions(y_true, y_pred, target_names):
    results = {}
    for i, name in enumerate(target_names):
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}

    results["overall"] = {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred, multioutput="uniform_average"),
    }
    return results


def format_metrics(metrics):
    lines = []
    for target, values in metrics.items():
        if target == "overall":
            continue
        lines.append(
            f"  {target}: RMSE={values['RMSE']:.5f}, "
            f"MAE={values['MAE']:.5f}, R²={values['R2']:.4f}"
        )
    overall = metrics["overall"]
    lines.append(
        f"  Overall: RMSE={overall['RMSE']:.5f}, "
        f"MAE={overall['MAE']:.5f}, R²={overall['R2']:.4f}"
    )
    return "\n".join(lines)
