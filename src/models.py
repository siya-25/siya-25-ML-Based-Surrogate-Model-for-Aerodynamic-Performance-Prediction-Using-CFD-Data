from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from config import RANDOM_STATE


def build_models():
    gpr_kernel = (
        ConstantKernel(1.0) * Matern(nu=2.5, length_scale=1.0)
        + WhiteKernel(noise_level=1e-3)
    )

    return {
        # --- Surrogate / probabilistic ---
        "Gaussian Process (Kriging)": {
            "estimator": MultiOutputRegressor(
                GaussianProcessRegressor(
                    kernel=gpr_kernel,
                    normalize_y=True,
                    random_state=RANDOM_STATE,
                    n_restarts_optimizer=5,
                )
            ),
            "param_grid": {
                "estimator__kernel__k1__k2__length_scale": [0.5, 1.0, 2.0],
                "estimator__alpha": [1e-10, 1e-8, 1e-6],
            },
        },
        # --- Tree-based ensembles ---
        "Decision Tree": {
            "estimator": DecisionTreeRegressor(random_state=RANDOM_STATE),
            "param_grid": {
                "max_depth": [3, 5, 8, None],
                "min_samples_leaf": [1, 2, 4],
            },
        },
        "Random Forest": {
            "estimator": RandomForestRegressor(random_state=RANDOM_STATE),
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, None],
                "min_samples_leaf": [1, 2],
            },
        },
        "Extra Trees": {
            "estimator": ExtraTreesRegressor(random_state=RANDOM_STATE),
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, None],
                "min_samples_leaf": [1, 2],
            },
        },
        "Gradient Boosting": {
            "estimator": MultiOutputRegressor(
                GradientBoostingRegressor(random_state=RANDOM_STATE)
            ),
            "param_grid": {
                "estimator__n_estimators": [50, 100],
                "estimator__max_depth": [3, 5],
                "estimator__learning_rate": [0.05, 0.1],
            },
        },
        "HistGradient Boosting": {
            "estimator": MultiOutputRegressor(
                HistGradientBoostingRegressor(random_state=RANDOM_STATE, max_iter=200)
            ),
            "param_grid": {
                "estimator__max_depth": [3, 5, None],
                "estimator__learning_rate": [0.05, 0.1],
                "estimator__l2_regularization": [0.0, 0.1],
            },
        },
        "AdaBoost": {
            "estimator": MultiOutputRegressor(
                AdaBoostRegressor(random_state=RANDOM_STATE)
            ),
            "param_grid": {
                "estimator__n_estimators": [50, 100, 200],
                "estimator__learning_rate": [0.05, 0.1, 1.0],
            },
        },
        "XGBoost": {
            "estimator": XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                verbosity=0,
            ),
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1, 0.2],
            },
        },
        # --- Instance-based & kernel methods ---
        "k-Nearest Neighbors": {
            "estimator": KNeighborsRegressor(),
            "param_grid": {
                "n_neighbors": [2, 3, 4, 5],
                "weights": ["uniform", "distance"],
                "p": [1, 2],
            },
        },
        "Support Vector Regression": {
            "estimator": MultiOutputRegressor(SVR()),
            "param_grid": {
                "estimator__C": [0.1, 1.0, 10.0],
                "estimator__epsilon": [0.01, 0.1],
                "estimator__kernel": ["rbf", "linear"],
            },
        },
        # --- Linear models ---
        "Ridge Regression": {
            "estimator": Ridge(random_state=RANDOM_STATE),
            "param_grid": {
                "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            },
        },
        "Elastic Net": {
            "estimator": ElasticNet(random_state=RANDOM_STATE, max_iter=5000),
            "param_grid": {
                "alpha": [0.001, 0.01, 0.1, 1.0],
                "l1_ratio": [0.2, 0.5, 0.8],
            },
        },
        # --- Neural network ---
        "Artificial Neural Network": {
            "estimator": MLPRegressor(
                max_iter=5000,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=RANDOM_STATE,
            ),
            "param_grid": {
                "hidden_layer_sizes": [(32,), (64, 32), (64, 32, 16)],
                "alpha": [1e-4, 1e-3, 1e-2],
                "learning_rate_init": [0.001, 0.01],
            },
        },
    }


def train_with_tuning(name, spec, X_train, y_train):
    search = GridSearchCV(
        spec["estimator"],
        spec["param_grid"],
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, -search.best_score_
