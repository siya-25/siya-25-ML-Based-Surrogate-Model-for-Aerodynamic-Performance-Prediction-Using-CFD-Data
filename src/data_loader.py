import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import DATA_PATH, INPUT_FEATURES, OUTPUT_TARGETS, RANDOM_STATE


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[INPUT_FEATURES].values
    y = df[OUTPUT_TARGETS].values
    return X, y, df


def split_and_scale(X, y, test_size=0.2, val_size=0.15):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=RANDOM_STATE
    )
    relative_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - relative_val, random_state=RANDOM_STATE
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s = x_scaler.transform(X_val)
    X_test_s = x_scaler.transform(X_test)

    y_train_s = y_scaler.fit_transform(y_train)
    y_val_s = y_scaler.transform(y_val)
    y_test_s = y_scaler.transform(y_test)

    return {
        "X_train": X_train_s,
        "X_val": X_val_s,
        "X_test": X_test_s,
        "X_train_raw": X_train,
        "X_val_raw": X_val,
        "X_test_raw": X_test,
        "y_train": y_train_s,
        "y_val": y_val_s,
        "y_test": y_test_s,
        "y_train_raw": y_train,
        "y_val_raw": y_val,
        "y_test_raw": y_test,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
    }
