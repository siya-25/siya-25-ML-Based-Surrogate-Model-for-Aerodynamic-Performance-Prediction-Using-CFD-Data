import pickle
from pathlib import Path


def save_model(artifact, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(artifact, f)


def load_model(path):
    with open(Path(path), "rb") as f:
        return pickle.load(f)
