from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "aero_data.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = RESULTS_DIR / "plots"

INPUT_FEATURES = ["Mach", "AoA_deg", "Altitude_m"]
OUTPUT_TARGETS = ["CL", "CD", "CM"]
RANDOM_STATE = 42
