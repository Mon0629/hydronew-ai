import yaml
import logging
from pathlib import Path
from typing import Any, Dict
import os

# --- Logging setup ---
def setup_logging(log_file: str = "logs/app.log", level: str = "INFO") -> None:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler())

# --- Config loader ---
def load_config(config_path: str = "src/config/config.yaml") -> Dict[str, Any]:

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    required_sections = ["data", "model", "metrics", "logging"]
    missing_sections = [s for s in required_sections if s not in config]
    if missing_sections:
        raise ValueError(f"Missing top-level config sections: {missing_sections}")

    return config

# --- Validation helper ---
def validate_columns(data: Dict[str, Any], required_columns: list) -> None:

    missing = [col for col in required_columns if col not in data.keys()]
    if missing:
        logging.warning(f"Missing required columns: {missing}")
    else:
        logging.info("All required columns are present.")

# model cleanup
def model_cleanup(model_dir: str, base_name: str = "random_forest", keep_last_n: int = 5) -> None:

    os.makedirs(model_dir, exist_ok=True)

    models = [f for f in os.listdir(model_dir) if f.startswith(base_name + "_") and f.endswith(".pkl")]

    def extract_date(filename: str) -> int:
        try:
            date_part = filename.replace(f"{base_name}_", "").replace(".pkl", "")
            return int(date_part)
        except ValueError:
            return 0
        
    models_sorted = sorted(models, key=extract_date, reverse=True)

    for old_model in models_sorted[keep_last_n:]:
        old_path = os.path.join(model_dir, old_model)
        try:
            os.remove(old_path)
            logging.info(f"Deleted old model: {old_path}")
        except Exception as e:
            logging.warning(f"Failed to delete {old_path}: {e}")

    logging.info(f"Model cleanup completed. Retained last {keep_last_n} models.")


if __name__ == "__main__":
    setup_logging("logs/utils.log")
    config = load_config()
    logging.info(f"Loaded config from src/config/config.yaml")

    required_cols = config["data"].get("required_columns", [])

    sample_data = {"ph": 7, "turbidity": 50, "tds": 300, "status": "Safe"}
    validate_columns(sample_data, required_cols)
