import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from utils import load_config, setup_logging


config = load_config()
log_file = config["logging"].get("validation_log", "logs/validation.log")
setup_logging(log_file, config["logging"].get("level", "INFO"))

def validate_dataframe(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Validate a dataframe based on config rules.
    Invalid rows are logged and saved to invalid_records_path.
    Returns a cleaned dataframe.
    """
    data_config = config["data"]
    invalid_records = []


    for col in data_config["required_columns"]:
        if col not in df.columns:
            logging.error(f"Missing required column: {col}")
            raise KeyError(f"Missing required column: {col}")

    for col, rules in data_config.get("validation", {}).items():
        if col not in df.columns:
            continue


        if "min" in rules:
            mask = df[col] < rules["min"]
            if mask.any():
                logging.warning(f"{mask.sum()} rows in '{col}' below min ({rules['min']})")
                invalid_records.append(df[mask])
                df = df[~mask]

        if "max" in rules:
            mask = df[col] > rules["max"]
            if mask.any():
                logging.warning(f"{mask.sum()} rows in '{col}' above max ({rules['max']})")
                invalid_records.append(df[mask])
                df = df[~mask]


        if "allowed_values" in rules:
            mask = ~df[col].isin(rules["allowed_values"])
            if mask.any():
                logging.warning(f"{mask.sum()} rows in '{col}' not in allowed values")
                invalid_records.append(df[mask])
                df = df[~mask]


        if col == "timestamp" and "format" in rules:
            valid_ts = pd.to_datetime(df[col], format=rules["format"], errors='coerce')
            mask = valid_ts.isna()
            if mask.any():
                logging.warning(f"{mask.sum()} rows have invalid timestamp format")
                invalid_records.append(df[mask])
                df = df[~mask]


    if invalid_records:
        invalid_df = pd.concat(invalid_records)
        invalid_path = Path(data_config.get("invalid_records_path", "data/invalid/"))
        invalid_path.mkdir(parents=True, exist_ok=True)
        invalid_file = invalid_path / f"invalid_records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        invalid_df.to_csv(invalid_file, index=False)
        logging.info(f"Saved {len(invalid_df)} invalid records to {invalid_file}")

    logging.info(f"Validation complete. {len(df)} valid rows remain.")
    return df

if __name__ == "__main__":
    df = pd.read_csv(config["data"]["raw_path"])
    df_valid = validate_dataframe(df, config)
