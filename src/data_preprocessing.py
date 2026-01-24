import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging
import joblib

from .utils import load_config, setup_logging
from .validation import validate_dataframe

# ======================
# THRESHOLDS
# ======================
PH_MIN, PH_MAX = 6.5, 8.0
TDS_MAX = 500
TURB_MAX = 5

ROLLING_WINDOW = 5
EPS = 1e-6


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for sensor validation & rule aggregation.
    NO label leakage.
    """

    # ----------------------
    # DISTANCE FEATURES
    # ----------------------
    df["ph_distance"] = np.where(
        df["ph"] < PH_MIN,
        PH_MIN - df["ph"],
        np.where(df["ph"] > PH_MAX, df["ph"] - PH_MAX, 0)
    )

    df["tds_distance"] = np.maximum(0, df["tds"] - TDS_MAX)
    df["turbidity_distance"] = np.maximum(0, df["turbidity"] - TURB_MAX)

    # ----------------------
    # RATIO FEATURES
    # ----------------------
    df["ph_tds_ratio"] = df["ph"] / (df["tds"] + EPS)
    df["turbidity_ph_ratio"] = df["turbidity"] / (df["ph"] + EPS)
    df["tds_turbidity_ratio"] = df["tds"] / (df["turbidity"] + EPS)

    # ----------------------
    # RAW ENGINEERED SIGNALS
    # ----------------------
    df["ph_raw"] = df["ph"] / 14.0
    df["tds_raw"] = np.log1p(df["tds"]) / np.log1p(30000)
    df["turbidity_raw"] = np.log1p(df["turbidity"]) / np.log1p(50)

    # ----------------------
    # NOISE-SENSITIVE METRICS
    # ----------------------
    df["ph_noise_score"] = np.abs(
        df["ph"] - df["ph"].rolling(ROLLING_WINDOW, min_periods=1).mean()
    )

    df["tds_noise_score"] = np.abs(
        df["tds"] - df["tds"].rolling(ROLLING_WINDOW, min_periods=1).mean()
    )

    df["turbidity_noise_score"] = np.abs(
        df["turbidity"] - df["turbidity"].rolling(ROLLING_WINDOW, min_periods=1).mean()
    )

    df["instability_score"] = (
        df["ph_noise_score"]
        + df["tds_noise_score"] / 1000
        + df["turbidity_noise_score"]
    )

    return df


def preprocess_data(config):
    data_config = config["data"]
    raw_path = data_config["raw_path"]
    processed_train_path = data_config["train_processed_path"]
    processed_test_path = data_config["test_processed_path"]
    test_size = data_config["test_size"]
    random_state = data_config["random_state"]
    missing_strategy = data_config.get("missing_value_strategy", {})

    # Ensure output directories exist
    Path(processed_train_path).parent.mkdir(parents=True, exist_ok=True)
    Path(processed_test_path).parent.mkdir(parents=True, exist_ok=True)

    logging.info("Loading raw dataset...")
    df = pd.read_csv(raw_path)
    logging.info(f"Loaded dataset with shape: {df.shape}")

    # Validate schema & sensor bounds
    df = validate_dataframe(df, config)

    # ----------------------
    # MISSING VALUES
    # ----------------------
    for col, strategy in missing_strategy.items():
        if col not in df.columns:
            continue

        if strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median":
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mode":
            df[col] = df[col].fillna(df[col].mode()[0])
        elif strategy == "zero":
            df[col] = df[col].fillna(0)
        elif strategy == "drop":
            df = df.dropna(subset=[col])

        logging.info(f"Applied missing value strategy '{strategy}' to column '{col}'")

    # ----------------------
    # FEATURE ENGINEERING
    # ----------------------
    df = engineer_features(df)

    feature_cols = [
        # Distance
        "ph_distance", "tds_distance", "turbidity_distance",

        # Ratios
        "ph_tds_ratio", "turbidity_ph_ratio", "tds_turbidity_ratio",

        # Raw engineered
        "ph_raw", "tds_raw", "turbidity_raw",

        # Noise-sensitive
        "ph_noise_score", "tds_noise_score",
        "turbidity_noise_score", "instability_score"
    ]

    label_col = data_config["label_column"]

    X = df[feature_cols]
    y = df[label_col]

    # ----------------------
    # LABEL ENCODING
    # ----------------------
    label_encoder_path = config["model"].get("label_encoder_path")
    if y.dtype == "object" or y.dtype.name == "category":
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
        if label_encoder_path:
            joblib.dump(encoder, label_encoder_path)
            logging.info(f"Saved label encoder to {label_encoder_path}")

    # ----------------------
    # TRAIN / TEST SPLIT
    # ----------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # ----------------------
    # SAVE OUTPUT
    # ----------------------
    train_df = pd.concat([X_train, pd.Series(y_train, name=label_col)], axis=1)
    test_df = pd.concat([X_test, pd.Series(y_test, name=label_col)], axis=1)

    train_df.to_csv(processed_train_path, index=False)
    test_df.to_csv(processed_test_path, index=False)

    logging.info(f"Saved training data to: {processed_train_path}")
    logging.info(f"Saved testing data to: {processed_test_path}")
    logging.info("Data preprocessing completed successfully!")

    return processed_train_path, processed_test_path


if __name__ == "__main__":
    config = load_config()
    setup_logging(
        config["logging"]["preprocessing_log"],
        level=config["logging"].get("level", "INFO")
    )
    preprocess_data(config)
