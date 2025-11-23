import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging
import joblib

from utils import load_config, setup_logging
from validation import validate_dataframe


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features based on water treatment thresholds and ratios.
    """
    # Distance from safe range
    df['ph_distance'] = np.where(
        df['ph'] < 6.5,
        6.5 - df['ph'],
        np.where(df['ph'] > 8.0, df['ph'] - 8.0, 0)
    )

    # tds distance above threshold 1500
    df['tds_distance'] = np.where(df['tds'] > 1500, df['tds'] - 1500, 0)

    # turbidity distance above threshold 5
    df['turbidity_distance'] = np.where(df['turbidity'] > 5, df['turbidity'] - 5, 0)

    # Binary compliance flags
    df['ph_compliant'] = ((df['ph'] >= 6.5) & (df['ph'] <= 8.0)).astype(int)
    df['tds_compliant'] = (df['tds'] < 1500).astype(int)
    df['turbidity_compliant'] = (df['turbidity'] < 5).astype(int)

    # Aggregate scores
    df['compliance_score'] = df['ph_compliant'] + df['tds_compliant'] + df['turbidity_compliant']
    df['weighted_score'] = 2*df['ph_compliant'] + df['tds_compliant'] + 2*df['turbidity_compliant']

    # Ratios / interaction features
    df['ph_tds_ratio'] = df['ph'] / (df['tds'] + 1)
    df['turbidity_ph_ratio'] = df['turbidity'] / (df['ph'] + 1)
    df['tds_turbidity_ratio'] = df['tds'] / (df['turbidity'] + 1)

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

    # Validate dataframe
    df = validate_dataframe(df, config)

    # Handle missing values per-column
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

    # Feature engineering
    df = engineer_features(df)

    feature_cols = [
        'ph_distance', 'tds_distance', 'turbidity_distance',
        'ph_compliant', 'tds_compliant', 'turbidity_compliant',
        'compliance_score', 'weighted_score',
        'ph_tds_ratio', 'turbidity_ph_ratio', 'tds_turbidity_ratio'
    ]
    label_col = data_config["label_column"]

    X = df[feature_cols]
    y = df[label_col]

    # Encode labels if needed
    label_encoder_path = config["model"].get("label_encoder_path")
    if y.dtype == "object" or y.dtype.name == "category":
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
        if label_encoder_path:
            joblib.dump(encoder, label_encoder_path)
            logging.info(f"Saved label encoder to {label_encoder_path}")

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Save processed CSVs
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
