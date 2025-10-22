import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging
import yaml
import joblib


def load_config(path="src/config/config.yaml"):
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(log_path, level="INFO"):
    """Set up logging configuration."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def preprocess_data(config):
    """Reads raw dataset, cleans it, encodes labels, splits into train/test, and saves processed files."""
    
    # Load parameters from config
    raw_path = config["data"]["raw_path"]
    processed_dir = config["data"]["processed_path"]
    required_columns = [col.lower() for col in config["data"]["required_columns"]]
    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]
    missing_strategy = config["data"].get("missing_value_strategy", "mean")

    os.makedirs(processed_dir, exist_ok=True)

    logging.info("Loading raw dataset...")
    df = pd.read_csv(raw_path)
    logging.info(f"Loaded dataset with shape: {df.shape}")

    df.columns = [col.strip().lower() for col in df.columns]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    logging.info("All required columns found.")

    # Basic Cleaning
    df = df.drop_duplicates()


    if missing_strategy == "drop":
        df = df.dropna()
        logging.info("Dropped rows with missing values.")
    elif missing_strategy == "mean":
        df = df.fillna(df.mean(numeric_only=True))
        logging.info("Filled missing values with column means.")
    elif missing_strategy == "median":
        df = df.fillna(df.median(numeric_only=True))
        logging.info("Filled missing values with column medians.")
    elif missing_strategy == "zero":
        df = df.fillna(0)
        logging.info("Replaced missing values with zeros.")


    feature_cols = [c for c in required_columns if c != "potability"]
    X = df[feature_cols]
    y = df["potability"]


    label_encoder_path = config["model"].get("label_encoder_path", os.path.join(processed_dir, "label_encoder.pkl"))
    if y.dtype == "object":
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
        joblib.dump(encoder, label_encoder_path)
        logging.info(f"Saved label encoder to {label_encoder_path}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")


    train_path = os.path.join(processed_dir, "water_quality_processed_train.csv")
    test_path = os.path.join(processed_dir, "water_quality_processed_test.csv")

    train_df = pd.concat([X_train, pd.Series(y_train, name="potability")], axis=1)
    test_df = pd.concat([X_test, pd.Series(y_test, name="potability")], axis=1)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logging.info(f"Saved training data to: {train_path}")
    logging.info(f"Saved testing data to: {test_path}")
    logging.info("Data preprocessing completed successfully!")

    return train_path, test_path


if __name__ == "__main__":

    config = load_config()


    setup_logging(
        config["logging"]["preprocessing_log"],
        level=config["logging"].get("level", "INFO")
    )


    preprocess_data(config)
