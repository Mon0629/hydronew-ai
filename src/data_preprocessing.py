import os
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import yaml

def load_config(path="src/config/config.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

# Logging
def setup_logging(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def preprocess_data(config):
    """Reads raw data, cleans it, and saves processed data"""

    raw_path = config["data"]["raw_path"]
    processed_path = config["data"]["processed_path"]
    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]

    # Make sure folders exist
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)

    logging.info("🔍 Loading raw dataset...")
    df = pd.read_csv(raw_path)

    logging.info(f"Loaded data with shape: {df.shape}")

    # Basic Cleaning
    df = df.dropna()

    df.columns = [col.strip().lower() for col in df.columns]


    required_columns = ["ph", "turbidity", "tds", "label"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col} in dataset")

    X = df[["ph", "turbidity", "tds"]]
    y = df["label"]

    # Encode labels if they are strings
    if y.dtype == "object":
        y = y.astype("category").cat.codes

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


    # Save Processed Data
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    processed_dir = os.path.dirname(processed_path)
    train_path = os.path.join(processed_dir, "train.csv")
    test_path = os.path.join(processed_dir, "test.csv")

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    logging.info(f"Training data saved to: {train_path}")
    logging.info(f"Testing data saved to: {test_path}")
    logging.info("Data preprocessing completed successfully!")


if __name__ == "__main__":
    config = load_config()
    setup_logging(config["logging"]["training_log"])
    preprocess_data(config)