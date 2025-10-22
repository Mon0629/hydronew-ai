import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import yaml
import os
import logging
import json
import re


#load config
def load_config(path="src/config/config.yaml"):
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
    

def setup_logging(log_path: str, level="INFO"):
    """Set up logging configuration."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode='w'
    )

def get_next_model_version(base_dir, base_name="random_forest"):
    """
    Automatically find the next model version number.
    E.g., if v1_random_forest.pkl and v2_random_forest.pkl exist,
    returns path for v3_random_forest.pkl
    """
    os.makedirs(base_dir, exist_ok=True)
    existing_files = os.listdir(base_dir)

    version_numbers = []
    pattern = re.compile(r"v(\d+)_{}\.pkl".format(base_name))
    for filename in existing_files:
        match = pattern.match(filename)
        if match:
            version_numbers.append(int(match.group(1)))

    next_version = max(version_numbers, default=0) + 1
    next_model_filename = f"v{next_version}_{base_name}.pkl"
    return os.path.join(base_dir, next_model_filename), next_version

def train_model(config):
    """Train a RandomForest model on the processed training data and evaluate on test data."""
    
    train_path = config["data"]["train_processed_path"]
    test_path = config["data"]["test_processed_path"]
    model_path = config["model"]["model_path"]
    metrics_log_path = config["metrics"]["log_path"]

    (config['logging']['training_log'], config['logging'].get('level', 'INFO'))

    logging.info("Loading preprocessed training data...")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    x_train = train_data.drop("potability", axis=1)
    y_train = train_data["potability"]
    x_test = test_data.drop("potability", axis=1)
    y_test = test_data["potability"]

    model_params = config["model"]["params"]
    model_type = config["model"]["type"].strip().lower()

    if model_type in ["random_forest", "randomforest"]:
        model = RandomForestClassifier(**model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    logging.info("Training the model...")
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    report = classification_report(y_test, y_predict, output_dict=True)
    logging.info(f"Model Accuracy: {accuracy:.4f}")
    logging.info(f"Classification Report: {report}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}")

    metrics_path = config["metrics"]["log_path"]

    base_dir = os.path.dirname(model_path)
    base_name = os.path.basename(model_path).replace(".pkl", "").split("_", 1)[-1]
    versioned_model_path, version = get_next_model_version(base_dir, base_name)

    # Save model with version number
    joblib.dump(model, versioned_model_path)
    logging.info(f"Model saved to {versioned_model_path}")

    # Save versioned metrics
    metrics_versioned_path = os.path.join(base_dir, f"v{version}_metrics.json")
    with open(metrics_versioned_path, "w") as f:
        json.dump({
            "version": version,
            "accuracy": accuracy,
            "classification_report": report
        }, f, indent=4)
    logging.info(f"Metrics saved to {metrics_versioned_path}")

    print("Model training complete!")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Model version v{version} saved to {versioned_model_path}")


if __name__ == "__main__":
    config = load_config()
    train_model(config)
    


