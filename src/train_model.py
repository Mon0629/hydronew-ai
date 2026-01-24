import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, log_loss
import joblib
import logging
import json
from imblearn.over_sampling import SMOTE
from datetime import datetime

from .utils import load_config, setup_logging, model_cleanup

# -----------------------
# Helper functions
# -----------------------
def get_next_model_version(base_dir, base_name="random_forest"):
    os.makedirs(base_dir, exist_ok=True)
    existing_files = os.listdir(base_dir)
    version_numbers = []
    for filename in existing_files:
        if filename.startswith("v") and filename.endswith(".pkl") and base_name in filename:
            try:
                v = int(filename.split("_")[0][1:])
                version_numbers.append(v)
            except:
                pass
    next_version = max(version_numbers, default=0) + 1
    next_model_filename = f"v{next_version}_{base_name}.pkl"
    return os.path.join(base_dir, next_model_filename), next_version

def add_sensor_noise(X, noise_level=0.01):
    """Add small Gaussian noise to sensor features to prevent memorization."""
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

# -----------------------
# Main training function
# -----------------------
def train_model(config):
    logging.info("Loading preprocessed training and test data...")
    train_path = config["data"]["train_processed_path"]
    test_path = config["data"]["test_processed_path"]
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    feature_cols = [
        # Distance features
        'ph_distance', 'tds_distance', 'turbidity_distance',
        # Ratios
        'ph_tds_ratio', 'turbidity_ph_ratio', 'tds_turbidity_ratio',
        # Raw engineered signals
        'ph_raw', 'tds_raw', 'turbidity_raw',
        # Noise-sensitive metrics
        'ph_noise_score', 'tds_noise_score', 'turbidity_noise_score', 'instability_score'
    ]
    label_col = config["data"]["label_column"]

    # Separate features and label
    X_train = df_train[feature_cols]
    y_train = df_train[label_col]
    X_test = df_test[feature_cols]
    y_test = df_test[label_col]

    # -----------------------
    # Add noise to mimic real sensor variation
    # -----------------------
    X_train = add_sensor_noise(X_train, noise_level=0.01)
    X_test = add_sensor_noise(X_test, noise_level=0.01)

    # -----------------------
    # Apply SMOTE if minority class exists
    # -----------------------
    min_class_count = min(y_train.value_counts())
    if min_class_count > 1:
        k_neighbors = min(5, min_class_count - 1)
        smote = SMOTE(random_state=config["data"]["random_state"], k_neighbors=k_neighbors)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logging.info(f"After SMOTE, class distribution: {dict(pd.Series(y_train).value_counts())}")
    else:
        logging.warning("SMOTE skipped: minority class too small.")

    # -----------------------
    # Model setup (reduced complexity)
    # -----------------------
    model_params = {
        'n_estimators': 150,
        'max_depth': 8,
        'min_samples_leaf': 5,
        'max_features': 0.7,
        'random_state': config["data"]["random_state"],
        'class_weight': 'balanced'
    }
    model = RandomForestClassifier(**model_params)

    logging.info(f"Training RandomForest model with parameters: {model_params}")
    model.fit(X_train, y_train)

    # -----------------------
    # Predictions & evaluation
    # -----------------------
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    try:
        roc_auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        roc_auc = float('nan')
        logging.warning("ROC-AUC cannot be computed: only one class in y_test")
    logloss = log_loss(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)

    logging.info(f"Model Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}, Log Loss: {logloss:.4f}")
    logging.info(f"Classification Report: {report}")

    # -----------------------
    # Save models
    # -----------------------
    timestamp = datetime.now().strftime("%Y%m%d")

    base_model_path = os.path.join(config["model"]["base_path"], config["model"]["filename"])
    os.makedirs(os.path.dirname(base_model_path), exist_ok=True)
    joblib.dump(model, base_model_path)
    logging.info(f"Saved base model to {base_model_path}")

    model_cleanup(config["model"]["base_path"], base_name="random_forest", keep_last_n=5)

    versioned_model_path = os.path.join(config["model"]["base_path"], f"random_forest_{timestamp}.pkl")
    joblib.dump(model, versioned_model_path)
    logging.info(f"Saved timestamped model to {versioned_model_path}")

    metrics_path = os.path.join("metrics", f"metrics_{timestamp}.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "log_loss": logloss,
            "classification_report": report
        }, f, indent=4)
    logging.info(f"Saved metrics to {metrics_path}")

    print(f"Model training complete! Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}, Log Loss: {logloss:.4f}")
    print(f"Model saved as {versioned_model_path}")


# -----------------------
# Entry point
# -----------------------
if __name__ == "__main__":
    config = load_config()
    setup_logging(config["logging"]["training_log"], level=config["logging"].get("level", "INFO"))
    train_model(config)
