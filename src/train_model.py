import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
import logging
import json
from imblearn.over_sampling import SMOTE
from datetime import datetime

from utils import load_config, setup_logging, model_cleanup


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


def train_model(config):
    logging.info("Loading preprocessed training and test data...")
    train_path = config["data"]["train_processed_path"]
    test_path = config["data"]["test_processed_path"]
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    feature_cols = [
        'ph_distance', 'tds_distance', 'turbidity_distance',
        'ph_compliant', 'tds_compliant', 'turbidity_compliant',
        'compliance_score', 'weighted_score',
        'ph_tds_ratio', 'turbidity_ph_ratio', 'tds_turbidity_ratio'
    ]
    label_col = config["data"]["label_column"]

    X_train = df_train[feature_cols]
    y_train = df_train[label_col]
    X_test = df_test[feature_cols]
    y_test = df_test[label_col]

    # SMOTE with safe k_neighbors
    min_class_count = min(y_train.value_counts())
    if min_class_count > 1:
        k_neighbors = min(5, min_class_count - 1)
        smote = SMOTE(random_state=config["data"]["random_state"], k_neighbors=k_neighbors)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logging.info(f"After SMOTE, class distribution: {dict(pd.Series(y_train).value_counts())}")
    else:
        logging.warning("SMOTE skipped: minority class too small.")

    model_params = config["model"]["params"]
    model = RandomForestClassifier(**model_params)

    logging.info(f"Training {config['model']['type']} model with parameters: {model_params}")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    try:
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    except ValueError:
        roc_auc = float('nan')
        logging.warning("ROC-AUC cannot be computed: only one class in y_test")
    report = classification_report(y_test, y_pred, output_dict=True)

    logging.info(f"Model Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc}")
    logging.info(f"Classification Report: {report}")

    timestamp = datetime.now().strftime("%Y%m%d")

    # Save base model (always overwritten)
    base_model_path = os.path.join(config["model"]["base_path"], config["model"]["filename"])
    os.makedirs(os.path.dirname(base_model_path), exist_ok=True)
    joblib.dump(model, base_model_path)
    logging.info(f"Saved base model to {base_model_path}")

    model_cleanup(config["model"]["base_path"], base_name="random_forest", keep_last_n=5)

    # Save timestamped model (versioned)
    versioned_model_filename = f"random_forest_{timestamp}.pkl"
    versioned_model_path = os.path.join(config["model"]["base_path"], versioned_model_filename)
    joblib.dump(model, versioned_model_path)
    logging.info(f"Saved timestamped model to {versioned_model_path}")

    # Save metrics using same timestamp
    metrics_path = os.path.join("metrics", f"metrics_{timestamp}.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "classification_report": report
        }, f, indent=4)
    logging.info(f"Saved metrics to {metrics_path}")

    print(f"Model training complete! Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc}")
    print(f"Model saved as {versioned_model_path}")

if __name__ == "__main__":
    config = load_config()
    setup_logging(config["logging"]["training_log"], level=config["logging"].get("level", "INFO"))
    train_model(config)
