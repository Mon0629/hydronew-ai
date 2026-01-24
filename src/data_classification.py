import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path to import mqtt_client and src modules
sys.path.append(str(Path(__file__).parent.parent))

from service.mqtt_client import MQTTClient
from database.connect_database import connect_database, update_sensor_reading, close_connection, get_db_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent.parent / "logs" / "classification.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class WaterQualityClassifier:

    
    # Thresholds from preprocessing
    PH_MIN, PH_MAX = 6.5, 8.0
    TDS_MAX = 500
    TURB_MAX = 5
    EPS = 1e-6
    
    def __init__(self, model_path: str = "models/random_forest.pkl",
                 db_host: str = None, db_user: str = None, 
                 db_password: str = None, db_name: str = None):

        self.model_path = Path(__file__).parent.parent / model_path
        self.model = None
        self.mqtt_client = None
        self.db_connection = None
        
        self.load_model()
        self.db_connection = connect_database()
        
        logger.info("="*60)
        logger.info("Water Quality Classification Service Started")
        logger.info("="*60)
    
    def load_model(self):
        """Load the trained model from disk."""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found at: {self.model_path}")
            
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded successfully from: {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def engineer_features(self, data: dict) -> pd.DataFrame:

        # Extract sensor values (handle case variations)
        ph = float(data.get('pH') or data.get('ph', 0))
        tds = float(data.get('TDS') or data.get('tds', 0))
        turbidity = float(data.get('Turbidity') or data.get('turbidity', 0))
        
        # Create base dataframe
        df = pd.DataFrame({
            'ph': [ph],
            'tds': [tds],
            'turbidity': [turbidity]
        })
        
        # Distance features
        df["ph_distance"] = np.where(
            df["ph"] < self.PH_MIN,
            self.PH_MIN - df["ph"],
            np.where(df["ph"] > self.PH_MAX, df["ph"] - self.PH_MAX, 0)
        )
        df["tds_distance"] = np.maximum(0, df["tds"] - self.TDS_MAX)
        df["turbidity_distance"] = np.maximum(0, df["turbidity"] - self.TURB_MAX)
        
        # Ratio features
        df["ph_tds_ratio"] = df["ph"] / (df["tds"] + self.EPS)
        df["turbidity_ph_ratio"] = df["turbidity"] / (df["ph"] + self.EPS)
        df["tds_turbidity_ratio"] = df["tds"] / (df["turbidity"] + self.EPS)
        
        # Raw engineered signals
        df["ph_raw"] = df["ph"] / 14.0
        df["tds_raw"] = np.log1p(df["tds"]) / np.log1p(30000)
        df["turbidity_raw"] = np.log1p(df["turbidity"]) / np.log1p(50)
        
        # For single predictions, noise scores are 0
        df["ph_noise_score"] = 0.0
        df["tds_noise_score"] = 0.0
        df["turbidity_noise_score"] = 0.0
        df["instability_score"] = 0.0
        
        # Select only the feature columns needed by the model
        feature_cols = [
            'ph_distance', 'tds_distance', 'turbidity_distance',
            'ph_tds_ratio', 'turbidity_ph_ratio', 'tds_turbidity_ratio',
            'ph_raw', 'tds_raw', 'turbidity_raw',
            'ph_noise_score', 'tds_noise_score', 'turbidity_noise_score',
            'instability_score'
        ]
        
        return df[feature_cols]
    
    def classify(self, sensor_data: dict) -> dict:

        try:
            # Engineer features
            features_df = self.engineer_features(sensor_data)
            
            # Get prediction
            prediction = self.model.predict(features_df)[0]
            
            # Get probability/confidence
            probabilities = self.model.predict_proba(features_df)[0]
            confidence = float(max(probabilities)) * 100  # Convert to percentage
            
            # Map to label
            label = "good" if prediction == 1 else "bad"
            
            result = {
                'classification': int(prediction),  # 0 or 1
                'confidence': round(confidence, 2),  # Percentage
                'label': label,
                'probabilities': {
                    'bad': round(float(probabilities[0]) * 100, 2),
                    'good': round(float(probabilities[1]) * 100, 2)
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error during classification: {e}")
            raise
    
    def parse_iot_data(self, payload: str) -> dict:

        try:
            # Split payload by comma
            parts = payload.strip().split(',')
            
            if len(parts) < 2:
                raise ValueError(f"Invalid payload format: {payload}")
            
            # First part is water type
            water_type = parts[0].strip()
            
            # Parse sensor values
            sensor_data = {}
            for part in parts[1:]:
                if ':' in part:
                    key, value = part.split(':', 1)
                    sensor_data[key.strip()] = float(value.strip())
            
            return {
                'water_type': water_type,
                'sensor_data': sensor_data
            }
            
        except Exception as e:
            logger.error(f"Error parsing IoT data: {e}")
            raise
    
    def process_mqtt_message(self, topic: str, payload: str):
        """
        Process incoming MQTT messages and classify water quality.
        
        Expected payload format (first line is serial number):
        device_serial_number:BT20120
        dirty_water,ph:6.82,tds:2.41,turbidity:2.77,water_level:1.98
        clean_water,ph:7.12,tds:1.85,turbidity:0.92,water_level:2.10
        hydroponics_water,ph:6.45,tds:2.30,humidity:3.01,ec:1000
        
        Args:
            topic: MQTT topic (e.g., hydronew/ai/classification/...)
            payload: Raw string payload from IoT (multiple lines)
        """
        try:
            # Split payload by newlines
            lines = payload.strip().split('\n')
            
            if len(lines) < 2:
                logger.warning(f"Invalid payload format. Expected serial number on first line and data on subsequent lines.")
                return
            
            # First line should contain the serial number
            first_line = lines[0].strip()
            if not first_line.startswith('device_serial_number:'):
                logger.warning(f"First line must start with 'device_serial_number:'. Got: {first_line}")
                return
            
            # Extract serial number
            serial_number = first_line.split(':', 1)[1].strip()
            
            logger.info("="*60)
            logger.info(f"Received message from device: {serial_number}")
            logger.info("="*60)
            
            # Store processed water data as JSON objects
            water_samples = []
            
            # Process each data line (skip the first line which is serial number)
            for line in lines[1:]:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                try:
                    # Parse IoT data for this line
                    parsed_data = self.parse_iot_data(line)
                    water_type = parsed_data['water_type']
                    sensors = parsed_data['sensor_data']
                    
                    logger.info(f"Processing: {water_type}")
                    logger.info(f"  Sensor Data: {sensors}")
                    
                    # Create sample object
                    sample = {
                        'water_type': water_type,
                        'sensors': sensors
                    }
                    
                    # Only classify clean_water and dirty_water
                    if water_type not in ['clean_water', 'dirty_water']:
                        logger.info(f"  Skipping classification (not clean/dirty water)")
                        # Add without classification
                        water_samples.append(sample)
                        continue
                    
                    # Check if we have required sensors (ph, tds, turbidity)
                    # Handle both lowercase and uppercase variations
                    has_ph = 'ph' in sensors or 'pH' in sensors
                    has_tds = 'tds' in sensors or 'TDS' in sensors
                    has_turbidity = 'turbidity' in sensors or 'Turbidity' in sensors
                    
                    if not (has_ph and has_tds and has_turbidity):
                        missing = []
                        if not has_ph: missing.append('pH')
                        if not has_tds: missing.append('TDS')
                        if not has_turbidity: missing.append('Turbidity')
                        logger.warning(f"  Missing required sensors: {missing}. Cannot classify.")
                        # Add without classification
                        water_samples.append(sample)
                        continue
                    
                    # Prepare data for classification (only ph, tds, turbidity)
                    classification_data = {
                        'ph': sensors['ph'],
                        'tds': sensors['tds'],
                        'turbidity': sensors['turbidity']
                    }
                    
                    # Classify
                    result = self.classify(classification_data)
                    
                    logger.info(f"  Classification: {result['label']} (confidence: {result['confidence']}%)")
                    
                    # Add classification results to sample
                    sample['ai_classification'] = result['label']
                    sample['confidence'] = result['confidence']
                    
                    water_samples.append(sample)
                    
                except ValueError as e:
                    logger.error(f"  Invalid data format for line '{line}': {e}")
                except Exception as e:
                    logger.error(f"  Error processing line '{line}': {e}", exc_info=True)
            
            # Build JSON payload
            json_payload = {
                'device_serial_number': serial_number,
                'sensor_data': water_samples
            }
            
            # Convert to JSON string
            final_payload = json.dumps(json_payload, indent=2)
            
            # Publish to backend topic
            backend_topic = "hydronew/ai-classification/backend"
            self.mqtt_client.publish(backend_topic, final_payload)
            
            logger.info("-"*60)
            logger.info(f"Published to: {backend_topic}")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
    
    def start(self):

        try:
            # Create MQTT client
            self.mqtt_client = MQTTClient()
            
            # Connect to broker
            logger.info("Connecting to MQTT broker...")
            if not self.mqtt_client.connect(timeout=15):
                logger.error("Failed to connect to MQTT broker")
                return
            
            logger.info(" Connected to MQTT broker")
            
            # Subscribe to classification topic
            topic = "hydronew/ai/classification/#"
            logger.info(f"Subscribing to topic: {topic}")
            
            if self.mqtt_client.subscribe(topic, message_handler=self.process_mqtt_message):
                logger.info(f" Subscribed to: {topic}")
            else:
                logger.error(f"Failed to subscribe to: {topic}")
                return
            
            logger.info("="*60)
            logger.info(" Classification service is running...")
            logger.info("   Listening for water quality data...")
            logger.info("   Processing sensor_system_id: 1, 2 only")
            logger.info("   Press Ctrl+C to stop")
            logger.info("="*60)
            
            # Listen forever
            self.mqtt_client.listen_forever()
            
        except KeyboardInterrupt:
            logger.info("\n Service stopped by user")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
        finally:
            if self.mqtt_client:
                self.mqtt_client.disconnect()
            close_connection(self.db_connection)
            logger.info("Classification service stopped")


def main():

    # Check if model exists
    model_path = Path(__file__).parent.parent / "models" / "random_forest.pkl"
    if not model_path.exists():
        print(f" Error: Model not found at {model_path}")
        print("Run: python src/train_model.py")
        return
    
    # Start the classifier
    classifier = WaterQualityClassifier()
    classifier.start()


if __name__ == "__main__":
    main()
