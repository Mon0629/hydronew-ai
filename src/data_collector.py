import os
import pandas as pd
from firebase_admin import db
from firebase.firebase_config import initialize_firebase
from datetime import datetime

# Initialize Firebase
initialize_firebase()

STATUS_MAPPING = {
    "Unsafe for plants" : 0,
    "Safe for plants": 1,
}

# List all relevant Firebase env variables
def pull_data():
    ref = db.reference(os.getenv("FIREBASE_DATA_PATH", "sensorData"))
    data = ref.get()
    return data

def clean_data(raw_data):
    cleaned_data = []
    for key, records in raw_data.items():
        cleaned_data.append({
            "ph": records.get("ph"),
            "turbidity": records.get("turbidity"),
            "tds": records.get("tds"),
            "status": STATUS_MAPPING.get(records.get("status"), 0),
            "timestamp": records.get("timestamp")
        })
    df = pd.DataFrame(cleaned_data)
    print("Cleaned DataFrame:\n", df.head())
    return df

def save_to_csv(df, folder):
    os.makedirs(folder, exist_ok=True)
    
    today_str = datetime.now().strftime("%Y%m%d")
    filename = f"sensor_data_{today_str}.csv"
    filepath = os.path.join(folder, filename)
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filename}")


if __name__ == "__main__":
    raw_data = pull_data()
    if raw_data:
        df = clean_data(raw_data)
        save_to_csv(df, "data/raw")