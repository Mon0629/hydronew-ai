
import json
import time
from datetime import datetime
from mqtt_client import MQTTClient


def handle_incoming_message(topic: str, payload: str):

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n{'='*60}")
    print(f"Message Received at {timestamp}")
    print(f"{'='*60}")
    print(f"Topic: {topic}")
    
    # Try to parse as JSON
    try:
        data = json.loads(payload)
        print(f"Data (JSON):")
        for key, value in data.items():
            print(f"   • {key}: {value}")
    except json.JSONDecodeError:
        # Not JSON, display as plain text
        print(f"Payload: {payload}")
    
    print(f"{'='*60}\n")


def main():
    """
    Main function to set up MQTT subscription and listen forever.
    """
    print("\n" + "="*60)
    print("HydroNew AI - MQTT Subscriber")
    print("="*60)
    
    # Create MQTT client with your credentials
    print("\nInitializing MQTT client...")
    client = MQTTClient()
    
    # Connect to broker
    print("Connecting to MQTT broker...")
    if not client.connect(timeout=15):
        print(" Failed to connect to MQTT broker")
        print("\nPlease check:")
        print("  • Internet connection")
        print("  • Broker URL and credentials")
        print("  • Firewall settings (port 8883)")
        return
    
    print("Connected successfully!")
    
    # Subscribe to topics
    print("\n Setting up subscriptions...")
    
    topics = [
        "hydronew/ai/classification",                      # All hydronew topics
    ]
    
    # Subscribe to each topic
    for topic in topics:
        if client.subscribe(topic, message_handler=handle_incoming_message):
            print(f" Subscribed to: {topic}")
        else:
            print(f" Failed to subscribe to: {topic}")
    
    # Listen forever (blocking call)
    client.listen_forever()


if __name__ == "__main__":
    main()
