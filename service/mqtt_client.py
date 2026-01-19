

import paho.mqtt.client as mqtt
import json
import logging
from typing import Callable, Optional
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MQTTClient:

    
    def __init__(
        self,
        broker: str = "960858f8c9cd49548edc44f8b9fac4e9.s1.eu.hivemq.cloud",
        port: int = 8883,
        username: str = "Biotech",
        password: str = "Momorevillame24",
        client_id: str = "hydronew_ai_client"
    ):

        self.broker = broker
        self.port = port
        self.username = username
        self.password = password
        self.client_id = client_id
        
        # Initialize MQTT client
        self.client = mqtt.Client(client_id=self.client_id, protocol=mqtt.MQTTv311)
        
        # Set username and password
        self.client.username_pw_set(self.username, self.password)
        
        # Enable TLS for secure connection
        self.client.tls_set()
        
        # Set callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        
        # Store custom message handler
        self.message_handler: Optional[Callable] = None
        
        # Connection status
        self.connected = False
        
        logger.info(f"MQTT Client initialized for broker: {self.broker}")
    
    def _on_connect(self, client, userdata, flags, rc):

        if rc == 0:
            self.connected = True
            logger.info("Successfully connected to MQTT broker")
        else:
            self.connected = False
            error_messages = {
                1: "Connection refused - incorrect protocol version",
                2: "Connection refused - invalid client identifier",
                3: "Connection refused - server unavailable",
                4: "Connection refused - bad username or password",
                5: "Connection refused - not authorized"
            }
            error_msg = error_messages.get(rc, f"Connection refused - unknown error code {rc}")
            logger.error(f"Failed to connect to MQTT broker: {error_msg}")
    
    def _on_disconnect(self, client, userdata, rc):

        self.connected = False
        if rc != 0:
            logger.warning(f"Unexpected disconnection from MQTT broker (code: {rc})")
        else:
            logger.info("Disconnected from MQTT broker")
    
    def _on_message(self, client, userdata, msg):

        topic = msg.topic
        payload = msg.payload.decode('utf-8')
        
        logger.info(f"Received message on topic '{topic}': {payload}")
        
        # Call custom message handler if set
        if self.message_handler:
            try:
                self.message_handler(topic, payload)
            except Exception as e:
                logger.error(f"Error in custom message handler: {e}")
    
    def connect(self, timeout: int = 10) -> bool:

        try:
            logger.info(f"Connecting to MQTT broker at {self.broker}:{self.port}")
            self.client.connect(self.broker, self.port, keepalive=60)
            
            # Start the network loop in a background thread
            self.client.loop_start()
            
            # Wait for connection to establish
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if self.connected:
                logger.info("Connection established successfully")
                return True
            else:
                logger.error("Connection timeout")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to MQTT broker: {e}")
            return False
    
    def disconnect(self):

        try:
            self.client.loop_stop()
            self.client.disconnect()
            logger.info("Disconnected from MQTT broker")
        except Exception as e:
            logger.error(f"Error disconnecting from MQTT broker: {e}")
    
    def subscribe(
        self,
        topic: str,
        qos: int = 1,
        message_handler: Optional[Callable[[str, str], None]] = None
    ) -> bool:

        if not self.connected:
            logger.error("Cannot subscribe - not connected to broker")
            return False
        
        try:
            # Set custom message handler if provided
            if message_handler:
                self.message_handler = message_handler
            
            # Subscribe to the topic
            result, mid = self.client.subscribe(topic, qos)
            
            if result == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"Successfully subscribed to topic: {topic} (QoS: {qos})")
                return True
            else:
                logger.error(f"Failed to subscribe to topic: {topic}")
                return False
                
        except Exception as e:
            logger.error(f"Error subscribing to topic '{topic}': {e}")
            return False
    
    def unsubscribe(self, topic: str) -> bool:

        if not self.connected:
            logger.error("Cannot unsubscribe - not connected to broker")
            return False
        
        try:
            result, mid = self.client.unsubscribe(topic)
            
            if result == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"Successfully unsubscribed from topic: {topic}")
                return True
            else:
                logger.error(f"Failed to unsubscribe from topic: {topic}")
                return False
                
        except Exception as e:
            logger.error(f"Error unsubscribing from topic '{topic}': {e}")
            return False
    
    def publish(self, topic: str, payload: str, qos: int = 1, retain: bool = False) -> bool:

        if not self.connected:
            logger.error("Cannot publish - not connected to broker")
            return False
        
        try:
            result = self.client.publish(topic, payload, qos, retain)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"Successfully published to topic: {topic}")
                return True
            else:
                logger.error(f"Failed to publish to topic: {topic}")
                return False
                
        except Exception as e:
            logger.error(f"Error publishing to topic '{topic}': {e}")
            return False
    
    def listen_forever(self):

        if not self.connected:
            logger.error("Cannot listen - not connected to broker")
            return
        
        logger.info(" Listening for messages... (Press Ctrl+C to stop)")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\n Stopping listener...")
        finally:
            self.disconnect()


def main():

    
    # Create MQTT client instance
    client = MQTTClient()
    
    # Connect to the broker
    if client.connect():
        print(" Connected to MQTT broker successfully!")
        
        client.listen_forever()
            
    else:
        print(" Failed to connect to MQTT broker")


if __name__ == "__main__":
    main()
