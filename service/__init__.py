"""
MQTT Service Module for HydroNew AI Project.

This module provides MQTT client functionality for subscribing to and publishing
messages to an MQTT broker.
"""

from .mqtt_client import MQTTClient

__all__ = ['MQTTClient']
__version__ = '1.0.0'
