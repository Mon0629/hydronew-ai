
import os
import logging
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


def get_db_config():
    """
    Get database configuration from environment variables.
    
    Returns:
        dict: Database configuration dictionary
    """
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '3306')),
        'user': os.getenv('DB_USER') or os.getenv('DB_USERNAME', 'root'),
        'password': os.getenv('DB_PASSWORD', ''),
        'database': os.getenv('DB_NAME') or os.getenv('DB_DATABASE', 'hydronew')
    }


def connect_database(db_config=None):

    if db_config is None:
        db_config = get_db_config()
    
    try:
        connection = mysql.connector.connect(**db_config)
        if connection.is_connected():
            logger.info(f" Connected to MySQL database: {db_config['database']}")
            logger.info(f"  Host: {db_config['host']}:{db_config.get('port', 3306)}")
            return connection
    except Error as e:
        logger.warning(f"  Could not connect to MySQL database: {e}")
        logger.warning("   Classification will continue without database updates")
        return None


def update_sensor_reading(connection, record_id, classification, confidence):

    if not connection or not connection.is_connected():
        logger.debug("Database not connected, skipping update")
        return False
    
    try:
        cursor = connection.cursor()
        
        classification_label = "good" if classification == 1 else "bad"

        # Update query
        update_query = """
            UPDATE sensor_readings 
            SET ai_classification = %s, confidence = %s
            WHERE id = %s
        """
        
        cursor.execute(update_query, (classification_label, confidence, record_id))
        connection.commit()
        
        logger.info(f" Database updated: ID={record_id}, Classification={classification_label}, Confidence={confidence}%")
        
        cursor.close()
        return True
        
    except Error as e:
        logger.error(f"Database update failed: {e}")
        return False


def close_connection(connection):
    """
    Close database connection safely.
    
    Args:
        connection: MySQL connection object
    """
    if connection and connection.is_connected():
        connection.close()
        logger.info(" Database connection closed")