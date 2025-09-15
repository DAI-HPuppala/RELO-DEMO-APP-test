#!/usr/bin/env python3
"""
Splunk HTTP Event Collector (HEC) Data Sender

This script formats data into JSON format and sends it to Splunk HEC endpoint.
Supports both single events and batch processing.
"""

import json
import requests
import time
import logging
from typing import Dict, List, Union, Optional
from datetime import datetime
import urllib3
import base64
import os

# Disable SSL warnings if using self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SplunkHECClient:
    """
    Splunk HTTP Event Collector client for sending data to Splunk.
    """
    
    def __init__(self, 
                 hec_url: str, 
                 hec_token: str, 
                 index: str = "main",
                 source: str = "python_script",
                 sourcetype: str = "json",
                 verify_ssl: bool = True):
        """
        Initialize the Splunk HEC client.
        
        Args:
            hec_url: Splunk HEC endpoint URL (e.g., https://splunk-server:8088/services/collector)
            hec_token: HEC authentication token
            index: Splunk index to send data to
            source: Source field for events
            sourcetype: Sourcetype field for events
            verify_ssl: Whether to verify SSL certificates
        """
        self.hec_url = hec_url.rstrip('/')
        self.hec_token = hec_token
        self.index = index
        self.source = source
        self.sourcetype = sourcetype
        self.verify_ssl = verify_ssl
        
        # Set up session with headers
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Splunk {self.hec_token}',
            'Content-Type': 'application/json'
        })
    
    def format_event(self, 
                     event_data: Union[Dict, str], 
                     timestamp: Optional[float] = None,
                     host: Optional[str] = None,
                     index: Optional[str] = None,
                     source: Optional[str] = None,
                     sourcetype: Optional[str] = None) -> Dict:
        """
        Format data into Splunk HEC event format.
        
        Args:
            event_data: The actual event data (dict or string)
            timestamp: Unix timestamp (defaults to current time)
            host: Host field (optional)
            index: Index override (optional)
            source: Source override (optional)
            sourcetype: Sourcetype override (optional)
            
        Returns:
            Formatted event dictionary
        """
        event = {
            "time": timestamp or time.time(),
            "event": event_data,
            "index": index or self.index,
            "source": source or self.source,
            "sourcetype": sourcetype or self.sourcetype
        }
        
        if host:
            event["host"] = host
            
        return event
    
    def send_event(self, 
                   event_data: Union[Dict, str], 
                   **kwargs) -> bool:
        """
        Send a single event to Splunk HEC.
        
        Args:
            event_data: The event data to send
            **kwargs: Additional arguments for format_event
            
        Returns:
            True if successful, False otherwise
        """
        try:
            event = self.format_event(event_data, **kwargs)
            response = self.session.post(
                f"{self.hec_url}/services/collector",
                data=json.dumps(event),
                verify=self.verify_ssl,
                timeout=(0.5, 5)
            )
            
            if response.status_code == 200:
                logger.info(f"Event sent successfully: {response.json()}")
                return True
            else:
                logger.error(f"Failed to send event. Status: {response.status_code}, Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending event: {str(e)}")
            return False
    
    def send_batch(self, 
                   events: List[Union[Dict, str]], 
                   **kwargs) -> bool:
        """
        Send multiple events in a single request.
        
        Args:
            events: List of event data
            **kwargs: Additional arguments for format_event
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Format all events
            formatted_events = []
            for event_data in events:
                formatted_events.append(self.format_event(event_data, **kwargs))
            
            # Join events with newlines for batch submission
            batch_data = '\n'.join([json.dumps(event) for event in formatted_events])
            
            response = self.session.post(
                f"{self.hec_url}/services/collector",
                data=batch_data,
                verify=self.verify_ssl,
                timeout=(0.5, 5)
            )
            
            if response.status_code == 200:
                logger.info(f"Batch sent successfully. {len(events)} events processed.")
                return True
            else:
                logger.error(f"Failed to send batch. Status: {response.status_code}, Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending batch: {str(e)}")
            return False
    
    def test_connection(self) -> bool:
        """
        Test the connection to Splunk HEC.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            test_event = {
                "message": "Connection test",
                "timestamp": datetime.now().isoformat(),
                "test": True
            }
            
            return self.send_event(test_event)
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def send_image(self, image_bytes: bytes, filename: Optional[str] = None, **kwargs) -> bool:
        """Encode image bytes to base64 and send as an event. """
        b64 = base64.b64encode(image_bytes).decode('ascii')
        event_data = {"filename": filename or os.path.basename(filename) if filename else "", "image_b64": b64}
        return self.send_event(event_data, **kwargs)


def example_usage():
    """
    Example usage of the SplunkHECClient.
    """
    # Configuration
    HEC_URL = "https://your-splunk-server:8088"
    HEC_TOKEN = "your-hec-token-here"
    INDEX = "main"
    
    # Initialize client
    client = SplunkHECClient(
        hec_url=HEC_URL,
        hec_token=HEC_TOKEN,
        index=INDEX,
        source="python_app",
        sourcetype="custom_logs",
        verify_ssl=False  # Set to True in production with valid certificates
    )
    
    # Test connection
    logger.info("Testing connection to Splunk HEC...")
    if not client.test_connection():
        logger.error("Connection test failed. Please check your configuration.")
        return
    
    # Example 1: Send a single event
    logger.info("Sending single event...")
    single_event = {
        "level": "INFO",
        "message": "Application started successfully",
        "user": "admin",
        "module": "authentication",
        "duration_ms": 150
    }
    client.send_event(single_event)
    
    # Example 2: Send multiple events in batch
    logger.info("Sending batch events...")
    batch_events = [
        {
            "level": "DEBUG",
            "message": "User login attempt",
            "user": "john_doe",
            "ip": "192.168.1.100",
            "success": True
        },
        {
            "level": "WARN",
            "message": "High CPU usage detected",
            "cpu_percent": 85.5,
            "server": "web-01"
        },
        {
            "level": "ERROR",
            "message": "Database connection failed",
            "database": "user_db",
            "retry_count": 3,
            "error_code": "CONN_TIMEOUT"
        }
    ]
    client.send_batch(batch_events)
    
    # Example 3: Send with custom metadata
    logger.info("Sending event with custom metadata...")
    custom_event = {
        "transaction_id": "TXN-12345",
        "amount": 99.99,
        "currency": "USD",
        "status": "completed"
    }
    client.send_event(
        custom_event,
        host="payment-server",
        source="payment_processor",
        sourcetype="transaction_logs"
    )
    
    logger.info("All examples completed!")


def send_csv_data_to_splunk(csv_file_path: str, client: SplunkHECClient):
    """
    Example function to read CSV data and send to Splunk.
    
    Args:
        csv_file_path: Path to CSV file
        client: SplunkHECClient instance
    """
    import csv
    
    try:
        with open(csv_file_path, 'r') as file:
            csv_reader = csv.DictReader(file)
            events = []
            
            for row in csv_reader:
                # Convert CSV row to event
                event = dict(row)
                events.append(event)
                
                # Send in batches of 100
                if len(events) >= 100:
                    client.send_batch(events)
                    events = []
            
            # Send remaining events
            if events:
                client.send_batch(events)
                
        logger.info(f"Successfully processed CSV file: {csv_file_path}")
        
    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_file_path}")
    except Exception as e:
        logger.error(f"Error processing CSV file: {str(e)}")


if __name__ == "__main__":
    # Run the example
    example_usage()
    
    # Uncomment below to test CSV processing
    # csv_file = "sample_data.csv"
    # client = SplunkHECClient("https://your-splunk:8088", "your-token", verify_ssl=False)
    # send_csv_data_to_splunk(csv_file, client)