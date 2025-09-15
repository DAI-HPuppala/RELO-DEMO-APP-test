#!/usr/bin/env python3
"""Test script for manual mode integration"""

import asyncio
import json
import websockets
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_manual_mode():
    """Test manual mode flow"""
    uri = "ws://localhost:8000/ws"
    
    async with websockets.connect(uri) as websocket:
        logger.info("Connected to WebSocket")
        
        # Start session
        await websocket.send(json.dumps({
            "type": "start_session",
            "mode": "manual",
            "camera_config": {"source": "webcam"}
        }))
        
        # Wait for session created
        response = await websocket.recv()
        data = json.loads(response)
        logger.info(f"Session created: {data}")
        session_id = data.get("session_id")
        
        if not session_id:
            logger.error("No session ID received")
            return
        
        # Create WebRTC offer (simplified for testing)
        await websocket.send(json.dumps({
            "type": "offer",
            "session_id": session_id,
            "sdp": "dummy_sdp_for_testing"
        }))
        
        # Wait for answer
        response = await websocket.recv()
        logger.info(f"WebRTC answer received: {json.loads(response).get('type')}")
        
        # Start manual mode monitoring
        await websocket.send(json.dumps({
            "type": "start_automatic",
            "session_id": session_id,
            "manual_mode": True,
            "agent_timers": {
                "initial_classifier": 2.0,
                "detail_extractor": 2.0,
                "damage_detector": 2.0
            }
        }))
        
        # Wait for manual mode started
        response = await websocket.recv()
        data = json.loads(response)
        logger.info(f"Manual mode response: {data}")
        
        # Listen for agent updates
        for _ in range(5):
            response = await websocket.recv()
            data = json.loads(response)
            logger.info(f"Update: {data.get('type')} - {data.get('agent', '')}")
            
            # If waiting for command, send next
            if data.get("waiting_for_command"):
                await asyncio.sleep(1)
                logger.info("Sending manual_next command")
                await websocket.send(json.dumps({
                    "type": "manual_next",
                    "session_id": session_id
                }))
        
        # Stop session
        await websocket.send(json.dumps({
            "type": "stop_session",
            "session_id": session_id
        }))
        
        logger.info("Test completed")

if __name__ == "__main__":
    asyncio.run(test_manual_mode())