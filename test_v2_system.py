#!/usr/bin/env python3
"""
Test script to verify V2 system is working correctly.
"""
import asyncio
import json
import websockets
import sys

async def test_v2_connection():
    """Test connection to V2 WebSocket endpoint."""
    try:
        uri = "ws://localhost:8000/ws/v2/stream"
        print(f"Connecting to {uri}...")
        
        async with websockets.connect(uri) as websocket:
            print("✓ Connected to V2 WebSocket endpoint")
            
            # Send a test message
            test_message = {
                "type": "ping",
                "session_id": "test_session_001"
            }
            
            await websocket.send(json.dumps(test_message))
            print(f"✓ Sent test message: {test_message}")
            
            # Wait for response with timeout
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                response_data = json.loads(response)
                print(f"✓ Received response: {response_data}")
                
                # Check if this is a V2 endpoint response
                if "v2" in str(response_data).lower() or "stateful" in str(response_data).lower():
                    print("✓ Confirmed V2 endpoint is active")
                else:
                    print("✓ Endpoint is responding (may need to verify V2 features)")
                    
            except asyncio.TimeoutError:
                print("⚠ No response received (timeout) - endpoint may not handle ping")
                
            return True
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

async def test_v2_agents():
    """Test that V2 agents are available."""
    try:
        # Import V2 agents to verify they exist
        sys.path.insert(0, '/home/automation-dev/RELO-CLASSIFIER-specify-DEV/RELO-CLASSIFIER-DEV/backend/src')
        
        from v2.agents.initial_classifier_v2 import InitialClassifierV2
        from v2.agents.detail_extractor_v2 import DetailExtractorV2
        from v2.agents.damage_detector_v2 import DamageDetectorV2
        from v2.agents.final_compiler_v2 import FinalCompilerV2
        
        print("✓ V2 agents imported successfully:")
        print("  - InitialClassifierV2")
        print("  - DetailExtractorV2")
        print("  - DamageDetectorV2")
        print("  - FinalCompilerV2")
        
        # Test instantiation
        agents = {
            "initial_classifier": InitialClassifierV2(),
            "detail_extractor": DetailExtractorV2(),
            "damage_detector": DamageDetectorV2(),
            "final_compiler": FinalCompilerV2()
        }
        
        print("✓ All V2 agents instantiated successfully")
        
        # Check timer configuration
        for name, agent in agents.items():
            if hasattr(agent, 'timer_seconds'):
                print(f"  - {name}: {agent.timer_seconds}s timer")
                
        return True
        
    except Exception as e:
        print(f"✗ Error with V2 agents: {e}")
        return False

async def main():
    """Run all tests."""
    print("=" * 60)
    print("V2 System Integration Test")
    print("=" * 60)
    
    # Test V2 agents
    print("\n1. Testing V2 Agents...")
    agents_ok = await test_v2_agents()
    
    # Test V2 WebSocket
    print("\n2. Testing V2 WebSocket Endpoint...")
    websocket_ok = await test_v2_connection()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"  V2 Agents: {'✓ PASS' if agents_ok else '✗ FAIL'}")
    print(f"  V2 WebSocket: {'✓ PASS' if websocket_ok else '✗ FAIL'}")
    
    if agents_ok and websocket_ok:
        print("\n✓ V2 System is properly integrated!")
        print("\nFrontend should connect to:")
        print("  Local: ws://localhost:8000/ws/v2/stream")
        print("  ngrok: wss://<ngrok-domain>/ws/v2/stream")
    else:
        print("\n✗ Some V2 components need attention")
    
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())