#!/usr/bin/env python3
"""Test VLM warmup with Denali logo"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from services.vlm_service import VLMService

async def test_vlm_warmup():
    """Test the VLM warmup functionality"""
    print("🔥 Testing VLM GPU lock-in and warmup...")
    
    # Create VLM service
    vlm_service = VLMService()
    
    # Progress callback
    async def progress_callback(message: str, progress: int):
        print(f"VLM [{progress:3d}%]: {message}")
    
    # Run warmup
    try:
        result = await vlm_service.initialize_and_warmup(progress_callback=progress_callback)
        print(f"✅ VLM warmup result: {result}")
        
        # Test status
        status = await vlm_service.get_warmup_status()
        print(f"📊 VLM status: {status}")
        
        # Test health check
        health = await vlm_service.health_check()
        print(f"🔍 VLM health: {health}")
        
    except Exception as e:
        print(f"❌ VLM warmup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_vlm_warmup())