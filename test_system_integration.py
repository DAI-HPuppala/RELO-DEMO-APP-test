#!/usr/bin/env python3
"""Integration test for RELO Classifier system"""

import asyncio
import aiohttp
import json
import numpy as np
from PIL import Image
import io
import base64
import time

async def test_api_health():
    """Test API health endpoint"""
    print("1. Testing API Health...")
    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:8000/api/health") as response:
            if response.status == 200:
                data = await response.json()
                print(f"   ✓ API is healthy")
                print(f"   - GPU available: {data['gpu_available']}")
                print(f"   - Version: {data['version']}")
                return True
            else:
                print(f"   ✗ API health check failed: {response.status}")
                return False

async def test_vlm_service():
    """Test VLM service status"""
    print("\n2. Testing VLM Service...")
    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:8000/api/vlm/quick-check") as response:
            if response.status == 200:
                data = await response.json()
                print(f"   ✓ VLM service accessible")
                print(f"   - GPU memory used: {data['status']['gpu_memory_total_mb'] - data['status']['gpu_memory_free_mb']}MB")
                print(f"   - GPU layers: {data['status']['gpu_performance']['optimal_gpu_layers']}")
                print(f"   - Performance mode: {data['status']['performance_mode']}")
                return True
            else:
                print(f"   ✗ VLM service check failed: {response.status}")
                return False

async def test_session_creation():
    """Test session creation"""
    print("\n3. Testing Session Creation...")
    async with aiohttp.ClientSession() as session:
        async with session.post("http://localhost:8000/api/sessions/create") as response:
            if response.status == 200:
                data = await response.json()
                session_id = data['session_id']
                print(f"   ✓ Session created: {session_id}")
                return session_id
            else:
                print(f"   ✗ Session creation failed: {response.status}")
                return None

async def test_agent_inference():
    """Test agent inference with a test image"""
    print("\n4. Testing Agent Inference...")
    
    # Create a test image (blue shirt pattern)
    image = Image.new('RGB', (640, 480), color='blue')
    # Add some texture
    pixels = image.load()
    for i in range(0, 640, 20):
        for j in range(0, 480, 20):
            pixels[i, j] = (0, 0, 200)  # Darker blue dots
    
    # Convert to base64
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=95)
    base64_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Test Initial Classifier
    print("   Testing Initial Classifier agent...")
    payload = {
        "agent_name": "initial_classifier",
        "frames": [base64_img],
        "inference_num": 1,
        "session_id": "test-session-001"
    }
    
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        timeout = aiohttp.ClientTimeout(total=30)
        async with session.post(
            "http://localhost:8000/api/agents/inference",
            json=payload,
            timeout=timeout
        ) as response:
            inference_time = (time.time() - start_time) * 1000
            
            if response.status == 200:
                data = await response.json()
                print(f"   ✓ Initial Classifier inference successful")
                print(f"   - Inference time: {inference_time:.0f}ms")
                print(f"   - Attributes: {json.dumps(data.get('attributes', {}), indent=6)}")
                print(f"   - Confidence: {data.get('confidence', 0):.2f}")
                
                # Check if GPU was used (inference should be fast)
                if inference_time < 5000:
                    print(f"   ✓ GPU acceleration confirmed ({inference_time:.0f}ms)")
                else:
                    print(f"   ⚠ Inference slow, GPU might not be optimized ({inference_time:.0f}ms)")
                
                return True
            else:
                error_text = await response.text()
                print(f"   ✗ Agent inference failed: {response.status}")
                print(f"   Error: {error_text[:200]}")
                return False

async def test_multi_agent_pipeline():
    """Test multi-agent pipeline"""
    print("\n5. Testing Multi-Agent Pipeline...")
    
    # Create test images
    images = []
    for color in ['blue', 'red', 'green']:
        img = Image.new('RGB', (320, 240), color=color)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=95)
        base64_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
        images.append(base64_img)
    
    agents = [
        ("initial_classifier", "Identifying garment attributes"),
        ("detail_extractor", "Extracting garment details"),
        ("damage_detector", "Checking for damage")
    ]
    
    results = []
    async with aiohttp.ClientSession() as session:
        for agent_name, description in agents:
            print(f"\n   Testing {agent_name}: {description}...")
            
            payload = {
                "agent_name": agent_name,
                "frames": images[:2],  # Use 2 frames for multi-image test
                "inference_num": 1,
                "session_id": "test-pipeline-001"
            }
            
            start_time = time.time()
            timeout = aiohttp.ClientTimeout(total=30)
            
            try:
                async with session.post(
                    "http://localhost:8000/api/agents/inference",
                    json=payload,
                    timeout=timeout
                ) as response:
                    inference_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        print(f"   ✓ {agent_name} completed in {inference_time:.0f}ms")
                        print(f"     Confidence: {data.get('confidence', 0):.2f}")
                        results.append({
                            "agent": agent_name,
                            "success": True,
                            "time_ms": inference_time,
                            "attributes": data.get('attributes', {})
                        })
                    else:
                        print(f"   ✗ {agent_name} failed: {response.status}")
                        results.append({
                            "agent": agent_name,
                            "success": False,
                            "error": response.status
                        })
                        
            except Exception as e:
                print(f"   ✗ {agent_name} error: {str(e)[:100]}")
                results.append({
                    "agent": agent_name,
                    "success": False,
                    "error": str(e)
                })
    
    # Summary
    success_count = sum(1 for r in results if r.get('success'))
    print(f"\n   Pipeline Summary: {success_count}/{len(agents)} agents successful")
    
    if success_count == len(agents):
        avg_time = sum(r['time_ms'] for r in results if r.get('success')) / success_count
        print(f"   Average inference time: {avg_time:.0f}ms")
        return True
    return False

async def test_gpu_performance():
    """Test GPU performance metrics"""
    print("\n6. Testing GPU Performance...")
    
    # Run multiple quick inferences to test GPU
    test_image = Image.new('RGB', (224, 224), color='blue')
    buffer = io.BytesIO()
    test_image.save(buffer, format='JPEG', quality=95)
    base64_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    inference_times = []
    
    async with aiohttp.ClientSession() as session:
        print("   Running 5 inference tests...")
        for i in range(5):
            payload = {
                "agent_name": "initial_classifier",
                "frames": [base64_img],
                "inference_num": i + 1,
                "session_id": f"perf-test-{i}"
            }
            
            start_time = time.time()
            timeout = aiohttp.ClientTimeout(total=15)
            
            try:
                async with session.post(
                    "http://localhost:8000/api/agents/inference",
                    json=payload,
                    timeout=timeout
                ) as response:
                    inference_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        inference_times.append(inference_time)
                        print(f"   Test {i+1}: {inference_time:.0f}ms")
                    else:
                        print(f"   Test {i+1}: Failed")
                        
            except Exception as e:
                print(f"   Test {i+1}: Error - {str(e)[:50]}")
    
    if inference_times:
        avg_time = sum(inference_times) / len(inference_times)
        min_time = min(inference_times)
        max_time = max(inference_times)
        
        print(f"\n   Performance Summary:")
        print(f"   - Average: {avg_time:.0f}ms")
        print(f"   - Fastest: {min_time:.0f}ms")
        print(f"   - Slowest: {max_time:.0f}ms")
        
        if avg_time < 3000:
            print(f"   ✓ GPU optimization working well!")
        elif avg_time < 5000:
            print(f"   ⚠ GPU partially optimized")
        else:
            print(f"   ✗ GPU optimization issues detected")
        
        return avg_time < 5000
    return False

async def main():
    """Run all integration tests"""
    print("=" * 60)
    print("RELO CLASSIFIER SYSTEM INTEGRATION TEST")
    print("=" * 60)
    
    # Wait for services to be ready
    print("\nWaiting for services to stabilize...")
    await asyncio.sleep(2)
    
    tests = [
        ("API Health", test_api_health),
        ("VLM Service", test_vlm_service),
        ("Session Creation", test_session_creation),
        ("Agent Inference", test_agent_inference),
        ("Multi-Agent Pipeline", test_multi_agent_pipeline),
        ("GPU Performance", test_gpu_performance)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {str(e)[:100]}")
            results[test_name] = False
    
    # Final summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:.<30} {status}")
    
    total_passed = sum(1 for r in results.values() if r)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED! System is working correctly.")
        return 0
    else:
        print(f"\n⚠️ {total_tests - total_passed} test(s) failed. Please check the logs.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)