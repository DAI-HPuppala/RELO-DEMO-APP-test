#!/usr/bin/env python3
"""
Test script for VLM integration
Tests the VLM service with sample images to ensure it's working correctly
"""

import asyncio
import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from services.vlm_service import VLMService


async def create_test_images() -> list[np.ndarray]:
    """Create sample test images for testing"""
    images = []
    
    # Create a simple shirt-like image
    img1 = Image.new('RGB', (640, 480), color='blue')
    draw1 = ImageDraw.Draw(img1)
    
    # Draw a basic shirt shape
    draw1.rectangle([150, 100, 490, 400], fill='lightblue', outline='navy', width=3)
    draw1.rectangle([200, 100, 440, 200], fill='blue', outline='navy', width=2)  # Torso
    draw1.rectangle([100, 120, 200, 250], fill='blue', outline='navy', width=2)  # Left sleeve
    draw1.rectangle([440, 120, 540, 250], fill='blue', outline='navy', width=2)  # Right sleeve
    
    # Add some text
    try:
        font = ImageFont.load_default()
        draw1.text((250, 300), "Test Shirt", fill='white', font=font)
        draw1.text((200, 350), "Size: M", fill='white', font=font)
    except:
        draw1.text((250, 300), "Test Shirt", fill='white')
        draw1.text((200, 350), "Size: M", fill='white')
    
    # Convert to numpy array
    images.append(np.array(img1))
    
    # Create a second image with slight variation
    img2 = Image.new('RGB', (640, 480), color='lightgray')
    draw2 = ImageDraw.Draw(img2)
    
    # Draw the same shirt from a different angle
    draw2.rectangle([140, 110, 500, 390], fill='blue', outline='darkblue', width=3)
    draw2.rectangle([190, 110, 450, 210], fill='lightblue', outline='darkblue', width=2)
    
    try:
        font = ImageFont.load_default()
        draw2.text((250, 320), "COTTON", fill='white', font=font)
        draw2.text((200, 360), "Brand: TestCo", fill='white', font=font)
    except:
        draw2.text((250, 320), "COTTON", fill='white')
        draw2.text((200, 360), "Brand: TestCo", fill='white')
    
    images.append(np.array(img2))
    
    print(f"Created {len(images)} test images")
    return images


async def test_vlm_health():
    """Test VLM service health check"""
    print("🏥 Testing VLM Service Health...")
    vlm_service = VLMService()
    
    healthy = await vlm_service.health_check()
    if healthy:
        print("✅ VLM service is healthy and qwen2.5-vl model is available")
        return True
    else:
        print("❌ VLM service health check failed")
        print("   - Make sure Ollama is running: ollama serve")
        print("   - Make sure qwen2.5-vl model is installed: ollama pull qwen2.5-vl")
        return False


async def test_single_image_inference():
    """Test inference with a single image"""
    print("\n🖼️ Testing Single Image Inference...")
    
    vlm_service = VLMService()
    test_images = await create_test_images()
    
    prompt = "Analyze this garment image and identify: item_type, color, pattern, neckline, sleeve_type, closure_type"
    
    try:
        result = await vlm_service.infer([test_images[0]], prompt)
        
        print(f"Result: {result}")
        
        if result.get('error'):
            print(f"❌ Inference failed: {result['error']}")
            return False
        
        attributes = result.get('attributes', {})
        confidence = result.get('confidence', 0.0)
        
        print(f"✅ Inference successful!")
        print(f"   Confidence: {confidence:.2f}")
        print(f"   Attributes: {attributes}")
        
        # Check if we got reasonable results
        if attributes and confidence > 0.1:
            return True
        else:
            print("⚠️ Warning: Low confidence or empty attributes")
            return False
            
    except Exception as e:
        print(f"❌ Inference exception: {e}")
        return False


async def test_multi_image_inference():
    """Test inference with multiple images"""
    print("\n🖼️🖼️ Testing Multi-Image Inference...")
    
    vlm_service = VLMService()
    test_images = await create_test_images()
    
    prompt = "These 2 images show the same garment from different angles. Analyze and identify: item_type, color, pattern, neckline, sleeve_type, closure_type"
    
    try:
        result = await vlm_service.infer(test_images, prompt)
        
        print(f"Result: {result}")
        
        if result.get('error'):
            print(f"❌ Multi-image inference failed: {result['error']}")
            return False
        
        attributes = result.get('attributes', {})
        confidence = result.get('confidence', 0.0)
        
        print(f"✅ Multi-image inference successful!")
        print(f"   Confidence: {confidence:.2f}")
        print(f"   Attributes: {attributes}")
        
        return True
            
    except Exception as e:
        print(f"❌ Multi-image inference exception: {e}")
        return False


async def test_all_agent_types():
    """Test all agent types with appropriate prompts"""
    print("\n🤖 Testing All Agent Types...")
    
    vlm_service = VLMService()
    test_images = await create_test_images()
    
    agent_tests = [
        {
            "name": "Initial Classifier",
            "prompt": "Analyze this garment image and identify: item_type, color, pattern, neckline, sleeve_type, closure_type"
        },
        {
            "name": "Detail Extractor", 
            "prompt": "Extract details from this garment: brand, size, material, care instructions"
        },
        {
            "name": "Damage Detector",
            "prompt": "Inspect this garment for damage. Identify: is_damaged (yes/no), damage_type, damage_severity"
        }
    ]
    
    success_count = 0
    
    for test in agent_tests:
        print(f"\n  Testing {test['name']}...")
        try:
            result = await vlm_service.infer([test_images[0]], test['prompt'])
            
            if result.get('error'):
                print(f"    ❌ {test['name']} failed: {result['error']}")
            else:
                attributes = result.get('attributes', {})
                confidence = result.get('confidence', 0.0)
                print(f"    ✅ {test['name']} - Confidence: {confidence:.2f}, Attributes: {len(attributes)} found")
                success_count += 1
                
        except Exception as e:
            print(f"    ❌ {test['name']} exception: {e}")
    
    print(f"\n🤖 Agent Test Results: {success_count}/{len(agent_tests)} successful")
    return success_count == len(agent_tests)


async def main():
    """Main test function"""
    print("🧪 VLM Integration Test Suite")
    print("=" * 40)
    
    # Test 1: Health check
    if not await test_vlm_health():
        print("\n❌ VLM service not available. Please check Ollama setup.")
        return False
    
    # Test 2: Single image inference
    if not await test_single_image_inference():
        print("\n❌ Single image inference test failed")
        return False
    
    # Test 3: Multi-image inference
    if not await test_multi_image_inference():
        print("\n❌ Multi-image inference test failed")
        return False
    
    # Test 4: All agent types
    if not await test_all_agent_types():
        print("\n⚠️ Some agent types failed, but core functionality works")
    
    print("\n" + "=" * 40)
    print("🎉 VLM Integration Tests Complete!")
    print("✅ The system is ready to use real VLM inference instead of mock data")
    print("\nNext steps:")
    print("1. Start the backend: python backend/src/api/main.py")  
    print("2. Open the frontend and start a classification session")
    print("3. Observe real VLM analysis in the logs and UI")
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)