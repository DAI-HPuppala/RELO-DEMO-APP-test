#!/usr/bin/env python3
"""Test Ollama vision model with a captured frame."""

import base64
import json
import requests
import sys
from pathlib import Path

def test_ollama_vision(image_path):
    """Test Ollama vision model with an image."""
    
    # Read and encode image
    with open(image_path, "rb") as f:
        image_data = f.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    
    # Prepare request
    request_data = {
        "model": "qwen2.5vl:3b",
        "prompt": "What do you see in this image? Describe any clothing items visible.",
        "images": [image_base64],
        "stream": False
    }
    
    print(f"Testing with image: {image_path}")
    print(f"Image size: {len(image_data)} bytes")
    print(f"Base64 size: {len(image_base64)} chars")
    print("Sending request to Ollama (this may take 30-60 seconds)...")
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=request_data,
            timeout=120  # 2 minute timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Success!")
            print(f"Response: {result.get('response', 'No response')}")
            print(f"Total duration: {result.get('total_duration', 0) / 1e9:.2f} seconds")
            print(f"Load duration: {result.get('load_duration', 0) / 1e9:.2f} seconds")
            print(f"Eval duration: {result.get('eval_duration', 0) / 1e9:.2f} seconds")
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.Timeout:
        print("\n❌ Request timed out after 120 seconds")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    # Find a test image
    frames_dir = Path("backend/captured_frames")
    
    # Get the most recent frame
    frames = list(frames_dir.glob("**/*.jpg"))
    
    if frames:
        # Use the most recent frame
        test_frame = sorted(frames, key=lambda x: x.stat().st_mtime)[-1]
        print(f"Found {len(frames)} frames, using most recent: {test_frame}")
        test_ollama_vision(test_frame)
    else:
        print("No captured frames found. Please run the application first to capture some frames.")
        sys.exit(1)