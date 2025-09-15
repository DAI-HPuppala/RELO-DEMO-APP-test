# Quickstart Guide: Intelligent Returns Classifier System

## Prerequisites

- Python 3.11+
- GPU with 4-6GB VRAM
- Ollama installed
- Camera (webcam, RealSense, or Zebra CV60)
- Modern web browser (Chrome/Firefox/Edge)

## Installation

### 1. Install Ollama and Pull Model
```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Qwen2.5-VL 3B model
ollama pull qwen2.5-vl:3b

# Verify model is available
ollama list
```

### 2. Install Python Dependencies
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install backend dependencies
pip install -r backend/requirements.txt
```

### 3. Verify Camera Access
```bash
# Test camera availability
python backend/src/cli/camera_test.py --list

# Test specific camera
python backend/src/cli/camera_test.py --source webcam
```

## Running the System

### 1. Start the Backend Server
```bash
# From repository root
cd backend
python src/api/main.py

# Server starts on http://localhost:8000
# WebSocket available at ws://localhost:8000/ws/stream
```

### 2. Open the Frontend
```bash
# Open in browser
open frontend/src/index.html
# Or navigate to: file:///path/to/frontend/src/index.html
```

## Basic Usage

### Automatic Mode (Default)

1. **Start Monitoring**
   - Click "Start Monitoring" button
   - System begins automatic classification
   - Each agent runs for configured duration

2. **View Progressive Results**
   - Watch live camera feed on left panel
   - See real-time attribute detection in progress panel
   - Monitor current agent and timer in status bar

3. **Review Final Results**
   - After all agents complete (~15 seconds)
   - Final classification appears in bottom panel
   - Export results with "Export JSON" button

### Manual Mode

1. **Switch to Manual Mode**
   - Toggle "Manual Mode" switch
   - Agents wait for operator trigger

2. **Trigger Analysis**
   - Click "Analyze Frame" to trigger current agent
   - Review results before proceeding
   - Click "Next Agent" to continue pipeline

3. **Frame Capture**
   - Click "Capture Frame" to freeze current view
   - Useful for detailed inspection
   - Analysis uses captured frame

## Testing Individual Components

### Test WebRTC Connection
```bash
python backend/tests/contract/test_webrtc_connection.py
```

### Test VLM Service
```bash
python backend/src/cli/vlm_test.py --image samples/shirt.jpg
```

### Test Agent Pipeline
```bash
python backend/src/cli/agent_cli.py --mode test --image samples/shirt.jpg
```

### Test Session Management
```bash
python backend/tests/integration/test_session_lifecycle.py
```

## Quick Verification Script

```python
#!/usr/bin/env python3
"""Quick system verification"""

import asyncio
import aiohttp
import json

async def verify_system():
    """Verify all components are working"""
    
    # Check health endpoint
    async with aiohttp.ClientSession() as session:
        async with session.get('http://localhost:8000/api/health') as resp:
            health = await resp.json()
            print(f"✓ Server healthy: {health['status']}")
            print(f"✓ GPU available: {health['gpu_available']}")
            print(f"✓ Model loaded: {health['model_loaded']}")
            print(f"✓ Camera connected: {health['camera_connected']}")
    
    # Start test session
    async with aiohttp.ClientSession() as session:
        # Create session
        async with session.post('http://localhost:8000/api/session/start',
                                json={'mode': 'automatic'}) as resp:
            data = await resp.json()
            session_id = data['session_id']
            print(f"✓ Session created: {session_id}")
        
        # Wait for processing
        await asyncio.sleep(16)
        
        # Get results
        async with session.get(f'http://localhost:8000/api/session/{session_id}/results') as resp:
            results = await resp.json()
            if results['final_classification']:
                print("✓ Classification complete:")
                print(f"  Type: {results['final_classification']['type']}")
                print(f"  Color: {results['final_classification']['color_primary']}")
                print(f"  Brand: {results['final_classification']['brand']}")
                print(f"  Size: {results['final_classification']['size']}")

if __name__ == '__main__':
    asyncio.run(verify_system())
```

## Common Operations

### Export Session Results
```javascript
// In browser console
fetch(`/api/session/${sessionId}/export`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({format: 'json'})
})
.then(response => response.json())
.then(data => window.location.href = data.download_url);
```

### Switch Operating Modes
```javascript
// Switch to manual mode
websocket.send(JSON.stringify({
    type: 'switch_mode',
    session_id: currentSessionId,
    mode: 'manual'
}));
```

### Manual Agent Trigger
```javascript
// Trigger specific agent
websocket.send(JSON.stringify({
    type: 'manual_trigger',
    session_id: currentSessionId,
    agent: 'damage_detector'
}));
```

## Troubleshooting

### Camera Not Detected
```bash
# Check camera permissions
ls -l /dev/video*

# Test with OpenCV directly
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

### GPU Memory Issues
```bash
# Check GPU usage
nvidia-smi

# Restart Ollama service
systemctl restart ollama
```

### WebRTC Connection Failed
```bash
# Check if ports are open
netstat -an | grep 8000

# Test WebSocket connection
wscat -c ws://localhost:8000/ws/stream
```

### Model Not Loading
```bash
# Verify model exists
ollama list | grep qwen

# Test model directly
ollama run qwen2.5-vl:3b "What do you see?" --image samples/shirt.jpg
```

## Performance Monitoring

### Check System Metrics
```bash
# Monitor in real-time
watch -n 1 'nvidia-smi | grep python'
```

### View Logs
```bash
# Backend logs
tail -f backend/logs/app.log

# Session logs
ls -la backend/logs/sessions/
```

## Development Mode

### Run with Debug Logging
```bash
# Set debug environment
export DEBUG=true
export LOG_LEVEL=DEBUG

# Start server with auto-reload
uvicorn backend.src.api.main:app --reload --log-level debug
```

### Test with Mock Camera
```bash
# Use video file instead of camera
export CAMERA_SOURCE=samples/test_video.mp4
python backend/src/api/main.py
```

## Next Steps

1. Configure agent timers in `backend/config.yaml`
2. Adjust camera settings for your hardware
3. Customize UI colors/layout in `frontend/src/styles.css`
4. Add custom agents in `backend/src/agents/`
5. Set up production deployment with systemd service