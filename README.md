# Intelligent Returns Classifier System

Real-time clothing returns classification using WebRTC streaming and AI vision (Qwen2.5-VL 3B).

## Features

- **Live Video Streaming**: WebRTC-based real-time camera feed (RealSense/Webcam)
- **Multi-Agent AI Pipeline**: Specialized agents for different classification tasks
- **Progressive Results**: Real-time updates as analysis progresses
- **Dual Modes**: Automatic (timer-based) and Manual (operator-controlled)
- **RealSense Support**: Native Intel RealSense camera integration with webcam fallback

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Intel RealSense camera (optional, will use webcam if not available)
- Modern web browser (Chrome/Firefox/Edge)
- GPU with 4-6GB VRAM (for AI model)
- Ollama installed (for VLM)

### 2. Installation

```bash
# Clone the repository
cd RELO-CLASSIFIER-DEV

# Install backend dependencies
cd backend
pip install -r requirements.txt
cd ..

# Install Ollama and model (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5-vl:3b
```

### 3. Test System Components

```bash
# Run system test
python test_system.py
```

This will check:
- Backend modules load correctly
- RealSense camera availability
- Webcam fallback
- Frontend files exist

### 4. Start the Backend Server

```bash
cd backend
python src/api/main.py
```

Server will start on `http://localhost:8000`

### 5. Open the Frontend

**Option 1: Direct file access**
- Open `frontend/src/index.html` in your browser

**Option 2: Serve via HTTP (recommended for WebRTC)**
```bash
# In a new terminal
python -m http.server 8080 --directory frontend/src
```
Then navigate to `http://localhost:8080`

## Usage

### Automatic Mode (Default)

1. Click **"Start Monitoring"**
2. Place clothing item in camera view
3. System automatically progresses through agents:
   - Initial Classifier (4s): Type, color, pattern
   - Detail Extractor (3s): Brand, size
   - Damage Detector (4s): Damage assessment
   - Final Compiler: Consolidated results
4. View final classification and export JSON

### Manual Mode

1. Toggle **"Manual Mode"** switch
2. Click **"Start Monitoring"**
3. Use control buttons:
   - **Capture Frame**: Freeze current view
   - **Analyze**: Trigger current agent
   - **Next Agent**: Skip to next agent
4. Review results at each step

## System Architecture

```
Frontend (Browser)          Backend (FastAPI)
     |                           |
     |-- WebSocket/WebRTC -------|
     |                           |
Video Display <--- Camera <--- RealSense/Webcam
     |                           |
Progress Panel <-- Results <-- Agent Pipeline
     |                           |
Control Panel --> Commands --> Session Manager
```

## Camera Support

The system supports multiple camera types with automatic fallback:

1. **Intel RealSense** (preferred)
   - Native depth sensing capabilities
   - Superior image quality
   - Automatic detection

2. **Webcam** (fallback)
   - Any USB or built-in camera
   - Automatic selection if RealSense unavailable

3. **Configuration**
   - Camera source set during session start
   - Automatic fallback to available camera

## API Endpoints

### REST API
- `POST /api/session/start` - Start classification session
- `GET /api/session/{id}/status` - Get session status
- `POST /api/session/{id}/trigger` - Manual agent trigger
- `GET /api/session/{id}/results` - Get final results
- `POST /api/session/{id}/export` - Export to JSON
- `GET /api/health` - System health check

### WebSocket
- `ws://localhost:8000/ws/stream` - WebRTC signaling and real-time updates

## Troubleshooting

### RealSense Not Detected

```bash
# Check RealSense connection
rs-enumerate-devices

# Ensure USB 3.0 connection
# Install Intel RealSense SDK if needed
```

### WebRTC Connection Failed

- Ensure backend is running on port 8000
- Check browser console for errors
- Use HTTP server for frontend (not file://)
- Check firewall settings

### No Video Stream

- Verify camera permissions in browser
- Check camera is not used by another application
- Try refreshing the page
- Check console for WebRTC errors

### CORS Issues

- Serve frontend via HTTP server (not file://)
- Or configure browser to allow local file access

## Development

### Project Structure

```
RELO-CLASSIFIER-DEV/
├── backend/
│   ├── src/
│   │   ├── api/          # FastAPI endpoints
│   │   ├── models/       # Data models
│   │   ├── services/     # Core services
│   │   └── agents/       # AI agents (future)
│   └── tests/            # Test suite
├── frontend/
│   └── src/
│       ├── index.html    # Main UI
│       ├── styles.css    # Styling
│       ├── app.js        # Main controller
│       ├── services/     # WebRTC client
│       └── components/   # UI components
└── specs/                # Documentation
```

### Running Tests

```bash
cd backend
pytest tests/ -v
```

### API Documentation

When backend is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Next Steps

- [ ] Implement VLM agent pipeline
- [ ] Add frame processing queue
- [ ] Implement damage detection logic
- [ ] Add session persistence
- [ ] Enhance progressive updates
- [ ] Add performance monitoring

## License

Proprietary - Denali Advanced Integration

## Support

For issues or questions, please contact the development team.