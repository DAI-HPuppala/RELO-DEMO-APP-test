# Research Document: Intelligent Returns Classifier System

## WebRTC Implementation with aiortc

**Decision**: Use aiortc library for Python WebRTC implementation  
**Rationale**: 
- Pure Python implementation, no external dependencies
- Supports both data channels and media streams
- Integrates well with asyncio and FastAPI
- Handles STUN/TURN for NAT traversal

**Alternatives Considered**:
- SimpleWebRTC: JavaScript only, would require Node.js backend
- Janus Gateway: Overcomplicated for single-stream use case
- Raw WebRTC: Too low-level, would require extensive implementation

**Implementation Approach**:
```python
# Peer connection for video stream
pc = RTCPeerConnection()
# Add local camera track
pc.addTrack(camera_track)
# Data channel for progressive results
data_channel = pc.createDataChannel("results")
```

## Ollama Integration for Qwen2.5-VL 3B

**Decision**: Use Ollama as local VLM hosting service  
**Rationale**:
- Simple REST API for model inference
- Handles model loading and GPU management
- Supports streaming responses
- Easy model swapping without code changes

**Alternatives Considered**:
- Direct Hugging Face Transformers: More complex GPU management
- ONNX Runtime: Limited VLM support
- TensorRT: Nvidia-specific, less flexible

**Implementation Approach**:
```python
import ollama
# Initialize with Qwen model
response = ollama.chat(
    model='qwen2.5-vl:3b',
    messages=[{
        'role': 'user',
        'content': prompt,
        'images': [base64_image]
    }]
)
```

## Multi-Agent Orchestration Pattern

**Decision**: Sequential pipeline with configurable timers  
**Rationale**:
- Simple to implement and debug
- Clear agent boundaries
- Easy to add/remove agents
- Supports both automatic and manual modes

**Alternatives Considered**:
- Parallel processing: Would complicate result merging
- Event-driven: Unnecessary complexity for sequential flow
- State machine: Over-engineered for linear pipeline

**Implementation Approach**:
```python
class AgentPipeline:
    def __init__(self):
        self.agents = [
            InitialClassifier(),
            DetailExtractor(), 
            DamageDetector(),
            FinalCompiler()
        ]
    
    async def process_frame(self, frame, mode='auto'):
        results = {}
        for agent in self.agents:
            if mode == 'auto':
                await agent.process_with_timer(frame)
            else:
                await agent.process_on_demand(frame)
            results[agent.name] = agent.result
        return results
```

## Progressive Result Streaming

**Decision**: WebRTC data channel with JSON messages  
**Rationale**:
- Real-time, bidirectional communication
- Low latency compared to WebSocket
- Reliable ordered delivery
- Part of existing WebRTC connection

**Alternatives Considered**:
- Server-Sent Events: Unidirectional only
- WebSocket alongside WebRTC: Redundant connection
- Polling: High latency, inefficient

**Implementation Approach**:
```javascript
// Frontend
dataChannel.onmessage = (event) => {
    const update = JSON.parse(event.data);
    updateProgressPanel(update);
};

// Backend
data_channel.send(json.dumps({
    'type': 'agent_result',
    'agent': 'initial_classifier',
    'attributes': {...},
    'timestamp': time.time()
}))
```

## Camera Compatibility Strategy

**Decision**: OpenCV with multiple backend support  
**Rationale**:
- Unified interface for all camera types
- Automatic backend selection
- Fallback options available
- Well-documented and maintained

**Alternatives Considered**:
- Direct SDK integration: Different API for each camera
- GStreamer: Complex pipeline configuration
- V4L2 only: Linux-specific, limited features

**Implementation Approach**:
```python
class CameraManager:
    def get_camera(self, source):
        # Try RealSense first
        if self.is_realsense_available():
            return cv2.VideoCapture(cv2.CAP_REALSENSE)
        # Try Zebra EBUS
        elif self.is_ebus_available():
            return cv2.VideoCapture(cv2.CAP_ARAVIS)
        # Fallback to default webcam
        else:
            return cv2.VideoCapture(0)
```

## Session Management Architecture

**Decision**: In-memory with UUID tracking and JSON persistence  
**Rationale**:
- Fast access during active sessions
- Simple implementation
- Easy session recovery in manual mode
- Automatic cleanup on restart

**Alternatives Considered**:
- Redis: Unnecessary for single-instance deployment
- SQLite: Overhead for temporary data
- File-based only: Slower for active sessions

**Implementation Approach**:
```python
class SessionManager:
    def __init__(self):
        self.active_sessions = {}
    
    def create_session(self):
        session_id = str(uuid.uuid4())
        self.active_sessions[session_id] = {
            'created': datetime.now(),
            'status': 'active',
            'results': {},
            'mode': 'auto'
        }
        return session_id
    
    def save_session(self, session_id):
        if self.active_sessions[session_id]['mode'] == 'manual':
            with open(f'logs/{session_id}.json', 'w') as f:
                json.dump(self.active_sessions[session_id], f)
```

## Frontend Architecture

**Decision**: Vanilla JavaScript with Web Components  
**Rationale**:
- No build process required
- Native browser APIs
- Lightweight and fast
- Easy to maintain

**Alternatives Considered**:
- React: Overkill for simple UI
- Vue: Unnecessary reactivity complexity
- Angular: Too heavy for embedded system

**Implementation Approach**:
```javascript
// Custom elements for modularity
class VideoDisplay extends HTMLElement {
    connectedCallback() {
        this.innerHTML = `<video id="remoteVideo" autoplay></video>`;
        this.initWebRTC();
    }
}
customElements.define('video-display', VideoDisplay);
```

## Performance Optimization Strategy

**Decision**: Frame queuing with adaptive processing  
**Rationale**:
- Prevents frame drops
- Maintains consistent FPS
- Allows GPU to process efficiently
- Adapts to system load

**Alternatives Considered**:
- Process every frame: Would overload GPU
- Fixed sampling: Might miss important frames
- Buffering all frames: Memory intensive

**Implementation Approach**:
```python
class FrameProcessor:
    def __init__(self, max_queue_size=10):
        self.frame_queue = asyncio.Queue(maxsize=max_queue_size)
        self.processing = False
    
    async def add_frame(self, frame):
        if not self.frame_queue.full():
            await self.frame_queue.put(frame)
    
    async def process_frames(self):
        while True:
            frame = await self.frame_queue.get()
            # Process with VLM
            await self.vlm_service.analyze(frame)
```

## Error Recovery Mechanisms

**Decision**: Graceful degradation with automatic reconnection  
**Rationale**:
- Maintains service availability
- Preserves session data
- User-friendly error handling
- Automatic recovery when possible

**Alternatives Considered**:
- Full system restart: Loses session data
- Manual intervention only: Poor user experience
- Fail fast: Too disruptive for production

**Implementation Approach**:
```python
class ResilientService:
    async def execute_with_retry(self, func, max_retries=3):
        for attempt in range(max_retries):
            try:
                return await func()
            except Exception as e:
                if attempt == max_retries - 1:
                    return self.fallback_result()
                await asyncio.sleep(2 ** attempt)
```

## Testing Strategy

**Decision**: Contract-first with real dependencies  
**Rationale**:
- Ensures API stability
- Tests actual system behavior
- Catches integration issues early
- Follows TDD principles

**Alternatives Considered**:
- Mock everything: Doesn't test real behavior
- Unit tests only: Misses integration issues
- Manual testing: Not repeatable or scalable

**Implementation Approach**:
```python
# Contract test example
async def test_webrtc_connection():
    # Test with real WebRTC connection
    pc = RTCPeerConnection()
    offer = await pc.createOffer()
    assert offer.type == "offer"
    assert offer.sdp is not None
    
# Integration test example  
async def test_agent_pipeline():
    # Test with real VLM
    pipeline = AgentPipeline()
    frame = load_test_image()
    results = await pipeline.process_frame(frame)
    assert 'initial_classifier' in results
    assert results['initial_classifier']['type'] in ['shirt', 'pants', 'dress']
```

## Summary of Key Decisions

1. **WebRTC**: aiortc for Python backend, native API for frontend
2. **VLM**: Ollama service hosting Qwen2.5-VL 3B
3. **Architecture**: Sequential agent pipeline with timers
4. **Streaming**: WebRTC data channels for progressive updates
5. **Camera**: OpenCV with multi-backend support
6. **Sessions**: In-memory with UUID and JSON persistence
7. **Frontend**: Vanilla JS with Web Components
8. **Performance**: Adaptive frame queuing
9. **Recovery**: Graceful degradation with auto-reconnect
10. **Testing**: Contract-first with real dependencies

All technical decisions align with requirements for local processing, real-time performance, and modular architecture.