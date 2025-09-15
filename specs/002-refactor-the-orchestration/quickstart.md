# Quickstart Guide: Stateful Agent Orchestration

**Feature**: Stateful Agent Orchestration with Multi-Image Inference  
**Version**: 2.0.0  
**Prerequisites**: Python 3.11, Ollama with qwen2.5vl:3b model, WebRTC-capable browser

## Quick Test Scenarios

### Scenario 1: Complete Automatic Flow with Multi-Inference
Verify that agents perform multiple inferences within their timer windows.

```bash
# Terminal 1: Start backend
cd backend
python src/api/main.py

# Terminal 2: Monitor logs
tail -f backend.log | grep -E "inference_num|timer_remaining|aggregation"

# Browser: Open frontend
http://localhost:8080

# Steps:
1. Click "Start Monitoring" (Automatic mode)
2. Place garment in camera view
3. Observe agent progression:
   - Initial Classifier (4s) → Expect 3-4 inferences
   - Detail Extractor (3s) → Expect 2-3 inferences  
   - Damage Detector (4s) → Expect 2-3 batch inferences
   - Final Compiler → Aggregation only
4. Verify final results show aggregated attributes

# Expected Log Pattern:
[Orchestrator] initial_classifier INFERENCE #1/4 - single_image
[Orchestrator] initial_classifier INFERENCE #2/4 - multi_image (2 frames)
[Orchestrator] initial_classifier INFERENCE #3/4 - multi_image (3 frames)
[Orchestrator] initial_classifier aggregating with majority voting
```

### Scenario 2: Pause and Resume During Processing
Test pause/resume functionality during agent execution.

```bash
# Setup same as Scenario 1, then:

# Steps:
1. Start automatic mode
2. Wait for Detail Extractor to begin (watch for "agent_started: detail_extractor")
3. Click "Pause" button
4. Verify:
   - Current agent stops
   - No new inferences start
   - UI shows paused state
5. Click "Resume"
6. Verify:
   - Detail Extractor restarts from beginning
   - Timer resets to full duration
   - Flow continues to completion

# Expected WebSocket Messages:
→ {"type": "pause_flow", "session_id": "..."}
← {"type": "flow_paused", "paused_agent": "detail_extractor", ...}
→ {"type": "resume_flow", "session_id": "...", "restart_agent": true}
← {"type": "flow_resumed", "resuming_agent": "detail_extractor"}
← {"type": "agent_started", "agent": "detail_extractor", "timer_seconds": 3.0}
```

### Scenario 3: Frame Sharing Validation
Verify initial agent's first frame is shared with damage detector.

```bash
# Enable debug frame saving
export SAVE_DEBUG_FRAMES=true

# Start system and run automatic mode

# After completion, verify frame sharing:
ls -la captured_frames/initial_classifier/ | head -2
ls -la captured_frames/damage_detector/

# Expected: First frame from initial_classifier appears in damage_detector's first batch

# Verify in logs:
grep "Sharing frame" backend.log
# Should show: "Registered initial first frame: frame_id_xxx for session yyy"
grep "damage_detector.*batch.*2 frames" backend.log  
# First inference should be batch (2+ frames)
```

### Scenario 4: Single Inference Fallback
Test aggregation when agent only completes 1-2 inferences (edge case testing).

```bash
# NOTE: This tests CUSTOM timer configuration (not the default 4,3,4 seconds)
# The system allows configurable timers for testing edge cases

# Reduce timer to force minimal inferences
# In browser console before starting:
window.customTimers = {
  initial_classifier: 1.5,  // Force 1 inference only
  detail_extractor: 2.0,    // Force 1-2 inferences
  damage_detector: 4.0      // Normal operation
};

# Start automatic mode with reduced timers

# Expected Aggregation Behavior:
- Initial (1 inference): Use that single result as final
- Detail (2 inferences): 
  * Use second inference values
  * If second has null → use first's value
  * If both null → keep as null
- Damage (3+ inferences): Use majority voting

# Verify in response:
{
  "agent": "initial_classifier",
  "results": {
    "total_inferences": 1,
    "aggregation_method": "single",
    ...
  }
}
```

### Scenario 5: Progressive Update Monitoring
Verify real-time inference updates reach frontend.

```bash
# Browser console (before starting):
window.addEventListener('message', (e) => {
  if (e.data.type === 'inference_update') {
    console.log(`Agent: ${e.data.agent}, Inference: ${e.data.inference_num}, Frames: ${e.data.frames_used}, Timer: ${e.data.timer_remaining}s`);
  }
});

# Start automatic mode and watch console

# Expected Output Pattern:
Agent: initial_classifier, Inference: 1, Frames: 1, Timer: 3.2s
Agent: initial_classifier, Inference: 2, Frames: 2, Timer: 2.1s
Agent: initial_classifier, Inference: 3, Frames: 3, Timer: 0.9s
Agent: detail_extractor, Inference: 1, Frames: 1, Timer: 2.5s
...
```

## Performance Validation

### GPU Utilization Check
```bash
# Monitor GPU during processing
watch -n 0.5 nvidia-smi

# Expected:
- Memory: ~6-7GB during inference
- Utilization: 80-95% during VLM processing
- No concurrent GPU processes (serialized access)
```

### Timer Accuracy Test
```python
# Test script: verify_timers.py
import asyncio
import time
from backend.src.services.stateful_orchestrator import StatefulOrchestrator

async def test_timer_accuracy():
    orchestrator = StatefulOrchestrator()
    
    # Mock frame provider
    frames = [np.zeros((480, 640, 3), dtype=np.uint8)] * 10
    frame_provider = lambda: frames.pop(0) if frames else None
    
    start = time.time()
    result = await orchestrator.run_agent_with_timer(
        "test_agent", 
        frame_provider, 
        timer_seconds=4.0
    )
    elapsed = time.time() - start
    
    assert 3.9 <= elapsed <= 4.1, f"Timer inaccurate: {elapsed}s"
    print(f"✓ Timer accuracy: {elapsed}s for 4s timer")

asyncio.run(test_timer_accuracy())
```

### Memory Leak Test
```bash
# Run 100 cycles and monitor memory
for i in {1..100}; do
    curl -X POST http://localhost:8080/api/test/cycle
    echo "Cycle $i complete"
    sleep 1
done

# Check for memory growth
ps aux | grep python | grep main.py
# RSS should stabilize, not continuously grow
```

## Debugging Commands

### Check Session State
```bash
# Get current session state
curl http://localhost:8080/api/session/{session_id}/state

# List all sessions
curl http://localhost:8080/api/sessions
```

### Force Agent Completion
```python
# Python console for debugging
from backend.src.api.main import get_session_manager
sm = get_session_manager()
session = sm.get_session("session_id_here")
session.agent_states["initial_classifier"].status = "completed"
```

### Verify Frame Registry
```python
# Check frame sharing
from backend.src.services.frame_registry import FrameRegistry
registry = FrameRegistry("session_id")
shared = registry.get_shared_frame("initial_first")
print(f"Shared frame: {shared.frame_id if shared else 'None'}")
```

## Common Issues

### Issue: Agent completes with 0 inferences
**Cause**: Timer too short or inference timeout  
**Fix**: Increase timer or check Ollama performance
```bash
# Test Ollama response time
time curl -X POST http://localhost:11434/api/generate \
  -d '{"model":"qwen2.5vl:3b","prompt":"test"}'
```

### Issue: Pause button disabled
**Cause**: Final compiler agent is running  
**Fix**: This is by design - final compiler cannot be paused

### Issue: Frame sharing not working
**Cause**: Frame registry not initialized  
**Fix**: Check session initialization includes frame_registry
```python
# Verify in websocket.py
session = SessionState(
    session_id=str(uuid4()),
    frame_registry=FrameRegistry(session_id)  # Must be present
)
```

### Issue: Aggregation using wrong method
**Cause**: Inference count threshold not met  
**Fix**: Verify aggregation logic
```python
# Check aggregation method selection
if len(inferences) >= 3:
    method = "majority_voting"
elif len(inferences) == 2:
    method = "last_with_fallback"
else:
    method = "single_inference"
```

## Success Criteria

✅ **Multi-Inference**: Each agent performs multiple inferences within timer  
✅ **Progressive Updates**: Frontend receives real-time inference updates  
✅ **Pause/Resume**: Flow can be paused and resumed correctly  
✅ **Frame Sharing**: Initial's first frame used by damage detector  
✅ **Aggregation**: Results correctly aggregated based on inference count  
✅ **GPU Isolation**: No concurrent GPU access, serialized processing  
✅ **Timer Accuracy**: Agents respect configured timer durations  
✅ **State Persistence**: Session state recoverable after pause  
✅ **Final Compilation**: All agent results combined with reasoning sequence  
✅ **Performance**: <1s per inference, <5s total per agent