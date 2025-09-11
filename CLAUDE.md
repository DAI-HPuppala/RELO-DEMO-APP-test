# Claude Code Context: Intelligent Returns Classifier

## Project Overview
AI-powered clothing returns classifier using WebRTC streaming and Vision-Language Model (Qwen2.5-VL 3B) for real-time attribute detection.

## Tech Stack
- **Backend**: Python 3.11, FastAPI, aiortc (WebRTC), Ollama (VLM)
- **Frontend**: Vanilla JS, WebRTC API, Web Components
- **Model**: Qwen2.5-VL 3B via Ollama
- **Testing**: pytest, Contract-first TDD

## Architecture
```
Frontend (Browser) <--WebRTC--> Backend (FastAPI)
                                    |
                              Agent Pipeline
                                    |
                              Ollama (VLM)
```

## Key Patterns

### Stateful Multi-Agent Pipeline (v2.0)
```python
# Each agent performs multiple inferences within timer window
agent_state = AgentState(timer_seconds=4.0)
while agent_state.timer_remaining > 1.0:
    frames = collect_frames()  # Single → Multi-image batch
    inference = await run_inference(frames)
    agent_state.add_inference(inference)
result = aggregate_results(agent_state.inferences)
```

### WebRTC Data Flow
- Video: Camera → Backend → Frame Processor → Agents
- Results: Agents → Data Channel → Frontend (real-time updates)
- Frame Isolation: Copy-on-capture prevents stream blocking

### Session State Management
- Stateful orchestration with pause/resume capability
- In-memory with async JSON checkpointing
- Frame registry for controlled sharing (initial→damage)

## Project Structure
```
backend/
├── src/
│   ├── models/      # Data models
│   ├── services/    # Core services (WebRTC, VLM)
│   ├── agents/      # Classification agents
│   └── api/         # FastAPI endpoints
frontend/
├── src/
│   ├── components/  # Web Components
│   └── services/    # WebRTC client
```

## Key Files
- `backend/src/api/websocket.py` - WebRTC signaling
- `backend/src/services/agent_orchestrator.py` - Pipeline coordinator
- `frontend/src/services/webrtc-client.js` - Browser WebRTC
- `specs/001-read-the-home/plan.md` - Implementation plan

## Development Commands
```bash
# Start backend
python backend/src/api/main.py

# Run tests (TDD - tests first!)
pytest backend/tests/contract/ -v

# Test agents
python backend/src/cli/agent_cli.py --test
```

## Current Focus
Implementing stateful agent orchestration with multi-image inference and pause/resume capability.

## Recent Changes
- [2025-09-11] Stateful orchestration with multi-image inference planned (v2.0)
- [2025-09-10] WebRTC protocol specification completed
- [2025-09-10] Multi-agent pipeline design finalized
- "Always use current existing venv"