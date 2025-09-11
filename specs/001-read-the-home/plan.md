# Implementation Plan: Intelligent Returns Classifier System

**Branch**: `001-read-the-home` | **Date**: 2025-09-10 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-read-the-home/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → Successfully loaded spec for Intelligent Returns Classifier System
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detected Project Type: web (frontend+backend with WebRTC streaming)
   → Structure Decision: Option 2 (Web application)
3. Evaluate Constitution Check section below
   → No violations detected - modular architecture with clear separation
   → Update Progress Tracking: Initial Constitution Check
4. Execute Phase 0 → research.md
   → Research WebRTC implementation patterns
   → Research VLM integration approaches
   → Research multi-agent orchestration
5. Execute Phase 1 → contracts, data-model.md, quickstart.md, CLAUDE.md
6. Re-evaluate Constitution Check section
   → Verify library-based approach maintained
   → Update Progress Tracking: Post-Design Constitution Check
7. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
8. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
Building an intelligent returns classifier system that uses AI vision to automatically classify returned clothing items. The system captures live video via WebRTC, processes frames through multiple specialized AI agents (using Qwen2.5-VL 3B model), and provides real-time progressive results with both automatic and manual operating modes.

## Technical Context
**Language/Version**: Python 3.11 (backend), HTML5/CSS3/JavaScript ES6+ (frontend)  
**Primary Dependencies**: 
  - Backend: FastAPI, aiortc (WebRTC), Ollama (VLM hosting), OpenCV, Pillow
  - Frontend: WebRTC API, vanilla JavaScript (no framework for simplicity)
**Storage**: In-memory during sessions, JSON logs for completed sessions  
**Testing**: pytest (backend), Jest (frontend JavaScript)  
**Target Platform**: Linux IPC (Industrial PC), browser-based frontend  
**Project Type**: web - Frontend UI + Python backend with WebRTC  
**Performance Goals**: 
  - Frame inference: 0.3-0.5 seconds
  - Pipeline completion: <15 seconds
  - WebRTC latency: <100ms
  - Near 100% accuracy for attribute detection
**Constraints**: 
  - No external services/APIs (all local)
  - GPU memory: 4-6GB for VLM
  - Single camera feed at a time
  - Automatic frame deletion after processing
**Scale/Scope**: 
  - Single operator interface
  - Continuous operation during business hours
  - Session-based processing with UUID tracking

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Simplicity**:
- Projects: 2 (frontend, backend) ✓
- Using framework directly? Yes - FastAPI without wrappers ✓
- Single data model? Yes - shared between agents ✓
- Avoiding patterns? Yes - direct services, no unnecessary abstractions ✓

**Architecture**:
- EVERY feature as library? Yes - modular agent system ✓
- Libraries listed:
  - webrtc_manager: WebRTC connection and stream handling
  - frame_processor: Frame extraction and queueing
  - vlm_service: Qwen2.5-VL model interface
  - agent_orchestrator: Multi-agent pipeline coordination
  - session_manager: UUID-based session tracking
- CLI per library: Each exposes CLI for testing/debugging
- Library docs: llms.txt format planned ✓

**Testing (NON-NEGOTIABLE)**:
- RED-GREEN-Refactor cycle enforced? Yes ✓
- Git commits show tests before implementation? Will enforce ✓
- Order: Contract→Integration→E2E→Unit strictly followed? Yes ✓
- Real dependencies used? Yes - actual camera, GPU, model ✓
- Integration tests for: WebRTC connections, agent pipeline, VLM inference ✓
- FORBIDDEN: Implementation before test - understood ✓

**Observability**:
- Structured logging included? Yes - JSON structured logs ✓
- Frontend logs → backend? Yes - via WebRTC data channel ✓
- Error context sufficient? Yes - frame ID, agent, timestamp ✓

**Versioning**:
- Version number assigned? 1.0.0 initial ✓
- BUILD increments on every change? Will implement ✓
- Breaking changes handled? N/A for initial version ✓

## Project Structure

### Documentation (this feature)
```
specs/001-read-the-home/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
# Option 2: Web application (frontend + backend)
backend/
├── src/
│   ├── models/
│   │   ├── clothing_item.py
│   │   ├── session.py
│   │   └── agent_result.py
│   ├── services/
│   │   ├── webrtc_manager.py
│   │   ├── frame_processor.py
│   │   ├── vlm_service.py
│   │   └── agent_orchestrator.py
│   ├── agents/
│   │   ├── initial_classifier.py
│   │   ├── detail_extractor.py
│   │   ├── damage_detector.py
│   │   └── final_compiler.py
│   ├── api/
│   │   ├── main.py
│   │   ├── websocket.py
│   │   └── routes.py
│   └── cli/
│       └── agent_cli.py
└── tests/
    ├── contract/
    ├── integration/
    └── unit/

frontend/
├── src/
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   ├── components/
│   │   ├── video-display.js
│   │   ├── progress-panel.js
│   │   ├── results-panel.js
│   │   └── control-panel.js
│   └── services/
│       ├── webrtc-client.js
│       └── session-manager.js
└── tests/
    └── components/
```

**Structure Decision**: Option 2 - Web application with frontend and backend separation

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context**:
   - WebRTC implementation with aiortc for Python
   - Ollama integration for Qwen2.5-VL 3B model
   - Multi-agent orchestration patterns
   - Progressive result streaming via WebRTC data channels
   - Camera compatibility (RealSense, Zebra CV60, webcams)

2. **Generate and dispatch research agents**:
   ```
   Task: "Research WebRTC implementation with aiortc for video streaming"
   Task: "Find best practices for Ollama VLM integration in Python"
   Task: "Research multi-agent coordination patterns for sequential processing"
   Task: "Investigate WebRTC data channel for real-time updates"
   Task: "Research camera API compatibility (RealSense, Zebra, webcam)"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: Technology choice and approach
   - Rationale: Why this approach fits requirements
   - Alternatives considered: Other options evaluated

**Output**: research.md with implementation decisions documented

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

1. **Extract entities from feature spec** → `data-model.md`:
   - ClothingItem: type, color, pattern, brand, size, condition
   - ClassificationSession: UUID, timestamp, status, results
   - AgentResult: agent_name, attributes, confidence, reasoning
   - CameraFeed: source, resolution, fps, status
   - ProcessingStatus: current_agent, timer, progress

2. **Generate API contracts** from functional requirements:
   - WebSocket: /ws/stream (WebRTC signaling and data)
   - POST: /api/session/start (initialize classification)
   - GET: /api/session/{id}/status (current progress)
   - POST: /api/session/{id}/trigger (manual mode trigger)
   - GET: /api/session/{id}/results (final classification)
   - POST: /api/session/{id}/export (JSON export)
   - Output WebSocket protocol and REST OpenAPI to `/contracts/`

3. **Generate contract tests** from contracts:
   - test_webrtc_connection.py
   - test_session_lifecycle.py
   - test_agent_pipeline.py
   - test_result_streaming.py

4. **Extract test scenarios** from user stories:
   - Automatic mode full cycle test
   - Manual mode with operator control
   - Camera disconnection recovery
   - Conflicting results resolution
   - Session resumption in manual mode

5. **Update CLAUDE.md incrementally**:
   - Add WebRTC, Ollama, multi-agent architecture
   - Document key patterns and conventions
   - Keep concise for token efficiency

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, CLAUDE.md

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Frontend foundation tasks (HTML, CSS, basic JS structure) [P]
- WebRTC client implementation tasks
- Backend WebSocket and API setup tasks [P]
- VLM service integration tasks
- Individual agent implementation tasks [P]
- Agent orchestration pipeline tasks
- Progressive result streaming tasks
- Session management tasks
- Integration testing tasks
- End-to-end testing tasks

**Ordering Strategy**:
- Tests first (TDD approach)
- Infrastructure before features
- Backend services before frontend integration
- Individual agents before orchestration
- Mark [P] for parallel execution where possible

**Estimated Output**: 30-35 numbered, ordered tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)  
**Phase 4**: Implementation (execute tasks.md following constitutional principles)  
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
*No violations - system follows constitutional principles*

## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented (none)

---
*Based on Constitution v2.1.1 - See `/memory/constitution.md`*