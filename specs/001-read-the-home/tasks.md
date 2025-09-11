# Tasks: Intelligent Returns Classifier System - Backend APIs

**Input**: Design documents from `/specs/001-read-the-home/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → Tech stack: Python 3.11, FastAPI, aiortc, Ollama
   → Structure: Web app (backend/frontend separation)
2. Load optional design documents:
   → data-model.md: 6 entities to create
   → contracts/: REST API and WebSocket protocols
   → research.md: Technology decisions confirmed
3. Generate tasks by category:
   → Setup: FastAPI project, dependencies
   → Tests: Contract tests for all endpoints
   → Core: Models, WebSocket handler, REST endpoints
   → Integration: WebRTC, session management
   → Polish: Error handling, logging
4. Apply task rules:
   → Different files = mark [P] for parallel
   → Tests before implementation (TDD)
5. Number tasks sequentially (T001-T035)
6. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- Backend: `backend/src/`, `backend/tests/`
- Frontend: `frontend/src/` (future phase)
- All paths relative to repository root

## Phase 3.1: Setup
- [ ] T001 Create backend project structure (backend/src/{models,services,api,cli})
- [ ] T002 Initialize Python virtual environment and create backend/requirements.txt with FastAPI, aiortc, pydantic, pytest
- [ ] T003 [P] Configure pytest and create backend/pytest.ini
- [ ] T004 [P] Create backend/.env.example with OLLAMA_HOST, CAMERA_SOURCE, LOG_LEVEL

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

### Contract Tests - REST API
- [ ] T005 [P] Contract test POST /api/session/start in backend/tests/contract/test_session_start.py
- [ ] T006 [P] Contract test GET /api/session/{id}/status in backend/tests/contract/test_session_status.py
- [ ] T007 [P] Contract test POST /api/session/{id}/trigger in backend/tests/contract/test_session_trigger.py
- [ ] T008 [P] Contract test GET /api/session/{id}/results in backend/tests/contract/test_session_results.py
- [ ] T009 [P] Contract test POST /api/session/{id}/export in backend/tests/contract/test_session_export.py
- [ ] T010 [P] Contract test GET /api/health in backend/tests/contract/test_health.py

### Contract Tests - WebSocket
- [ ] T011 [P] WebSocket connection test in backend/tests/contract/test_websocket_connection.py
- [ ] T012 [P] WebRTC signaling test (offer/answer) in backend/tests/contract/test_webrtc_signaling.py
- [ ] T013 [P] Session message flow test in backend/tests/contract/test_websocket_messages.py

### Integration Tests
- [ ] T014 [P] Integration test automatic mode session lifecycle in backend/tests/integration/test_auto_mode.py
- [ ] T015 [P] Integration test manual mode with triggers in backend/tests/integration/test_manual_mode.py
- [ ] T016 [P] Integration test session recovery in backend/tests/integration/test_session_recovery.py

## Phase 3.3: Core Implementation (ONLY after tests are failing)

### Data Models
- [ ] T017 [P] ClothingItem model in backend/src/models/clothing_item.py
- [ ] T018 [P] ClassificationSession model in backend/src/models/session.py
- [ ] T019 [P] AgentResult model in backend/src/models/agent_result.py
- [ ] T020 [P] CameraFeed model in backend/src/models/camera_feed.py
- [ ] T021 [P] ProcessingStatus model in backend/src/models/processing_status.py
- [ ] T022 [P] FrameData model in backend/src/models/frame_data.py

### Core Services
- [ ] T023 SessionManager service in backend/src/services/session_manager.py
- [ ] T024 WebRTC manager in backend/src/services/webrtc_manager.py

### WebSocket Implementation
- [ ] T025 WebSocket connection handler in backend/src/api/websocket.py
- [ ] T026 WebRTC signaling implementation in backend/src/api/websocket.py (add to existing)
- [ ] T027 Message router for WebSocket types in backend/src/api/websocket.py (add to existing)

### REST API Endpoints
- [ ] T028 FastAPI app initialization in backend/src/api/main.py
- [ ] T029 POST /api/session/start endpoint in backend/src/api/routes/session.py
- [ ] T030 GET /api/session/{id}/status endpoint in backend/src/api/routes/session.py (add to existing)
- [ ] T031 POST /api/session/{id}/trigger endpoint in backend/src/api/routes/session.py (add to existing)
- [ ] T032 GET /api/session/{id}/results endpoint in backend/src/api/routes/session.py (add to existing)
- [ ] T033 POST /api/session/{id}/export endpoint in backend/src/api/routes/session.py (add to existing)
- [ ] T034 GET /api/health endpoint in backend/src/api/routes/health.py

## Phase 3.4: Integration
- [ ] T035 CORS configuration for frontend in backend/src/api/main.py (update existing)
- [ ] T036 WebSocket mount point configuration in backend/src/api/main.py (update existing)
- [ ] T037 Structured JSON logging setup in backend/src/services/logger.py
- [ ] T038 Session persistence to JSON files in backend/src/services/session_manager.py (update existing)
- [ ] T039 Error handling middleware in backend/src/api/middleware.py

## Phase 3.5: Frame-Pull Architecture Implementation
- [ ] T040 Add frame buffer to WebRTC Manager in backend/src/services/webrtc_manager.py (update existing)
- [ ] T041 Create Agent base class in backend/src/agents/base_agent.py
- [ ] T042 [P] Create InitialClassifier agent in backend/src/agents/initial_classifier.py
- [ ] T043 [P] Create DetailExtractor agent in backend/src/agents/detail_extractor.py
- [ ] T044 [P] Create DamageDetector agent in backend/src/agents/damage_detector.py
- [ ] T045 [P] Create FinalCompiler agent in backend/src/agents/final_compiler.py
- [ ] T046 Create Agent Orchestrator service in backend/src/services/agent_orchestrator.py
- [ ] T047 Enhance Camera Service for frame buffering in backend/src/services/camera_service.py (update existing)
- [ ] T048 Update WebSocket handler for agent triggers in backend/src/api/websocket.py (update existing)
- [ ] T049 [P] Frame pull contract test in backend/tests/contract/test_frame_pull.py
- [ ] T050 [P] Agent pipeline contract test in backend/tests/contract/test_agent_pipeline.py

## Phase 3.6: Polish
- [ ] T051 [P] API documentation generation in backend/src/api/main.py (add OpenAPI customization)
- [ ] T052 [P] Health check for Ollama connectivity in backend/src/api/routes/health.py (update existing)
- [ ] T053 WebSocket reconnection handling in backend/src/api/websocket.py (update existing)
- [ ] T054 Memory cleanup for completed sessions in backend/src/services/session_manager.py (update existing)
- [ ] T055 Run integration test suite and verify all pass

## Dependencies
- Setup (T001-T004) must complete first
- Tests (T005-T016) before implementation (T017-T034)
- Models (T017-T022) can run in parallel
- T023-T024 before T025-T027 (services before WebSocket)
- T028 before T029-T034 (app before routes)
- T025 before T026-T027 (WebSocket base before features)
- Integration (T035-T039) after core implementation
- Frame-Pull Architecture (T040-T050) after integration
- T040 before T041-T048 (frame buffer before agents)
- T041 before T042-T045 (base agent before specific agents)
- T046 after T042-T045 (orchestrator after individual agents)
- T049-T050 can run in parallel with implementation
- Polish (T051-T055) last

## Parallel Execution Examples

### Launch all contract tests together (T005-T013):
```bash
# Run in separate terminals or with & operator
Task: "Contract test POST /api/session/start in backend/tests/contract/test_session_start.py"
Task: "Contract test GET /api/session/{id}/status in backend/tests/contract/test_session_status.py"
Task: "Contract test POST /api/session/{id}/trigger in backend/tests/contract/test_session_trigger.py"
Task: "Contract test GET /api/session/{id}/results in backend/tests/contract/test_session_results.py"
Task: "Contract test POST /api/session/{id}/export in backend/tests/contract/test_session_export.py"
Task: "Contract test GET /api/health in backend/tests/contract/test_health.py"
Task: "WebSocket connection test in backend/tests/contract/test_websocket_connection.py"
Task: "WebRTC signaling test in backend/tests/contract/test_webrtc_signaling.py"
Task: "Session message flow test in backend/tests/contract/test_websocket_messages.py"
```

### Launch all models together (T017-T022):
```bash
Task: "ClothingItem model in backend/src/models/clothing_item.py"
Task: "ClassificationSession model in backend/src/models/session.py"
Task: "AgentResult model in backend/src/models/agent_result.py"
Task: "CameraFeed model in backend/src/models/camera_feed.py"
Task: "ProcessingStatus model in backend/src/models/processing_status.py"
Task: "FrameData model in backend/src/models/frame_data.py"
```

### Launch all agent implementations together (T042-T045):
```bash
Task: "Create InitialClassifier agent in backend/src/agents/initial_classifier.py"
Task: "Create DetailExtractor agent in backend/src/agents/detail_extractor.py"
Task: "Create DamageDetector agent in backend/src/agents/damage_detector.py"
Task: "Create FinalCompiler agent in backend/src/agents/final_compiler.py"
```

## Notes
- Focus on backend API and WebSocket first (frontend integration in next phase)
- WebRTC implementation includes signaling only (video processing in next phase)
- Session management is in-memory with JSON persistence
- All tests must fail before implementation (TDD strict)
- Commit after each task completion
- WebSocket and REST API work together for complete functionality
- Frame-pull architecture ensures smooth video streaming with on-demand AI processing

## Frame-Pull Architecture Details
- **T040**: Add frame buffering to WebRTC Manager to store latest frame per session
- **T041**: Base Agent class with common frame pulling and VLM integration logic
- **T042-T045**: Individual agents that pull fresh frames on trigger and analyze with Qwen2.5-VL
- **T046**: Agent Orchestrator manages sequential agent execution and result compilation
- **T047**: Camera Service stores frames in buffer while maintaining video stream
- **T048**: WebSocket handler processes agent triggers and returns results via data channels

## Next Phase Preview
After backend APIs and frame-pull architecture complete:
- Frontend WebRTC client implementation
- VLM service connection via Ollama (Qwen2.5-VL 3B)
- Progressive result streaming via data channels
- Manual and automatic trigger modes
- Session recovery and export functionality