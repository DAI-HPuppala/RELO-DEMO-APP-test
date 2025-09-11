# Tasks: Stateful Agent Orchestration with Multi-Image Inference

**Input**: Design documents from `/specs/002-refactor-the-orchestration/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/websocket-api-v2.json

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → If not found: ERROR "No implementation plan found"
   → Extract: tech stack, libraries, structure
2. Load optional design documents:
   → data-model.md: Extract entities → model tasks
   → contracts/: Each file → contract test task
   → research.md: Extract decisions → setup tasks
3. Generate tasks by category:
   → Setup: project init, dependencies, linting
   → Tests: contract tests, integration tests
   → Core: models, services, CLI commands
   → Integration: DB, middleware, logging
   → Polish: unit tests, performance, docs
4. Apply task rules:
   → Different files = mark [P] for parallel
   → Same file = sequential (no [P])
   → Tests before implementation (TDD)
5. Number tasks sequentially (T001, T002...)
6. Generate dependency graph
7. Create parallel execution examples
8. Validate task completeness:
   → All contracts have tests?
   → All entities have models?
   → All endpoints implemented?
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- **Web app structure**: `backend/src/`, `frontend/src/`
- All paths are relative to repository root

## Phase 3.1: Setup & Configuration
- [ ] T001 Create v2 directory structure for stateful components in backend/src/
- [ ] T002 Add asyncio semaphore configuration for GPU serialization
- [ ] T003 [P] Create session state persistence directory structure at backend/sessions/
- [ ] T004 [P] Configure frame storage directory at backend/captured_frames/

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

### Contract Tests (WebSocket API v2)
- [ ] T005 [P] Contract test for pause_flow message in backend/tests/contract/test_websocket_pause_flow.py
- [ ] T006 [P] Contract test for resume_flow message in backend/tests/contract/test_websocket_resume_flow.py
- [ ] T007 [P] Contract test for inference_update message in backend/tests/contract/test_websocket_inference_update.py
- [ ] T008 [P] Contract test for agent_started with timer in backend/tests/contract/test_websocket_agent_started.py
- [ ] T009 [P] Contract test for flow state transitions in backend/tests/contract/test_websocket_flow_states.py

### Integration Tests (From Quickstart Scenarios)
- [ ] T010 [P] Integration test for multi-inference flow in backend/tests/integration/test_multi_inference_flow.py
- [ ] T011 [P] Integration test for pause/resume during processing in backend/tests/integration/test_pause_resume.py
- [ ] T012 [P] Integration test for frame sharing between agents in backend/tests/integration/test_frame_sharing.py
- [ ] T013 [P] Integration test for single inference fallback in backend/tests/integration/test_single_inference_fallback.py
- [ ] T014 [P] Integration test for timer expiry handling in backend/tests/integration/test_timer_expiry.py

## Phase 3.3: Core Implementation (ONLY after tests are failing)

### Data Models (From data-model.md)
- [ ] T015 [P] Create AgentState model in backend/src/models/agent_state.py
- [ ] T016 [P] Create InferenceResult model in backend/src/models/inference_result.py
- [ ] T017 [P] Create SessionState model in backend/src/models/session_state.py
- [ ] T018 [P] Create Frame model in backend/src/models/frame.py
- [ ] T019 [P] Create AggregatedResult model in backend/src/models/aggregated_result.py
- [ ] T020 [P] Create FinalClassification model in backend/src/models/final_classification.py

### Core Services (From research.md decisions)
- [ ] T021 Create FrameRegistry service in backend/src/services/frame_registry.py
- [ ] T022 Create StatefulOrchestrator service in backend/src/services/stateful_orchestrator.py
- [ ] T023 Create MultiInferenceEngine service in backend/src/services/multi_inference_engine.py
- [ ] T024 Create FrameAggregator service in backend/src/services/frame_aggregator.py
- [ ] T025 Create SessionStateManager service in backend/src/services/session_state_manager.py

### Stateful Base Agent
- [ ] T026 Create StatefulBaseAgent class in backend/src/agents/stateful_base_agent.py
- [ ] T027 Implement timer-based inference loop in StatefulBaseAgent
- [ ] T028 Implement frame capture with retry logic in StatefulBaseAgent
- [ ] T029 Implement aggregation rules (1/2/3+ inferences) in StatefulBaseAgent

### Agent Refactoring
- [ ] T030 Refactor InitialClassifier to extend StatefulBaseAgent in backend/src/agents/initial_classifier_v2.py
- [ ] T031 Refactor DetailExtractor to extend StatefulBaseAgent in backend/src/agents/detail_extractor_v2.py
- [ ] T032 Refactor DamageDetector with frame sharing in backend/src/agents/damage_detector_v2.py
- [ ] T033 Refactor FinalCompiler with reasoning aggregation in backend/src/agents/final_compiler_v2.py

### WebSocket Protocol Enhancement
- [ ] T034 Update WebSocket handler for pause/resume in backend/src/api/websocket_v2.py
- [ ] T035 Implement inference_update messages in websocket_v2.py
- [ ] T036 Add timer information to agent_started messages in websocket_v2.py
- [ ] T037 Implement flow state management in websocket_v2.py

### Multi-Image Inference Support
- [ ] T038 Implement single-image first inference in MultiInferenceEngine
- [ ] T039 Implement multi-image batch inference with context prompt in MultiInferenceEngine
- [ ] T040 Implement frame accumulation logic in MultiInferenceEngine
- [ ] T041 Add GPU semaphore for serialized access in MultiInferenceEngine

## Phase 3.4: Integration

### Frame Management
- [ ] T042 Integrate FrameRegistry with WebRTC frame provider
- [ ] T043 Implement frame sharing protocol for initial→damage
- [ ] T044 Add frame cleanup and garbage collection
- [ ] T045 Implement debug frame saving to captured_frames/

### State Persistence
- [ ] T046 Implement JSON checkpointing in SessionStateManager
- [ ] T047 Add crash recovery from checkpoints
- [ ] T048 Implement session state queries for debugging
- [ ] T049 Add state transition logging

### Agent Orchestration
- [ ] T050 Wire StatefulOrchestrator with new agent implementations
- [ ] T051 Implement pause/resume state machine
- [ ] T052 Add progress callbacks for frontend updates
- [ ] T053 Implement custom timer configuration support

## Phase 3.5: Frontend Updates

### Components
- [ ] T054 [P] Create PauseResumeControl component in frontend/src/components/pause-resume-control.js
- [ ] T055 [P] Update AgentMonitor for multi-inference display in frontend/src/components/agent-monitor-v2.js
- [ ] T056 [P] Create InferenceProgressBar component in frontend/src/components/inference-progress-bar.js

### WebRTC Client Updates
- [ ] T057 Update WebRTC client for v2 protocol in frontend/src/services/webrtc-client-v2.js
- [ ] T058 Add pause/resume message handlers in webrtc-client-v2.js
- [ ] T059 Handle inference_update messages in webrtc-client-v2.js
- [ ] T060 Add custom timer configuration UI support

## Phase 3.6: Polish & Testing

### Performance Tests
- [ ] T061 [P] Test GPU memory usage with 5-image batches in backend/tests/performance/test_gpu_memory.py
- [ ] T062 [P] Test inference timing accuracy in backend/tests/performance/test_timer_accuracy.py
- [ ] T063 [P] Test frame capture latency in backend/tests/performance/test_frame_latency.py

### Unit Tests
- [ ] T064 [P] Unit tests for aggregation logic in backend/tests/unit/test_aggregation.py
- [ ] T065 [P] Unit tests for frame registry in backend/tests/unit/test_frame_registry.py
- [ ] T066 [P] Unit tests for state transitions in backend/tests/unit/test_state_transitions.py

### Documentation & Cleanup
- [ ] T067 [P] Update API documentation with v2 protocol
- [ ] T068 [P] Create migration guide from v1 to v2
- [ ] T069 Remove deprecated v1 code paths
- [ ] T070 Run full quickstart.md validation

## Dependencies
- Setup (T001-T004) must complete first
- Tests (T005-T014) before ANY implementation
- Models (T015-T020) before services (T021-T025)
- StatefulBaseAgent (T026-T029) before agent refactoring (T030-T033)
- Core services before integration (T042-T053)
- Backend complete before frontend (T054-T060)
- All implementation before polish (T061-T070)

## Parallel Execution Examples

### Parallel Group 1: Contract Tests (after setup)
```bash
# Launch T005-T009 together:
Task: "Contract test for pause_flow message in backend/tests/contract/test_websocket_pause_flow.py"
Task: "Contract test for resume_flow message in backend/tests/contract/test_websocket_resume_flow.py"
Task: "Contract test for inference_update message in backend/tests/contract/test_websocket_inference_update.py"
Task: "Contract test for agent_started with timer in backend/tests/contract/test_websocket_agent_started.py"
Task: "Contract test for flow state transitions in backend/tests/contract/test_websocket_flow_states.py"
```

### Parallel Group 2: Integration Tests
```bash
# Launch T010-T014 together:
Task: "Integration test for multi-inference flow in backend/tests/integration/test_multi_inference_flow.py"
Task: "Integration test for pause/resume during processing in backend/tests/integration/test_pause_resume.py"
Task: "Integration test for frame sharing between agents in backend/tests/integration/test_frame_sharing.py"
Task: "Integration test for single inference fallback in backend/tests/integration/test_single_inference_fallback.py"
Task: "Integration test for timer expiry handling in backend/tests/integration/test_timer_expiry.py"
```

### Parallel Group 3: Data Models
```bash
# Launch T015-T020 together:
Task: "Create AgentState model in backend/src/models/agent_state.py"
Task: "Create InferenceResult model in backend/src/models/inference_result.py"
Task: "Create SessionState model in backend/src/models/session_state.py"
Task: "Create Frame model in backend/src/models/frame.py"
Task: "Create AggregatedResult model in backend/src/models/aggregated_result.py"
Task: "Create FinalClassification model in backend/src/models/final_classification.py"
```

### Parallel Group 4: Frontend Components
```bash
# Launch T054-T056 together:
Task: "Create PauseResumeControl component in frontend/src/components/pause-resume-control.js"
Task: "Update AgentMonitor for multi-inference display in frontend/src/components/agent-monitor-v2.js"
Task: "Create InferenceProgressBar component in frontend/src/components/inference-progress-bar.js"
```

## Notes
- [P] tasks work on different files and have no dependencies
- Verify ALL tests fail before implementing (RED phase of TDD)
- Commit after each task with descriptive message
- GPU serialization is critical - test thoroughly
- Frame sharing protocol must be validated early
- Timer accuracy affects inference count - monitor closely

## Validation Checklist
*GATE: Checked before execution*

- [x] All WebSocket contract messages have tests (T005-T009)
- [x] All data model entities have creation tasks (T015-T020)
- [x] All quickstart scenarios have integration tests (T010-T014)
- [x] Tests come before implementation (Phase 3.2 before 3.3)
- [x] Parallel tasks work on different files
- [x] Each task specifies exact file path
- [x] No [P] task modifies same file as another [P] task

## Estimated Effort
- **Total Tasks**: 70
- **Parallel Groups**: 12 groups with 3-6 tasks each
- **Critical Path**: Setup → Tests → Base Agent → Services → Integration
- **Estimated Time**: 3-4 days with parallel execution