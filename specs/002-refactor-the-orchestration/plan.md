# Implementation Plan: Stateful Agent Orchestration with Multi-Image Inference

**Branch**: `002-refactor-the-orchestration` | **Date**: 2025-09-11 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-refactor-the-orchestration/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
4. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
5. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file (e.g., `CLAUDE.md` for Claude Code, `.github/copilot-instructions.md` for GitHub Copilot, or `GEMINI.md` for Gemini CLI).
6. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
7. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
8. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
Refactor the agent orchestration system to be stateful, allowing each agent to perform multiple inferences within its timer period. The system captures fresh frames progressively, performs both single-image and multi-image batch inferences, and aggregates results intelligently. Special frame sharing occurs between initial and damage agents. Includes pause/resume functionality for the agent flow in automatic mode.

## Technical Context
**Language/Version**: Python 3.11  
**Primary Dependencies**: FastAPI, aiortc (WebRTC), Ollama (VLM), asyncio, numpy, opencv-python  
**Storage**: In-memory state management with JSON persistence for session recovery  
**Testing**: pytest with async support  
**Target Platform**: Linux server with GPU acceleration  
**Project Type**: web (frontend+backend architecture)  
**Performance Goals**: <1s inference latency per agent, 30fps frame capture rate  
**Constraints**: GPU-exclusive for AI processing, CPU for WebRTC, <5s total per agent timer  
**Scale/Scope**: Single session per instance, 4 agents (initial, details, damage, final)

**Additional Context from User**:
- Stateful orchestration where next agents know previous agent responses (final agent sees all)
- WebRTC runs on CPU providing fresh frames on demand
- All AI inferences and agent processing solely on GPU using full resources
- Aggregation runs when agent timer <1s or timer completes after delayed response
- Pause/resume feature to stop agent flow mid-process (except during final compiler)
- Each session has its own state, one session = one cycle (initial→details→damage→final)

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Simplicity**:
- Projects: 2 (backend, frontend)
- Using framework directly? YES (FastAPI, aiortc without wrappers)
- Single data model? YES (shared agent result schema)
- Avoiding patterns? YES (direct service calls, no unnecessary abstractions)

**Architecture**:
- EVERY feature as library? YES (agent orchestration as library)
- Libraries listed: 
  - `stateful_orchestrator`: Manages agent lifecycle with state persistence
  - `multi_inference_engine`: Handles single/batch image processing
  - `frame_aggregator`: Intelligent result aggregation logic
  - `session_state_manager`: Pause/resume and state tracking
- CLI per library: Each library exposes CLI for testing/debugging
- Library docs: llms.txt format planned? YES

**Testing (NON-NEGOTIABLE)**:
- RED-GREEN-Refactor cycle enforced? YES
- Git commits show tests before implementation? YES
- Order: Contract→Integration→E2E→Unit strictly followed? YES
- Real dependencies used? YES (actual Ollama, WebRTC connections)
- Integration tests for: new libraries, contract changes, shared schemas? YES
- FORBIDDEN: Implementation before test, skipping RED phase ✓

**Observability**:
- Structured logging included? YES
- Frontend logs → backend? YES (via WebSocket)
- Error context sufficient? YES (frame IDs, inference numbers, agent states)

**Versioning**:
- Version number assigned? 2.0.0 (major refactor)
- BUILD increments on every change? YES
- Breaking changes handled? YES (new WebSocket protocol, migration plan)

## Project Structure

### Documentation (this feature)
```
specs/002-refactor-the-orchestration/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
# Option 2: Web application (frontend + backend detected)
backend/
├── src/
│   ├── models/
│   │   ├── agent_state.py      # Stateful agent models
│   │   └── inference_result.py # Multi-inference results
│   ├── services/
│   │   ├── stateful_orchestrator.py
│   │   ├── multi_inference_engine.py
│   │   ├── frame_aggregator.py
│   │   └── session_state_manager.py
│   ├── agents/
│   │   ├── stateful_base_agent.py
│   │   ├── initial_classifier_v2.py
│   │   ├── detail_extractor_v2.py
│   │   ├── damage_detector_v2.py
│   │   └── final_compiler_v2.py
│   └── api/
│       └── websocket_v2.py     # Enhanced WebSocket protocol
└── tests/
    ├── contract/
    ├── integration/
    └── unit/

frontend/
├── src/
│   ├── components/
│   │   ├── agent-monitor-v2.js
│   │   └── pause-resume-control.js
│   └── services/
│       └── webrtc-client-v2.js
└── tests/
```

**Structure Decision**: Option 2 (Web application) - Based on existing frontend+backend architecture

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context**:
   - GPU resource allocation strategy for Ollama
   - Optimal batch size for multi-image inference
   - Frame synchronization between CPU (WebRTC) and GPU (inference)
   - State persistence strategy for pause/resume

2. **Generate and dispatch research agents**:
   ```
   Task: "Research GPU memory management for Ollama batch inference"
   Task: "Find best practices for WebRTC frame buffering without blocking"
   Task: "Research asyncio patterns for timer-based state machines"
   Task: "Investigate Ollama multi-image prompt strategies"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all technical decisions documented

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

1. **Extract entities from feature spec** → `data-model.md`:
   - AgentState: timer, inference_count, frames_collected, status
   - InferenceResult: attributes, confidence, reasoning, frame_ids
   - SessionState: agents_completed, paused_at, shared_frames
   - AggregatedResult: finalized_attributes, aggregation_method

2. **Generate API contracts** from functional requirements:
   - WebSocket messages for pause/resume
   - Enhanced progress updates with inference counts
   - New data channel protocol for multi-inference results
   - Output OpenAPI/AsyncAPI schema to `/contracts/`

3. **Generate contract tests** from contracts:
   - Test stateful agent progression
   - Test pause/resume at different stages
   - Test frame sharing between agents
   - Test aggregation rules (3+ vs 1-2 inferences)

4. **Extract test scenarios** from user stories:
   - Complete automatic flow with multiple inferences
   - Pause during detail agent, resume and complete
   - Single inference fallback scenario
   - Frame sharing validation between initial and damage

5. **Update agent file incrementally**:
   - Run `/scripts/update-agent-context.sh claude`
   - Add stateful orchestration patterns
   - Document multi-image inference approach
   - Update recent changes with this refactor

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, CLAUDE.md

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Generate tasks for stateful agent refactoring
- Each agent needs timer management tasks
- Frame aggregation service tasks
- Pause/resume mechanism tasks
- WebSocket protocol enhancement tasks
- Frontend control integration tasks

**Ordering Strategy**:
- Models and state management first
- Base agent refactoring before specific agents
- Backend services before API changes
- Frontend updates after backend ready
- Integration tests throughout

**Estimated Output**: 30-35 numbered, ordered tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)  
**Phase 4**: Implementation (execute tasks.md following constitutional principles)  
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
*No violations - design follows constitutional principles*

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