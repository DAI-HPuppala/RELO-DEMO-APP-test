# Feature Specification: Stateful Agent Orchestration with Multi-Image Inference

**Feature Branch**: `002-refactor-the-orchestration`  
**Created**: 2025-09-11  
**Status**: Draft  
**Input**: User description: "Refactor the orchestration so that it is stateful and applies the same rules to all agents (initial, details, damage). When an agent is triggered, start its timer. The agent should first capture one fresh frame from the WebRTC pipeline and run a single-image inference with the VLM using the existing base prompt, returning a JSON. After that, as long as there is still more than 1 second left on the timer, the agent must repeat the following: capture a new fresh frame, combine it with all frames used in previous inferences, and send the entire set to the VLM as a batch (multi-image) inference. For these later runs, extend the base prompt with context such as: 'you are seeing the same garment from a different side.' This loop continues until the timer ends. Once the timer expires, aggregate all inference results: if the agent made 3 or more inferences, finalize each attribute by choosing the most consistent value across all results; if only 1 or 2 inferences were made, use the last inference's values, but if the last has any null attributes and the first has values for them, fall back to the first inference for those fields. Every agent must follow this same process. When all three agents finish, the Final compiler agent should gather their finalized outputs — type, color, pattern, neckline, sleeve, and closure from the initial agent; size and brand from the details agent; and is-damaged? plus damage type from the damage agent. It should also aggregate all reasoning responses in sequence, e.g., 1. reasoning of inference 1 from initial agent, 2. reasoning of inference 2 from initial agent, 3. reasoning of inference 1 from details agent, 4. reasoning of inference 2 from damage agent, and so on. The final compiler agent outputs a single JSON with all finalized attributes (type, color, pattern, neckline, sleeve, closure, brand, size, is-damaged, damage type) plus the aggregated reasoning. Log all inferences of each agent separately. Ensure that frames captured by any agent are not reused by other agents, with one exception: the very first frame captured by the initial agent must be stored and passed to the damage agent. For the damage agent's first inference, this stored initial frame should be combined with a new fresh frame it captures, making its very first inference a batch (multi-image) inference rather than a single-image inference. For the damage agent's subsequent inferences, it should then include both of these frames along with each newly captured frame, so the set grows (first inference = [initial agent's first frame + damage agent's first fresh frame], second inference = [those two frames + next fresh frame], third inference = [all previous frames + next fresh frame], and so on) until the timer ends. For every other agent, the first inference is always single-image, and all later inferences are multi-image batches as described."

## Execution Flow (main)
```
1. Parse user description from Input
   → If empty: ERROR "No feature description provided"
2. Extract key concepts from description
   → Identify: actors, actions, data, constraints
3. For each unclear aspect:
   → Mark with [NEEDS CLARIFICATION: specific question]
4. Fill User Scenarios & Testing section
   → If no clear user flow: ERROR "Cannot determine user scenarios"
5. Generate Functional Requirements
   → Each requirement must be testable
   → Mark ambiguous requirements
6. Identify Key Entities (if data involved)
7. Run Review Checklist
   → If any [NEEDS CLARIFICATION]: WARN "Spec has uncertainties"
   → If implementation details found: ERROR "Remove tech details"
8. Return: SUCCESS (spec ready for planning)
```

---

## ⚡ Quick Guidelines
- ✅ Focus on WHAT users need and WHY
- ❌ Avoid HOW to implement (no tech stack, APIs, code structure)
- 👥 Written for business stakeholders, not developers

### Section Requirements
- **Mandatory sections**: Must be completed for every feature
- **Optional sections**: Include only when relevant to the feature
- When a section doesn't apply, remove it entirely (don't leave as "N/A")

### For AI Generation
When creating this spec from a user prompt:
1. **Mark all ambiguities**: Use [NEEDS CLARIFICATION: specific question] for any assumption you'd need to make
2. **Don't guess**: If the prompt doesn't specify something (e.g., "login system" without auth method), mark it
3. **Think like a tester**: Every vague requirement should fail the "testable and unambiguous" checklist item
4. **Common underspecified areas**:
   - User types and permissions
   - Data retention/deletion policies  
   - Performance targets and scale
   - Error handling behaviors
   - Integration requirements
   - Security/compliance needs

---

## User Scenarios & Testing *(mandatory)*

### Primary User Story
As a warehouse operator processing clothing returns, I need the system to perform comprehensive multi-angle classification of each garment item. The system should automatically progress through initial classification, detail extraction, and damage detection phases, capturing multiple views of the garment during each phase to ensure accurate classification. Each classification agent should have enough time to analyze the garment from different angles, and the results should be aggregated intelligently to provide the most accurate final classification.

### Acceptance Scenarios
1. **Given** an operator initiates automatic classification mode with a garment in view, **When** the system runs the initial classification agent, **Then** the agent captures and analyzes multiple frames within its timer period, aggregating results to determine garment type, color, pattern, neckline, sleeve type, and closure type

2. **Given** the initial classification agent has completed, **When** the detail extraction agent runs, **Then** it captures fresh frames (not reusing previous frames) and analyzes them progressively to determine brand and size information

3. **Given** the initial classification agent has captured its first frame, **When** the damage detection agent runs, **Then** it uses the initial agent's first frame combined with its own fresh captures to detect damage from multiple angles

4. **Given** all three primary agents have completed their analysis, **When** the final compiler agent runs, **Then** it aggregates all finalized attributes and reasoning from all agents into a comprehensive classification result

5. **Given** an agent has performed multiple inferences within its timer, **When** results are aggregated, **Then** the system selects the most consistent values across all inferences (if 3+ inferences) or uses intelligent fallback rules (if 1-2 inferences)

### Edge Cases
- What happens when an agent only manages 1 inference within its timer period? Answer: Use that inference's results as the finalized attributes and reasoning.
- How does the system handle null/missing attributes in inference results? Answer: If null, for both the inferences, keep the final attribute as null. If null for second inference, use the first inference's attribute. If null for first inference, use the second inference's attribute.
- What occurs if frame capture fails during an agent's processing? Answer: The agent should retry to capture frames and perform inferences until the timer expires, and throw an error to the user in frontend and then continue to the next agent.
- How does the system respond when conflicting values appear across multiple inferences? Answer: The system should select the most consistent value across all inferences (if 3+ inferences) or if it is 1 inference then that response is the final attribute, and in case of 2 inferences then the system should use the last inference's values but fall back to first inference for any null attributes.
- What happens if the timer expires mid-inference? Answer: The agent should continue to perform inferences and complete it.

## Requirements *(mandatory)*

### Functional Requirements
- **FR-001**: System MUST provide a stateful orchestration mechanism that applies consistent processing rules to all classification agents
- **FR-002**: Each agent MUST operate with a configurable timer that controls its inference period
- **FR-003**: Each agent MUST capture fresh frames from the live video stream for each inference cycle
- **FR-004**: Agents MUST perform single-image inference for their first analysis, then multi-image batch inference for subsequent analyses within their timer period
- **FR-005**: System MUST prevent frame reuse between agents, except for the specific case where the damage agent receives the initial agent's first frame
- **FR-006**: Each agent MUST continue capturing and processing frames as long as more than 1 second remains on its timer
- **FR-007**: System MUST aggregate inference results using defined rules: most consistent value for 3+ inferences, or intelligent fallback for 1-2 inferences
- **FR-008**: Final compiler agent MUST gather specific attributes from each agent: type/color/pattern/neckline/sleeve/closure from initial, size/brand from details, damage status/type from damage detection
- **FR-009**: Final compiler MUST aggregate all reasoning responses in sequential order from all agents and inferences
- **FR-010**: System MUST log each inference separately with clear identification of agent name and inference number
- **FR-011**: Multi-image inferences MUST include contextual prompt additions indicating the images show the same garment from different angles
- **FR-012**: Damage detection agent MUST always perform multi-image inference starting from its first inference (using initial agent's first frame plus its own capture)
- **FR-013**: System MUST handle attribute conflicts by selecting the most frequently occurring value across all inferences when 3+ inferences are available
- **FR-014**: When only 1-2 inferences are available, system MUST use the last inference's values but fall back to first inference for any null attributes
- **FR-015**: Final output MUST be a single structured data format containing all finalized attributes and aggregated reasoning

### Key Entities
- **Classification Agent**: Represents an individual processing unit (initial, details, damage) with its own timer, inference logic, and result aggregation rules
- **Inference Result**: Contains attributes detected in a single inference cycle, including confidence scores and reasoning
- **Agent Timer**: Controls the duration of each agent's processing window with configurable time limits
- **Frame Set**: Collection of captured frames used for multi-image inference, growing with each cycle
- **Final Classification**: Aggregated result containing all finalized attributes from all agents plus combined reasoning
- **Shared Frame**: The specific first frame from initial agent that is passed to damage agent

---

## Review & Acceptance Checklist
*GATE: Automated checks run during main() execution*

### Content Quality
- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

### Requirement Completeness
- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous  
- [x] Success criteria are measurable
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

---

## Execution Status
*Updated by main() during processing*

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [x] Review checklist passed

---