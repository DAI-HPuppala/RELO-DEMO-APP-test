/**
 * Agent Monitor Component V2
 * Displays real-time agent status with timers and inference counters
 */

class AgentMonitorV2 extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        
        // Agent configuration
        this.agents = [
            { id: 'initial_classifier', name: 'Initial Classifier', icon: '🔍' },
            { id: 'detail_extractor', name: 'Detail Extractor', icon: '📋' },
            { id: 'damage_detector', name: 'Damage Detector', icon: '🔎' },
            { id: 'final_compiler', name: 'Final Compiler', icon: '✅' }
        ];
        
        // State
        this.currentAgent = null;
        this.agentStates = {};
        
        this.render();
    }
    
    render() {
        this.shadowRoot.innerHTML = `
            <style>
                :host {
                    display: block;
                    padding: 20px;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                }
                
                .monitor-container {
                    background: white;
                    border-radius: 12px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    padding: 20px;
                }
                
                .monitor-title {
                    font-size: 18px;
                    font-weight: 600;
                    margin-bottom: 20px;
                    color: #333;
                }
                
                .agents-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 15px;
                }
                
                .agent-card {
                    border: 2px solid #e0e0e0;
                    border-radius: 8px;
                    padding: 15px;
                    transition: all 0.3s ease;
                    position: relative;
                    overflow: hidden;
                }
                
                .agent-card.idle {
                    background: #f5f5f5;
                }
                
                .agent-card.running {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border-color: #667eea;
                    animation: pulse 2s infinite;
                }
                
                .agent-card.completed {
                    background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
                    border-color: #84fab0;
                }
                
                .agent-card.error {
                    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    color: white;
                    border-color: #f5576c;
                }
                
                .agent-card.paused {
                    background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%);
                    color: white;
                    border-color: #ffd89b;
                }
                
                @keyframes pulse {
                    0% { box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.7); }
                    50% { box-shadow: 0 0 0 10px rgba(102, 126, 234, 0); }
                    100% { box-shadow: 0 0 0 0 rgba(102, 126, 234, 0); }
                }
                
                .agent-header {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    margin-bottom: 10px;
                }
                
                .agent-icon {
                    font-size: 24px;
                    margin-right: 10px;
                }
                
                .agent-name {
                    font-weight: 600;
                    flex: 1;
                }
                
                .agent-status {
                    font-size: 12px;
                    padding: 2px 8px;
                    border-radius: 12px;
                    background: rgba(255,255,255,0.3);
                }
                
                .agent-timer {
                    display: flex;
                    align-items: center;
                    margin-bottom: 8px;
                    font-size: 14px;
                }
                
                .timer-icon {
                    margin-right: 5px;
                }
                
                .timer-bar {
                    width: 100%;
                    height: 6px;
                    background: rgba(255,255,255,0.3);
                    border-radius: 3px;
                    overflow: hidden;
                    margin-bottom: 8px;
                }
                
                .timer-progress {
                    height: 100%;
                    background: white;
                    border-radius: 3px;
                    transition: width 0.3s ease;
                }
                
                .inference-counter {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    font-size: 14px;
                }
                
                .inference-badges {
                    display: flex;
                    gap: 5px;
                    margin-top: 5px;
                }
                
                .inference-badge {
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    background: white;
                    opacity: 0.5;
                }
                
                .inference-badge.active {
                    opacity: 1;
                    animation: blink 0.5s;
                }
                
                @keyframes blink {
                    0%, 100% { transform: scale(1); }
                    50% { transform: scale(1.5); }
                }
                
                .multi-inference-indicator {
                    display: flex;
                    align-items: center;
                    margin-top: 8px;
                    font-size: 12px;
                    opacity: 0.8;
                }
                
                .frames-count {
                    margin-left: 5px;
                    font-weight: 600;
                }
            </style>
            
            <div class="monitor-container">
                <h2 class="monitor-title">Agent Pipeline Status</h2>
                <div class="agents-grid">
                    ${this.agents.map(agent => this.renderAgentCard(agent)).join('')}
                </div>
            </div>
        `;
    }
    
    renderAgentCard(agent) {
        const state = this.agentStates[agent.id] || { status: 'idle' };
        
        return `
            <div class="agent-card ${state.status}" data-agent="${agent.id}">
                <div class="agent-header">
                    <span class="agent-icon">${agent.icon}</span>
                    <span class="agent-name">${agent.name}</span>
                    <span class="agent-status">${state.status.toUpperCase()}</span>
                </div>
                
                ${state.timer ? `
                    <div class="agent-timer">
                        <span class="timer-icon">⏱</span>
                        <span>${state.timeRemaining || state.timer}s</span>
                    </div>
                    <div class="timer-bar">
                        <div class="timer-progress" style="width: ${state.progress || 0}%"></div>
                    </div>
                ` : ''}
                
                <div class="inference-counter">
                    <span>Inferences: ${state.inferenceCount || 0}</span>
                    ${state.inferenceType ? `
                        <span class="multi-inference-indicator">
                            ${state.inferenceType === 'multi_image' ? '🖼️' : '📷'}
                            <span class="frames-count">${state.framesUsed || 1}</span>
                        </span>
                    ` : ''}
                </div>
                
                ${state.inferenceCount > 0 ? `
                    <div class="inference-badges">
                        ${Array.from({length: Math.min(state.inferenceCount, 5)}, (_, i) => 
                            `<div class="inference-badge ${i === state.inferenceCount - 1 ? 'active' : ''}"></div>`
                        ).join('')}
                    </div>
                ` : ''}
            </div>
        `;
    }
    
    /**
     * Update agent state
     */
    updateAgent(agentId, state) {
        this.agentStates[agentId] = state;
        
        // Update specific agent card
        const card = this.shadowRoot.querySelector(`[data-agent="${agentId}"]`);
        if (card) {
            const agent = this.agents.find(a => a.id === agentId);
            if (agent) {
                card.outerHTML = this.renderAgentCard(agent);
            }
        }
    }
    
    /**
     * Handle agent started event
     */
    onAgentStarted(data) {
        this.currentAgent = data.agent;
        this.updateAgent(data.agent, {
            status: 'running',
            timer: data.timer_seconds,
            timeRemaining: data.timer_seconds,
            progress: 0,
            inferenceCount: 0,
            startTime: Date.now()
        });
        
        // Start timer animation
        this.startTimer(data.agent, data.timer_seconds);
    }
    
    /**
     * Handle inference update
     */
    onInferenceUpdate(data) {
        const state = this.agentStates[data.agent] || {};
        state.inferenceCount = (state.inferenceCount || 0) + 1;
        state.inferenceType = data.inference_type;
        state.framesUsed = data.frames_used;
        state.timeRemaining = data.timer_remaining;
        
        this.updateAgent(data.agent, state);
    }
    
    /**
     * Handle agent completed
     */
    onAgentCompleted(data) {
        const state = this.agentStates[data.agent] || {};
        state.status = 'completed';
        state.progress = 100;
        
        if (data.result) {
            state.totalInferences = data.result.total_inferences;
            state.aggregationMethod = data.result.aggregation_method;
        }
        
        this.updateAgent(data.agent, state);
    }
    
    /**
     * Handle agent error
     */
    onAgentError(data) {
        const state = this.agentStates[data.agent] || {};
        state.status = 'error';
        state.error = data.error;
        
        this.updateAgent(data.agent, state);
    }
    
    /**
     * Handle pause
     */
    onPause(data) {
        if (data.paused_agent) {
            const state = this.agentStates[data.paused_agent] || {};
            state.status = 'paused';
            state.timeRemaining = data.timer_remaining;
            
            this.updateAgent(data.paused_agent, state);
        }
    }
    
    /**
     * Handle resume
     */
    onResume(data) {
        if (data.resuming_agent) {
            const state = this.agentStates[data.resuming_agent] || {};
            state.status = 'running';
            
            if (data.restarted) {
                state.inferenceCount = 0;
                state.timeRemaining = data.timer_seconds;
            }
            
            this.updateAgent(data.resuming_agent, state);
            
            // Restart timer
            this.startTimer(data.resuming_agent, state.timeRemaining || data.timer_seconds);
        }
    }
    
    /**
     * Start timer animation
     */
    startTimer(agentId, duration) {
        const startTime = Date.now();
        const endTime = startTime + (duration * 1000);
        
        const updateTimer = () => {
            const now = Date.now();
            const remaining = Math.max(0, endTime - now);
            const progress = ((duration * 1000 - remaining) / (duration * 1000)) * 100;
            
            const state = this.agentStates[agentId] || {};
            state.timeRemaining = Math.ceil(remaining / 1000);
            state.progress = Math.min(100, progress);
            
            this.updateAgent(agentId, state);
            
            if (remaining > 0 && state.status === 'running') {
                requestAnimationFrame(updateTimer);
            }
        };
        
        requestAnimationFrame(updateTimer);
    }
    
    /**
     * Reset all agents
     */
    reset() {
        this.agentStates = {};
        this.currentAgent = null;
        this.render();
    }
}

// Register custom element
customElements.define('agent-monitor-v2', AgentMonitorV2);