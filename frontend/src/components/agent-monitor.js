/**
 * Agent Monitor Component
 * Displays real-time status of classification agents
 */
class AgentMonitor {
    constructor() {
        // Use shared agent configuration
        this.agents = window.AGENT_CONFIG ? window.AGENT_CONFIG.agents : [
            { id: 'initial_classifier', name: 'Initial Classifier', timer: 4.0 },
            { id: 'detail_extractor', name: 'Detail Extractor', timer: 3.0 },
            { id: 'damage_detector', name: 'Damage Detector', timer: 4.0 },
            { id: 'final_compiler', name: 'Final Compiler', timer: 2.0 }
        ];
        
        this.currentAgent = null;
        this.mode = 'automatic';
        this.timers = {};
        
        this.init();
    }
    
    /**
     * Initialize the component
     */
    init() {
        this.createAgentMonitorUI();
    }
    
    /**
     * Create the agent monitor UI
     */
    createAgentMonitorUI() {
        // Find or create container
        let container = document.getElementById('agentMonitor');
        if (!container) {
            // Create container if it doesn't exist
            container = document.createElement('div');
            container.id = 'agentMonitor';
            container.className = 'agent-monitor';
            
            // Insert after stream viewer
            const streamViewer = document.querySelector('.stream-viewer');
            if (streamViewer && streamViewer.parentNode) {
                streamViewer.parentNode.insertBefore(container, streamViewer.nextSibling);
            } else {
                document.querySelector('.main-content').appendChild(container);
            }
        }
        
        // Build UI
        container.innerHTML = `
            <div class="agent-monitor-header">
                <h3>Agent Processing Status</h3>
                <span class="agent-mode" id="agentMode">Mode: ${this.mode}</span>
            </div>
            <div class="agent-list" id="agentList">
                ${this.agents.map(agent => this.createAgentCard(agent)).join('')}
            </div>
            <div class="agent-progress-bar" id="overallProgress">
                <div class="progress-fill" id="overallProgressFill"></div>
                <span class="progress-text" id="overallProgressText">0% Complete</span>
            </div>
        `;
        
        // Add styles if not already added
        if (!document.getElementById('agentMonitorStyles')) {
            const style = document.createElement('style');
            style.id = 'agentMonitorStyles';
            style.textContent = `
                .agent-monitor {
                    background: var(--card-bg, #1a1a1a);
                    border-radius: 8px;
                    padding: 16px;
                    margin: 16px 0;
                    border: 1px solid var(--border-color, #333);
                }
                
                .agent-monitor-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 16px;
                }
                
                .agent-monitor-header h3 {
                    margin: 0;
                    color: var(--text-primary, #fff);
                    font-size: 18px;
                }
                
                .agent-mode {
                    background: var(--badge-bg, #2a2a2a);
                    color: var(--text-secondary, #999);
                    padding: 4px 12px;
                    border-radius: 4px;
                    font-size: 12px;
                    text-transform: uppercase;
                }
                
                .agent-list {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 12px;
                    margin-bottom: 16px;
                }
                
                .agent-card {
                    background: var(--card-inner-bg, #222);
                    border: 1px solid var(--border-color, #333);
                    border-radius: 6px;
                    padding: 12px;
                    transition: all 0.3s ease;
                }
                
                .agent-card.active {
                    border-color: var(--primary-color, #4CAF50);
                    background: var(--card-active-bg, #1e2e1e);
                    animation: pulse 1.5s infinite;
                }
                
                .agent-card.completed {
                    border-color: var(--success-color, #4CAF50);
                    opacity: 0.8;
                }
                
                @keyframes pulse {
                    0% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.4); }
                    50% { box-shadow: 0 0 0 10px rgba(76, 175, 80, 0); }
                    100% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); }
                }
                
                .agent-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 8px;
                }
                
                .agent-name {
                    font-weight: 500;
                    color: var(--text-primary, #fff);
                    font-size: 14px;
                }
                
                .agent-status {
                    display: flex;
                    align-items: center;
                    gap: 6px;
                }
                
                .status-indicator {
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    background: var(--status-idle, #666);
                }
                
                .status-indicator.pending {
                    background: var(--status-idle, #666);
                }
                
                .status-indicator.active {
                    background: var(--status-active, #FFA500);
                    animation: blink 1s infinite;
                }
                
                .status-indicator.completed {
                    background: var(--status-success, #4CAF50);
                }
                
                @keyframes blink {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.5; }
                }
                
                .status-text {
                    font-size: 11px;
                    color: var(--text-secondary, #999);
                    text-transform: uppercase;
                }
                
                .agent-timer {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    margin-top: 8px;
                }
                
                .timer-bar {
                    flex: 1;
                    height: 4px;
                    background: var(--progress-bg, #333);
                    border-radius: 2px;
                    overflow: hidden;
                }
                
                .timer-fill {
                    height: 100%;
                    background: var(--primary-color, #4CAF50);
                    transition: width 0.3s linear;
                    width: 0%;
                }
                
                .timer-text {
                    font-size: 12px;
                    color: var(--text-secondary, #999);
                    min-width: 40px;
                    text-align: right;
                }
                
                .agent-result {
                    margin-top: 8px;
                    padding-top: 8px;
                    border-top: 1px solid var(--border-color, #333);
                    font-size: 12px;
                    color: var(--text-secondary, #999);
                    display: none;
                }
                
                .agent-result.show {
                    display: block;
                }
                
                .result-confidence {
                    color: var(--primary-color, #4CAF50);
                    font-weight: 500;
                }
                
                .agent-progress-bar {
                    background: var(--progress-bg, #333);
                    height: 24px;
                    border-radius: 4px;
                    overflow: hidden;
                    position: relative;
                }
                
                .progress-fill {
                    height: 100%;
                    background: linear-gradient(90deg, var(--primary-color, #4CAF50), var(--primary-light, #66BB6A));
                    transition: width 0.5s ease;
                    width: 0%;
                }
                
                .progress-text {
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    color: var(--text-primary, #fff);
                    font-size: 12px;
                    font-weight: 500;
                }
            `;
            document.head.appendChild(style);
        }
    }
    
    /**
     * Create agent card HTML
     */
    createAgentCard(agent) {
        return `
            <div class="agent-card" id="agent-${agent.id}" data-agent="${agent.id}">
                <div class="agent-header">
                    <span class="agent-name">${agent.name}</span>
                    <div class="agent-status">
                        <div class="status-indicator pending" id="status-${agent.id}"></div>
                        <span class="status-text" id="status-text-${agent.id}">Pending</span>
                    </div>
                </div>
                <div class="agent-timer">
                    <div class="timer-bar">
                        <div class="timer-fill" id="timer-${agent.id}"></div>
                    </div>
                    <span class="timer-text" id="timer-text-${agent.id}">${agent.timer}s</span>
                </div>
                <div class="agent-result" id="result-${agent.id}">
                    <div>Confidence: <span class="result-confidence" id="confidence-${agent.id}">-</span></div>
                </div>
            </div>
        `;
    }
    
    /**
     * Handle agent started event
     */
    onAgentStarted(data) {
        const { agent, timer_seconds, sequence_position, total_agents, mode } = data;

        this.currentAgent = agent;
        this.mode = mode || 'automatic';
        
        // Update mode display
        const modeEl = document.getElementById('agentMode');
        if (modeEl) {
            modeEl.textContent = `Mode: ${this.mode}`;
        }
        
        // Update agent card
        const card = document.getElementById(`agent-${agent}`);
        if (card) {
            card.classList.add('active');
            card.classList.remove('completed');
        }
        
        // Update status
        const statusIndicator = document.getElementById(`status-${agent}`);
        const statusText = document.getElementById(`status-text-${agent}`);
        if (statusIndicator) {
            statusIndicator.className = 'status-indicator active';
        }
        if (statusText) {
            statusText.textContent = 'Processing';
        }
        
        // Start timer if in automatic mode
        if (mode === 'automatic' && timer_seconds) {
            this.startAgentTimer(agent, timer_seconds);
        }
        
        // Update overall progress
        if (sequence_position && total_agents) {
            this.updateOverallProgress(sequence_position - 1, total_agents);
        }
    }
    
    /**
     * Handle progress update
     */
    onProgressUpdate(data) {
        const { agent, progress, timer_remaining, frames_collected } = data;
        
        // Update timer display
        if (timer_remaining !== undefined) {
            const timerText = document.getElementById(`timer-text-${agent}`);
            if (timerText) {
                timerText.textContent = `${timer_remaining.toFixed(1)}s`;
            }
            
            // Update timer bar
            const timerFill = document.getElementById(`timer-${agent}`);
            if (timerFill) {
                timerFill.style.width = `${progress}%`;
            }
        }
        
        // Update status text with frame count
        if (frames_collected !== undefined) {
            const statusText = document.getElementById(`status-text-${agent}`);
            if (statusText) {
                statusText.textContent = `${frames_collected} frames`;
            }
        }
    }
    
    /**
     * Handle agent completed event
     */
    onAgentCompleted(data) {
        const { agent, results } = data;

        // Stop timer if running
        if (this.timers[agent]) {
            clearInterval(this.timers[agent]);
            delete this.timers[agent];
        }
        
        // Update agent card
        const card = document.getElementById(`agent-${agent}`);
        if (card) {
            card.classList.remove('active');
            card.classList.add('completed');
        }
        
        // Update status
        const statusIndicator = document.getElementById(`status-${agent}`);
        const statusText = document.getElementById(`status-text-${agent}`);
        if (statusIndicator) {
            statusIndicator.className = 'status-indicator completed';
        }
        if (statusText) {
            statusText.textContent = 'Complete';
        }
        
        // Update timer to 100%
        const timerFill = document.getElementById(`timer-${agent}`);
        if (timerFill) {
            timerFill.style.width = '100%';
        }
        
        // Show results if available
        if (results && results.confidence !== undefined) {
            const resultDiv = document.getElementById(`result-${agent}`);
            const confidenceSpan = document.getElementById(`confidence-${agent}`);
            
            if (resultDiv) {
                resultDiv.classList.add('show');
            }
            if (confidenceSpan) {
                confidenceSpan.textContent = `${(results.confidence * 100).toFixed(1)}%`;
            }
        }
    }
    
    /**
     * Start timer for agent
     */
    startAgentTimer(agent, duration) {
        // Clear any existing timer
        if (this.timers[agent]) {
            clearInterval(this.timers[agent]);
        }
        
        const startTime = Date.now();
        const timerFill = document.getElementById(`timer-${agent}`);
        const timerText = document.getElementById(`timer-text-${agent}`);
        
        // Update timer every 100ms for smooth animation
        this.timers[agent] = setInterval(() => {
            const elapsed = (Date.now() - startTime) / 1000;
            const remaining = Math.max(0, duration - elapsed);
            const progress = Math.min(100, (elapsed / duration) * 100);
            
            if (timerFill) {
                timerFill.style.width = `${progress}%`;
            }
            if (timerText) {
                timerText.textContent = `${remaining.toFixed(1)}s`;
            }
            
            // Stop timer when complete
            if (remaining <= 0) {
                clearInterval(this.timers[agent]);
                delete this.timers[agent];
            }
        }, 100);
    }
    
    /**
     * Update overall progress
     */
    updateOverallProgress(completed, total) {
        const progress = (completed / total) * 100;
        const progressFill = document.getElementById('overallProgressFill');
        const progressText = document.getElementById('overallProgressText');
        
        if (progressFill) {
            progressFill.style.width = `${progress}%`;
        }
        if (progressText) {
            progressText.textContent = `${Math.round(progress)}% Complete (${completed}/${total} agents)`;
        }
    }
    
    /**
     * Reset all agents
     */
    reset() {
        // Clear all timers
        Object.keys(this.timers).forEach(agent => {
            clearInterval(this.timers[agent]);
        });
        this.timers = {};
        
        // Reset all agent cards
        this.agents.forEach(agent => {
            const card = document.getElementById(`agent-${agent.id}`);
            if (card) {
                card.classList.remove('active', 'completed');
            }
            
            // Reset status
            const statusIndicator = document.getElementById(`status-${agent.id}`);
            const statusText = document.getElementById(`status-text-${agent.id}`);
            if (statusIndicator) {
                statusIndicator.className = 'status-indicator pending';
            }
            if (statusText) {
                statusText.textContent = 'Pending';
            }
            
            // Reset timer
            const timerFill = document.getElementById(`timer-${agent.id}`);
            const timerText = document.getElementById(`timer-text-${agent.id}`);
            if (timerFill) {
                timerFill.style.width = '0%';
            }
            if (timerText) {
                timerText.textContent = `${agent.timer}s`;
            }
            
            // Hide results
            const resultDiv = document.getElementById(`result-${agent.id}`);
            if (resultDiv) {
                resultDiv.classList.remove('show');
            }
        });
        
        // Reset overall progress
        this.updateOverallProgress(0, this.agents.length);
    }
    
    /**
     * Set mode
     */
    setMode(mode) {
        this.mode = mode;
        const modeEl = document.getElementById('agentMode');
        if (modeEl) {
            modeEl.textContent = `Mode: ${mode}`;
        }
    }
}

// Export for use
window.AgentMonitor = AgentMonitor;