/**
 * Progress Monitor Component
 * Handles real-time progressive updates display
 */
class ProgressMonitor {
    constructor() {
        this.agents = {
            initial: { 
                name: 'Initial Classifier', 
                timer: 4.0, 
                inferences: 0, 
                status: 'waiting',
                results: [],
                startTime: null
            },
            detail: { 
                name: 'Detail Extractor', 
                timer: 3.0, 
                inferences: 0, 
                status: 'waiting',
                results: [],
                startTime: null
            },
            damage: { 
                name: 'Damage Detector', 
                timer: 4.0, 
                inferences: 0, 
                status: 'waiting',
                results: [],
                startTime: null
            },
            final: { 
                name: 'Final Compiler', 
                timer: 2.0, 
                inferences: 0, 
                status: 'waiting',
                results: [],
                startTime: null
            }
        };
        
        this.inferenceLog = [];
        this.totalInferences = 0;
        this.timerIntervals = {};
    }

    /**
     * Reset all progress
     */
    reset() {
        Object.keys(this.agents).forEach(agentId => {
            this.resetAgent(agentId);
        });
        
        this.inferenceLog = [];
        this.totalInferences = 0;
        this.updateInferenceCount();
        this.clearLog();
    }

    /**
     * Reset individual agent
     */
    resetAgent(agentId) {
        const agent = this.agents[agentId];
        if (!agent) return;
        
        agent.inferences = 0;
        agent.status = 'waiting';
        agent.results = [];
        agent.startTime = null;
        
        // Clear timer interval if exists
        if (this.timerIntervals[agentId]) {
            clearInterval(this.timerIntervals[agentId]);
            delete this.timerIntervals[agentId];
        }
        
        // Update UI
        this.updateAgentUI(agentId);
    }

    /**
     * Handle agent started event
     */
    onAgentStarted(data) {
        const agentId = this.getAgentId(data.agent);
        if (!agentId) return;
        
        const agent = this.agents[agentId];
        agent.status = 'processing';
        agent.startTime = Date.now();
        agent.inferences = 0;
        agent.results = [];
        
        // Start timer animation
        this.startTimer(agentId);
        
        // Update UI
        this.updateAgentUI(agentId);
        
        // Add to log
        this.addLogEntry({
            type: 'agent_started',
            agent: data.agent,
            timestamp: new Date().toISOString()
        });
    }

    /**
     * Handle progress update
     */
    onProgressUpdate(data) {
        const agentId = this.getAgentId(data.agent);
        if (!agentId) return;
        
        const agent = this.agents[agentId];
        
        // Update inference count
        if (data.inference_num) {
            agent.inferences = data.inference_num;
            this.totalInferences++;
            this.updateTotalInferences();
        }
        
        // Store partial results
        if (data.partial_results) {
            agent.results.push({
                inference: data.inference_num || agent.inferences,
                results: data.partial_results,
                timestamp: new Date().toISOString()
            });
            
            // Display in agent card
            this.displayAgentResults(agentId, data.partial_results);
        }
        
        // Update progress percentage if provided
        if (data.progress !== undefined) {
            const percentage = (data.progress / 100) * agent.timer;
            this.updateTimerProgress(agentId, percentage);
        }
        
        // Add to log
        this.addLogEntry({
            type: 'inference',
            agent: data.agent,
            inference_num: data.inference_num || agent.inferences,
            attributes: data.partial_results,
            timestamp: new Date().toISOString()
        });
        
        // Update UI
        this.updateAgentUI(agentId);
    }

    /**
     * Handle agent completed
     */
    onAgentCompleted(data) {
        const agentId = this.getAgentId(data.agent);
        if (!agentId) return;
        
        const agent = this.agents[agentId];
        agent.status = 'completed';
        
        // Stop timer
        if (this.timerIntervals[agentId]) {
            clearInterval(this.timerIntervals[agentId]);
            delete this.timerIntervals[agentId];
        }
        
        // Set timer to full
        this.updateTimerProgress(agentId, agent.timer);
        
        // Store final results
        if (data.results) {
            agent.results.push({
                inference: 'final',
                results: data.results,
                timestamp: new Date().toISOString()
            });
            
            this.displayAgentResults(agentId, data.results);
        }
        
        // Add to log
        this.addLogEntry({
            type: 'agent_completed',
            agent: data.agent,
            total_inferences: agent.inferences,
            final_results: data.results,
            timestamp: new Date().toISOString()
        });
        
        // Update UI
        this.updateAgentUI(agentId);
        this.updateAgentsCompleted();
    }

    /**
     * Start timer animation for agent
     */
    startTimer(agentId) {
        const agent = this.agents[agentId];
        if (!agent) return;
        
        const startTime = Date.now();
        const duration = agent.timer * 1000; // Convert to milliseconds
        
        // Clear existing interval
        if (this.timerIntervals[agentId]) {
            clearInterval(this.timerIntervals[agentId]);
        }
        
        // Update timer every 100ms
        this.timerIntervals[agentId] = setInterval(() => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const secondsElapsed = (elapsed / 1000).toFixed(1);
            
            // Update timer display
            const timerElement = document.getElementById(`${agentId}-timer`);
            if (timerElement) {
                timerElement.textContent = `${secondsElapsed}s / ${agent.timer}s`;
            }
            
            // Update progress bar
            const fillElement = document.getElementById(`${agentId}-timer-fill`);
            if (fillElement) {
                fillElement.style.width = `${progress * 100}%`;
            }
            
            // Stop when complete
            if (progress >= 1) {
                clearInterval(this.timerIntervals[agentId]);
                delete this.timerIntervals[agentId];
            }
        }, 100);
    }

    /**
     * Update timer progress
     */
    updateTimerProgress(agentId, seconds) {
        const agent = this.agents[agentId];
        if (!agent) return;
        
        const progress = Math.min(seconds / agent.timer, 1);
        
        const fillElement = document.getElementById(`${agentId}-timer-fill`);
        if (fillElement) {
            fillElement.style.width = `${progress * 100}%`;
        }
        
        const timerElement = document.getElementById(`${agentId}-timer`);
        if (timerElement) {
            timerElement.textContent = `${seconds.toFixed(1)}s / ${agent.timer}s`;
        }
    }

    /**
     * Update agent UI
     */
    updateAgentUI(agentId) {
        const agent = this.agents[agentId];
        if (!agent) return;
        
        // Update card status
        const card = document.querySelector(`[data-agent="${agentId}"]`);
        if (card) {
            card.setAttribute('data-status', agent.status);
        }
        
        // Update status text
        const statusElement = document.getElementById(`${agentId}-status`);
        if (statusElement) {
            statusElement.textContent = agent.status.charAt(0).toUpperCase() + agent.status.slice(1);
            statusElement.className = `agent-status ${agent.status}`;
        }
        
        // Update inference count
        const inferenceElement = document.getElementById(`${agentId}-inferences`);
        if (inferenceElement) {
            inferenceElement.textContent = agent.inferences;
        }
    }

    /**
     * Display agent results in card
     */
    displayAgentResults(agentId, results) {
        const resultsElement = document.getElementById(`${agentId}-results`);
        if (!resultsElement) return;
        
        // Format results for display
        let html = '';
        if (typeof results === 'object') {
            for (const [key, value] of Object.entries(results)) {
                const formattedKey = key.replace(/_/g, ' ');
                html += `<div><strong>${formattedKey}:</strong> ${value}</div>`;
            }
        } else {
            html = `<div>${results}</div>`;
        }
        
        resultsElement.innerHTML = html;
    }

    /**
     * Add entry to inference log
     */
    addLogEntry(entry) {
        this.inferenceLog.push(entry);
        
        // Update log display
        const logContent = document.getElementById('logContent');
        if (!logContent) return;
        
        const logEntry = document.createElement('div');
        logEntry.className = 'log-entry';
        
        const timestamp = new Date(entry.timestamp).toLocaleTimeString();
        let content = `<span class="log-timestamp">[${timestamp}]</span>`;
        
        switch(entry.type) {
            case 'agent_started':
                content += `<strong>${entry.agent}</strong> started processing`;
                break;
            case 'inference':
                content += `<strong>${entry.agent}</strong> - Inference #${entry.inference_num}`;
                if (entry.attributes) {
                    content += `: ${JSON.stringify(entry.attributes)}`;
                }
                break;
            case 'agent_completed':
                content += `<strong>${entry.agent}</strong> completed (${entry.total_inferences} inferences)`;
                break;
        }
        
        logEntry.innerHTML = content;
        logContent.appendChild(logEntry);
        
        // Auto-scroll to bottom
        logContent.scrollTop = logContent.scrollHeight;
        
        // Update log count
        this.updateInferenceCount();
    }

    /**
     * Clear log
     */
    clearLog() {
        const logContent = document.getElementById('logContent');
        if (logContent) {
            logContent.innerHTML = '';
        }
        this.inferenceLog = [];
        this.updateInferenceCount();
    }

    /**
     * Update inference count display
     */
    updateInferenceCount() {
        const countElement = document.getElementById('inferenceCount');
        if (countElement) {
            countElement.textContent = `${this.inferenceLog.length} updates`;
        }
    }

    /**
     * Update total inferences display
     */
    updateTotalInferences() {
        const totalElement = document.getElementById('totalInferences');
        if (totalElement) {
            totalElement.textContent = this.totalInferences;
        }
    }

    /**
     * Update agents completed count
     */
    updateAgentsCompleted() {
        const completed = Object.values(this.agents).filter(a => a.status === 'completed').length;
        const totalAgents = Object.keys(this.agents).length;
        
        const element = document.getElementById('agentsCompleted');
        if (element) {
            element.textContent = `${completed}/${totalAgents}`;
        }
    }

    /**
     * Get agent ID from agent name
     */
    getAgentId(agentName) {
        const mapping = {
            'initial_classifier': 'initial',
            'detail_extractor': 'detail',
            'damage_detector': 'damage',
            'final_compiler': 'final',
            'initial': 'initial',
            'detail': 'detail',
            'damage': 'damage',
            'final': 'final'
        };
        
        return mapping[agentName];
    }

    /**
     * Get all agent results
     */
    getAllResults() {
        const results = {};
        Object.entries(this.agents).forEach(([agentId, agent]) => {
            if (agent.results.length > 0) {
                results[agentId] = agent.results;
            }
        });
        return results;
    }

    /**
     * Export progress data
     */
    exportProgressData() {
        return {
            agents: this.agents,
            inferenceLog: this.inferenceLog,
            totalInferences: this.totalInferences,
            timestamp: new Date().toISOString()
        };
    }
}

// Export for use in other modules
window.ProgressMonitor = ProgressMonitor;