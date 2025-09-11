/**
 * Progress Panel Component for displaying agent results
 */
class ProgressPanel {
    constructor() {
        this.progressBar = document.getElementById('progressBar');
        this.progressText = document.getElementById('progressText');
        this.agentCards = {
            'initial_classifier': document.getElementById('agent-initial'),
            'detail_extractor': document.getElementById('agent-detail'),
            'damage_detector': document.getElementById('agent-damage'),
            'final_compiler': document.getElementById('agent-final')
        };
    }

    /**
     * Update overall progress
     */
    updateProgress(percentage) {
        if (this.progressBar) {
            this.progressBar.style.width = `${percentage}%`;
        }
        if (this.progressText) {
            this.progressText.textContent = `${percentage}%`;
        }
    }

    /**
     * Update agent status
     */
    updateAgentStatus(agentId, status) {
        const card = this.agentCards[agentId];
        if (!card) return;

        // Update card styling
        card.classList.remove('active', 'completed');
        
        if (status === 'processing') {
            card.classList.add('active');
        } else if (status === 'completed') {
            card.classList.add('completed');
        }

        // Update status text
        const statusElement = card.querySelector('.agent-status');
        if (statusElement) {
            statusElement.textContent = status.charAt(0).toUpperCase() + status.slice(1);
            statusElement.className = `agent-status ${status}`;
        }
    }

    /**
     * Update agent results
     */
    updateAgentResults(agentId, results) {
        const card = this.agentCards[agentId];
        if (!card) return;

        const resultsContainer = card.querySelector('.agent-results');
        if (!resultsContainer) return;

        // Update specific result fields based on agent
        switch (agentId) {
            case 'initial_classifier':
                this.updateResultItem('initial-type', 'Type', results.type);
                this.updateResultItem('initial-color', 'Color', results.color || results.color_primary);
                this.updateResultItem('initial-pattern', 'Pattern', results.pattern);
                break;

            case 'detail_extractor':
                this.updateResultItem('detail-brand', 'Brand', results.brand);
                this.updateResultItem('detail-size', 'Size', results.size);
                break;

            case 'damage_detector':
                this.updateResultItem('damage-status', 'Damage', results.damage || (results.has_damage ? 'Yes' : 'No'));
                this.updateResultItem('damage-type', 'Type', results.damage_type);
                break;

            case 'final_compiler':
                resultsContainer.innerHTML = '<div class="result-item">Results compiled successfully</div>';
                break;
        }
    }

    /**
     * Update a specific result item
     */
    updateResultItem(elementId, label, value) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = `${label}: ${value || '--'}`;
            if (value) {
                element.style.fontWeight = '500';
            }
        }
    }

    /**
     * Handle progressive update
     */
    handleProgressiveUpdate(data) {
        const { agent, attributes, confidence, frames_processed } = data;
        
        // Update agent results
        this.updateAgentResults(agent, attributes);
        
        // If agent is processing, update status
        const agentCard = this.agentCards[agent];
        if (agentCard && !agentCard.classList.contains('completed')) {
            this.updateAgentStatus(agent, 'processing');
        }

        // Add confidence indicator if available
        if (confidence && agentCard) {
            const statusElement = agentCard.querySelector('.agent-status');
            if (statusElement) {
                statusElement.textContent = `Processing (${Math.round(confidence * 100)}% confidence)`;
            }
        }
    }

    /**
     * Handle agent started event
     */
    handleAgentStarted(data) {
        const { agent, timer_seconds } = data;
        
        this.updateAgentStatus(agent, 'processing');
        
        // Show timer if in automatic mode
        if (timer_seconds) {
            this.startAgentTimer(agent, timer_seconds);
        }
    }

    /**
     * Handle agent completed event
     */
    handleAgentCompleted(data) {
        const { agent, results } = data;
        
        this.updateAgentStatus(agent, 'completed');
        this.updateAgentResults(agent, results);
    }

    /**
     * Start timer for agent
     */
    startAgentTimer(agentId, seconds) {
        const card = this.agentCards[agentId];
        if (!card) return;

        const statusElement = card.querySelector('.agent-status');
        if (!statusElement) return;

        let remaining = seconds;
        const timerInterval = setInterval(() => {
            if (remaining > 0) {
                statusElement.textContent = `Processing (${remaining}s)`;
                remaining--;
            } else {
                clearInterval(timerInterval);
            }
        }, 1000);
    }

    /**
     * Reset all agents
     */
    resetAgents() {
        for (const agentId in this.agentCards) {
            this.updateAgentStatus(agentId, 'waiting');
            
            const card = this.agentCards[agentId];
            if (card) {
                const resultsContainer = card.querySelector('.agent-results');
                if (resultsContainer) {
                    // Reset to default content
                    if (agentId === 'initial_classifier') {
                        resultsContainer.innerHTML = `
                            <div class="result-item" id="initial-type">Type: --</div>
                            <div class="result-item" id="initial-color">Color: --</div>
                            <div class="result-item" id="initial-pattern">Pattern: --</div>
                        `;
                    } else if (agentId === 'detail_extractor') {
                        resultsContainer.innerHTML = `
                            <div class="result-item" id="detail-brand">Brand: --</div>
                            <div class="result-item" id="detail-size">Size: --</div>
                        `;
                    } else if (agentId === 'damage_detector') {
                        resultsContainer.innerHTML = `
                            <div class="result-item" id="damage-status">Damage: --</div>
                            <div class="result-item" id="damage-type">Type: --</div>
                        `;
                    } else if (agentId === 'final_compiler') {
                        resultsContainer.innerHTML = `
                            <div class="result-item">Compiling results...</div>
                        `;
                    }
                }
            }
        }
        
        this.updateProgress(0);
    }
}

// Export for use in other modules
window.ProgressPanel = ProgressPanel;