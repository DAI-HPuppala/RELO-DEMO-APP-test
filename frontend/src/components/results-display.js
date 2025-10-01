/**
 * Results Display Component
 * Manages the classification results display
 */
class ResultsDisplay {
    constructor() {
        this.resultsContainer = document.getElementById('resultsDisplay');
        this.progressContainer = document.getElementById('progressDisplay');
        this.exportBtn = document.getElementById('exportBtn');
        this.manualControls = document.getElementById('manualControls');
        
        this.currentResults = null;
        this.sessionId = null;
    }

    /**
     * Reset display
     */
    reset() {
        this.showNoResults();
        this.resetAgentProgress();
        this.currentResults = null;
        
        if (this.exportBtn) {
            this.exportBtn.disabled = true;
        }
    }

    /**
     * Show no results message
     */
    showNoResults() {
        if (this.resultsContainer) {
            this.resultsContainer.innerHTML = `
                <div class="no-results">
                    <p>No classification results yet</p>
                    <p class="text-muted">Start monitoring to begin classification</p>
                </div>
            `;
        }
    }

    /**
     * Update agent progress
     */
    updateAgentProgress(agentName, status) {
        const agentElement = document.querySelector(`[data-agent="${agentName}"]`);
        if (agentElement) {
            const statusElement = agentElement.querySelector('.agent-status');
            if (statusElement) {
                statusElement.textContent = status;
                statusElement.className = `agent-status ${status.toLowerCase().replace(' ', '-')}`;
            }
            
            // Update visual state
            agentElement.classList.remove('waiting', 'processing', 'completed', 'error');
            if (status === 'Processing') {
                agentElement.classList.add('processing');
            } else if (status === 'Completed') {
                agentElement.classList.add('completed');
            } else if (status === 'Error') {
                agentElement.classList.add('error');
            } else {
                agentElement.classList.add('waiting');
            }
        }
    }

    /**
     * Reset agent progress
     */
    resetAgentProgress() {
        const agents = ['initial', 'detail', 'damage', 'final'];
        agents.forEach(agent => {
            this.updateAgentProgress(agent, 'Waiting');
        });
    }

    /**
     * Display partial results
     */
    displayPartialResults(agentName, results) {
        if (!this.resultsContainer) return;
        
        // Create or update results section
        let resultsSection = document.getElementById(`results-${agentName}`);
        if (!resultsSection) {
            const noResults = this.resultsContainer.querySelector('.no-results');
            if (noResults) {
                this.resultsContainer.innerHTML = '';
            }
            
            resultsSection = document.createElement('div');
            resultsSection.id = `results-${agentName}`;
            resultsSection.className = 'result-section';
            this.resultsContainer.appendChild(resultsSection);
        }
        
        // Format agent display name
        const agentDisplayNames = {
            'initial': 'Initial Classification',
            'detail': 'Detail Extraction',
            'damage': 'Damage Detection',
            'final': 'Final Results'
        };
        
        const displayName = agentDisplayNames[agentName] || agentName;
        
        // Update section content
        resultsSection.innerHTML = `
            <h3>${displayName}</h3>
            <div class="result-content">
                ${this.formatResults(results)}
            </div>
        `;
    }

    /**
     * Display final results
     */
    displayFinalResults(results) {
        if (!this.resultsContainer) return;
        
        this.currentResults = results;
        
        // Enable export button
        if (this.exportBtn) {
            this.exportBtn.disabled = false;
        }
        
        // Create comprehensive results display
        this.resultsContainer.innerHTML = `
            <div class="final-results">
                <h3>Classification Complete</h3>
                
                <div class="result-summary">
                    <div class="summary-item">
                        <span class="label">Item Type:</span>
                        <span class="value">${results.item_type || 'Unknown'}</span>
                    </div>
                    <div class="summary-item">
                        <span class="label">Brand:</span>
                        <span class="value">${results.brand || 'Unknown'}</span>
                    </div>
                    <div class="summary-item">
                        <span class="label">Color:</span>
                        <span class="value">${results.color || 'Unknown'}</span>
                    </div>
                    <div class="summary-item">
                        <span class="label">Size:</span>
                        <span class="value">${results.size || 'Unknown'}</span>
                    </div>
                    <div class="summary-item">
                        <span class="label">Condition:</span>
                        <span class="value ${this.getConditionClass(results.condition)}">${results.condition || 'Unknown'}</span>
                    </div>
                    <div class="summary-item">
                        <span class="label">Return Valid:</span>
                        <span class="value ${results.return_valid ? 'valid' : 'invalid'}">${results.return_valid ? 'Yes' : 'No'}</span>
                    </div>
                </div>
                
                ${results.damage_details ? `
                    <div class="damage-details">
                        <h4>Damage Details</h4>
                        <ul>
                            ${results.damage_details.map(d => `<li>${d}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}
                
                ${results.reason ? `
                    <div class="return-reason">
                        <h4>Return Reason</h4>
                        <p>${results.reason}</p>
                    </div>
                ` : ''}
                
                <div class="result-metadata">
                    <div class="metadata-item">
                        <span class="label">Confidence:</span>
                        <span class="value">${Math.round((results.confidence || 0) * 100)}%</span>
                    </div>
                    <div class="metadata-item">
                        <span class="label">Processing Time:</span>
                        <span class="value">${results.processing_time || '0'}s</span>
                    </div>
                    <div class="metadata-item">
                        <span class="label">Frames Analyzed:</span>
                        <span class="value">${results.frames_analyzed || '0'}</span>
                    </div>
                </div>
            </div>
        `;
        
        // Mark all agents as completed
        this.updateAgentProgress('final', 'Completed');
    }

    /**
     * Format results object for display
     */
    formatResults(results) {
        if (!results) return '<p>No data</p>';
        
        if (typeof results === 'string') {
            return `<p>${results}</p>`;
        }
        
        if (typeof results === 'object') {
            const items = [];
            for (const [key, value] of Object.entries(results)) {
                const formattedKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                let formattedValue = value;
                
                if (typeof value === 'boolean') {
                    formattedValue = value ? 'Yes' : 'No';
                } else if (typeof value === 'number') {
                    formattedValue = value.toFixed(2);
                } else if (Array.isArray(value)) {
                    formattedValue = value.join(', ');
                } else if (value === null || value === undefined) {
                    formattedValue = 'N/A';
                }
                
                items.push(`
                    <div class="result-item">
                        <span class="result-key">${formattedKey}:</span>
                        <span class="result-value">${formattedValue}</span>
                    </div>
                `);
            }
            return items.join('');
        }
        
        return '<p>Invalid data format</p>';
    }

    /**
     * Get condition class for styling
     */
    getConditionClass(condition) {
        if (!condition) return '';
        
        const lowerCondition = condition.toLowerCase();
        if (lowerCondition.includes('new') || lowerCondition.includes('excellent')) {
            return 'condition-excellent';
        } else if (lowerCondition.includes('good')) {
            return 'condition-good';
        } else if (lowerCondition.includes('fair') || lowerCondition.includes('used')) {
            return 'condition-fair';
        } else if (lowerCondition.includes('damaged') || lowerCondition.includes('poor')) {
            return 'condition-poor';
        }
        return '';
    }

    /**
     * Set mode (automatic/manual)
     */
    setMode(mode) {
        if (this.manualControls) {
            this.manualControls.style.display = mode === 'manual' ? 'block' : 'none';
        }
    }

    /**
     * Get current results
     */
    getResults() {
        return this.currentResults;
    }
}

/**
 * Trigger agent manually
 */
window.triggerAgent = function(agentType) {
    if (window.webrtcClient && window.webrtcClient.getConnectionStatus()) {
        window.webrtcClient.triggerAgent(agentType);
        
        // Update UI to show processing
        if (window.resultsDisplay) {
            window.resultsDisplay.updateAgentProgress(agentType, 'Processing');
        }
    } else {
        console.error('WebRTC client not connected');
        alert('Please start monitoring first');
    }
};