/**
 * Control Panel Component
 */
class ControlPanel {
    constructor() {
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.modeToggle = document.getElementById('modeToggle');
        this.captureBtn = document.getElementById('captureBtn');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.nextBtn = document.getElementById('nextBtn');
        this.exportBtn = document.getElementById('exportBtn');
        this.manualControls = document.getElementById('manualControls');
        
        this.isMonitoring = false;
        this.mode = 'automatic';
        
        this.setupEventListeners();
    }

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        if (this.modeToggle) {
            this.modeToggle.addEventListener('change', (e) => {
                this.mode = e.target.checked ? 'manual' : 'automatic';
                this.onModeChange(this.mode);
            });
        }
    }

    /**
     * Enable start button
     */
    enableStart() {
        if (this.startBtn) {
            this.startBtn.disabled = false;
        }
    }

    /**
     * Disable start button
     */
    disableStart() {
        if (this.startBtn) {
            this.startBtn.disabled = true;
        }
    }

    /**
     * Set monitoring state
     */
    setMonitoring(isMonitoring) {
        this.isMonitoring = isMonitoring;
        
        if (isMonitoring) {
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            this.modeToggle.disabled = true;
            
            // Update button text
            const btnText = this.startBtn.querySelector('.btn-icon + text');
            if (btnText) {
                btnText.textContent = 'Monitoring...';
            }
        } else {
            this.startBtn.disabled = false;
            this.stopBtn.disabled = true;
            this.modeToggle.disabled = false;
            this.exportBtn.disabled = true;
            
            // Reset button text
            const btnText = this.startBtn.querySelector('.btn-icon + text');
            if (btnText) {
                btnText.textContent = 'Start Monitoring';
            }
        }
        
        // Show/hide manual controls based on mode and monitoring state
        this.updateManualControls();
    }

    /**
     * Update manual controls visibility
     */
    updateManualControls() {
        if (this.manualControls) {
            if (this.mode === 'manual' && this.isMonitoring) {
                this.manualControls.style.display = 'flex';
                this.enableManualButtons();
            } else {
                this.manualControls.style.display = 'none';
            }
        }
    }

    /**
     * Enable manual control buttons
     */
    enableManualButtons() {
        if (this.captureBtn) this.captureBtn.disabled = false;
        if (this.analyzeBtn) this.analyzeBtn.disabled = false;
        if (this.nextBtn) this.nextBtn.disabled = false;
    }

    /**
     * Disable manual control buttons
     */
    disableManualButtons() {
        if (this.captureBtn) this.captureBtn.disabled = true;
        if (this.analyzeBtn) this.analyzeBtn.disabled = true;
        if (this.nextBtn) this.nextBtn.disabled = true;
    }

    /**
     * Enable export button
     */
    enableExport() {
        if (this.exportBtn) {
            this.exportBtn.disabled = false;
        }
    }

    /**
     * Handle mode change
     */
    onModeChange(mode) {
        console.log(`Mode changed to: ${mode}`);
        
        // Update UI based on mode
        const modeLabel = this.modeToggle.parentElement.querySelector('.toggle-label');
        if (modeLabel) {
            modeLabel.textContent = mode === 'manual' ? 'Manual Mode' : 'Automatic Mode';
        }
        
        // Update status bar
        const currentModeElement = document.getElementById('currentMode');
        if (currentModeElement) {
            currentModeElement.textContent = mode.charAt(0).toUpperCase() + mode.slice(1);
        }
        
        // Update manual controls visibility
        this.updateManualControls();
        
        // Trigger callback if set
        if (this.modeChangeCallback) {
            this.modeChangeCallback(mode);
        }
    }

    /**
     * Set callbacks
     */
    setCallbacks(callbacks) {
        if (this.startBtn && callbacks.onStart) {
            this.startBtn.addEventListener('click', callbacks.onStart);
        }
        
        if (this.stopBtn && callbacks.onStop) {
            this.stopBtn.addEventListener('click', callbacks.onStop);
        }
        
        if (this.captureBtn && callbacks.onCapture) {
            this.captureBtn.addEventListener('click', callbacks.onCapture);
        }
        
        if (this.analyzeBtn && callbacks.onAnalyze) {
            this.analyzeBtn.addEventListener('click', callbacks.onAnalyze);
        }
        
        if (this.nextBtn && callbacks.onNext) {
            this.nextBtn.addEventListener('click', callbacks.onNext);
        }
        
        if (this.exportBtn && callbacks.onExport) {
            this.exportBtn.addEventListener('click', callbacks.onExport);
        }
        
        if (callbacks.onModeChange) {
            this.modeChangeCallback = callbacks.onModeChange;
        }
    }

    /**
     * Update status display
     */
    updateStatus(status) {
        // Update timer
        if (status.timer_remaining !== undefined) {
            const timerElement = document.getElementById('timerValue');
            if (timerElement) {
                timerElement.textContent = status.timer_remaining > 0 
                    ? `${Math.round(status.timer_remaining)}s` 
                    : '--';
            }
        }
        
        // Update current agent
        if (status.current_agent) {
            const agentElement = document.getElementById('currentAgent');
            if (agentElement) {
                const agentNames = {
                    'initial_classifier': 'Initial Classifier',
                    'detail_extractor': 'Detail Extractor',
                    'damage_detector': 'Damage Detector',
                    'final_compiler': 'Final Compiler'
                };
                agentElement.textContent = agentNames[status.current_agent] || status.current_agent;
            }
        }
        
        // Update GPU usage
        if (status.gpu_usage !== undefined) {
            const gpuElement = document.getElementById('gpuUsage');
            if (gpuElement) {
                gpuElement.textContent = `${Math.round(status.gpu_usage)}%`;
            }
        }
    }

    /**
     * Update session ID display
     */
    updateSessionId(sessionId) {
        const sessionElement = document.getElementById('sessionId');
        if (sessionElement) {
            sessionElement.textContent = sessionId ? sessionId.substring(0, 8) + '...' : '--';
            sessionElement.title = sessionId || '';
        }
    }

    /**
     * Show final results
     */
    showFinalResults(results) {
        const finalResultsElement = document.getElementById('finalResults');
        if (!finalResultsElement) return;
        
        finalResultsElement.style.display = 'block';
        
        // Update result fields
        const fields = {
            'final-type': results.type,
            'final-color': results.color_primary || results.color,
            'final-pattern': results.pattern,
            'final-brand': results.brand,
            'final-size': results.size,
            'final-damage': results.has_damage ? `Yes - ${results.damage_type || 'Unknown'}` : 'No'
        };
        
        for (const [id, value] of Object.entries(fields)) {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value || '--';
            }
        }
        
        // Enable export button
        this.enableExport();
    }

    /**
     * Hide final results
     */
    hideFinalResults() {
        const finalResultsElement = document.getElementById('finalResults');
        if (finalResultsElement) {
            finalResultsElement.style.display = 'none';
        }
    }

    /**
     * Update metadata
     */
    updateMetadata(processingTime, framesAnalyzed, confidence) {
        const timeElement = document.getElementById('processingTime');
        if (timeElement) {
            timeElement.textContent = processingTime ? processingTime.toFixed(1) : '--';
        }
        
        const framesElement = document.getElementById('framesAnalyzed');
        if (framesElement) {
            framesElement.textContent = framesAnalyzed || '--';
        }
        
        const confidenceElement = document.getElementById('confidence');
        if (confidenceElement) {
            confidenceElement.textContent = confidence ? Math.round(confidence * 100) : '--';
        }
    }

    /**
     * Reset controls
     */
    reset() {
        this.setMonitoring(false);
        this.hideFinalResults();
        this.updateSessionId(null);
        this.updateStatus({
            timer_remaining: 0,
            current_agent: null,
            gpu_usage: 0
        });
    }
}

// Export for use in other modules
window.ControlPanel = ControlPanel;