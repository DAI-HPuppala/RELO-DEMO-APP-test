/**
 * Integrated Application Controller
 * Combines professional UI with all existing backend functionality
 */
class IntegratedApplicationController {
    constructor() {
        // Get WebRTC client from initialization
        this.webrtcClient = window.webrtcClient;
        
        // Initialize all components (both original and professional)
        this.streamViewer = new StreamViewer();
        this.statusMonitor = new StatusMonitor();
        this.agentMonitor = new AgentMonitor();
        
        // Initialize professional components
        this.csvExporter = new CSVExporter();
        this.progressMonitor = new ProgressMonitor();
        
        // Use professional results display if available, otherwise fallback
        if (window.ResultsDisplayProfessional) {
            this.resultsDisplay = new ResultsDisplayProfessional();
        } else {
            this.resultsDisplay = new ResultsDisplay();
        }
        
        // Store globally for access
        window.streamViewer = this.streamViewer;
        window.resultsDisplay = this.resultsDisplay;
        window.statusMonitor = this.statusMonitor;
        window.agentMonitor = this.agentMonitor;
        window.progressMonitor = this.progressMonitor;
        window.csvExporter = this.csvExporter;
        
        // State management
        this.sessionId = null;
        this.monitoringActive = false;
        this.mode = 'automatic';
        this.isPaused = false;
        this.startTime = null;
        this.processingTimer = null;
        
        // System monitoring
        this.systemMonitor = null;
        this.systemUptime = Date.now();
        this.setupSystemMonitoring();
        
        // User-friendly logs
        this.logsBuffer = [];
        this.maxLogs = 100;
        
        // Display initial stream if available
        if (window.initVideoStream) {
            this.streamViewer.setStream(window.initVideoStream);
        }
        
        this.init();
    }

    /**
     * Initialize application
     */
    async init() {
        console.log('[Integrated App] Initializing with professional UI and full backend integration');
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Set up WebRTC callbacks
        this.setupWebRTCCallbacks();
        
        // Update initial status
        this.statusMonitor.updateConnectionStatus(true);
        this.statusMonitor.updateSystemStatus('Ready');
        this.updateSystemStatus('Ready');
        
        // Check for existing sessions in localStorage
        if (this.resultsDisplay.getAllResults) {
            const sessions = this.resultsDisplay.getAllResults();
            console.log(`[Integrated App] Found ${sessions.length} existing sessions`);
        }
    }

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Control buttons
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const pauseBtn = document.getElementById('pauseBtn');
        const resumeBtn = document.getElementById('resumeBtn');
        
        if (startBtn) {
            startBtn.addEventListener('click', () => this.startMonitoring());
        }
        
        if (stopBtn) {
            stopBtn.addEventListener('click', () => this.stopMonitoring());
        }
        
        if (pauseBtn) {
            pauseBtn.addEventListener('click', () => this.pauseMonitoring());
        }
        
        if (resumeBtn) {
            resumeBtn.addEventListener('click', () => this.resumeMonitoring());
        }
        
        // Mode selector
        const modeRadios = document.querySelectorAll('input[name="mode"]');
        modeRadios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.mode = e.target.value;
                console.log('[Integrated App] Mode changed to:', this.mode);
                this.resultsDisplay.setMode && this.resultsDisplay.setMode(this.mode);
                this.agentMonitor.setMode && this.agentMonitor.setMode(this.mode);
                this.updateManualControls();
                this.updatePauseResumeButtons();
            });
        });
        
        // Export buttons
        const exportBtn = document.getElementById('exportBtn');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportResults());
        }
        
        // Clear progress button
        const clearProgressBtn = document.getElementById('clearProgressBtn');
        if (clearProgressBtn) {
            clearProgressBtn.addEventListener('click', () => {
                this.progressMonitor.clearLog();
            });
        }
    }

    /**
     * Set up WebRTC callbacks
     */
    setupWebRTCCallbacks() {
        // Stream events
        this.webrtcClient.on('stream', (stream) => {
            console.log('[Integrated App] Received video stream');
            this.streamViewer.setStream(stream);
            this.statusMonitor.updateStreamStatus(true);
            this.statusMonitor.updateCameraStatus(true);
            this.updateIndicator('streamIndicator', true);
            this.updateIndicator('cameraIndicator', true);
        });
        
        // Data channel messages
        this.webrtcClient.on('message', (data) => {
            this.handleDataChannelMessage(data);
        });
        
        // Connection events
        this.webrtcClient.on('connected', () => {
            console.log('[Integrated App] WebRTC connected');
            this.statusMonitor.updateConnectionStatus(true);
            this.updateIndicator('connectionIndicator', true);
        });
        
        this.webrtcClient.on('disconnected', () => {
            console.log('[Integrated App] WebRTC disconnected');
            this.statusMonitor.updateConnectionStatus(false);
            this.updateIndicator('connectionIndicator', false);
        });
        
        // Agent events
        this.webrtcClient.on('agentStarted', (data) => {
            this.handleAgentStarted(data);
        });
        
        this.webrtcClient.on('progressUpdate', (data) => {
            this.handleProgressUpdate(data);
        });
        
        this.webrtcClient.on('agentCompleted', (data) => {
            this.handleAgentCompleted(data);
        });
        
        this.webrtcClient.on('finalResults', (data) => {
            this.handleFinalResults(data);
        });
        
        // New V2 events
        this.webrtcClient.on('inferenceUpdate', (data) => {
            this.handleInferenceUpdate(data);
        });
        
        this.webrtcClient.on('inferenceResult', (data) => {
            this.handleInferenceResult(data);
        });
    }

    /**
     * Handle data channel messages
     */
    handleDataChannelMessage(data) {
        console.log('[Integrated App] Data channel message:', data.type);
        
        switch(data.type) {
            case 'agent_started':
                this.handleAgentStarted(data);
                break;
            case 'progress_update':
                this.handleProgressUpdate(data);
                break;
            case 'agent_completed':
                this.handleAgentCompleted(data);
                break;
            case 'final_results':
                this.handleFinalResults(data);
                break;
            case 'status_update':
                this.handleStatusUpdate(data);
                break;
            case 'flow_paused':
                this.handleFlowPaused(data);
                break;
            case 'flow_resumed':
                this.handleFlowResumed(data);
                break;
            case 'inference_update':
                this.handleInferenceUpdate(data);
                break;
            case 'inference_result':
                this.handleInferenceResult(data);
                break;
            default:
                console.log('Unknown message type:', data.type);
        }
    }

    /**
     * Start monitoring
     */
    async startMonitoring() {
        try {
            console.log('[Integrated App] Starting classification monitoring');
            
            // Reset components
            this.resultsDisplay.reset();
            this.progressMonitor.reset();
            this.statusMonitor.startMonitoring();
            this.agentMonitor.reset();
            this.agentMonitor.setMode(this.mode);
            
            // Start processing timer
            this.startTime = Date.now();
            this.startProcessingTimer();
            
            // Update UI state
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            if (startBtn) startBtn.disabled = true;
            if (stopBtn) stopBtn.disabled = false;
            
            this.monitoringActive = true;
            this.isPaused = false;
            this.streamViewer.hideOverlay();
            this.updatePauseResumeButtons();
            this.updateManualControls();
            
            // Start automatic mode if selected
            if (this.webrtcClient.sessionId && this.mode === 'automatic') {
                console.log('[Integrated App] Starting automatic mode');
                
                const agentTimers = window.AGENT_CONFIG ? 
                    window.AGENT_CONFIG.getTimers() : 
                    {
                        initial_classifier: 4.0,
                        detail_extractor: 3.0,
                        damage_detector: 4.0,
                        final_compiler: 2.0
                    };
                
                this.webrtcClient.startAutomatic(agentTimers);
                this.statusMonitor.updateSystemStatus('Processing - Automatic Mode');
                this.updateSystemStatus('Processing - Automatic Mode');
            } else if (this.mode === 'manual') {
                // Send mode change for manual mode
                if (this.webrtcClient.dataChannel && this.webrtcClient.dataChannel.readyState === 'open') {
                    this.webrtcClient.dataChannel.send(JSON.stringify({
                        type: 'mode_change',
                        mode: this.mode
                    }));
                }
                this.statusMonitor.updateSystemStatus('Manual Mode Active');
                this.updateSystemStatus('Ready - Manual Mode');
            }
            
            // Get or generate session ID
            this.sessionId = this.webrtcClient.sessionId || 'session_' + Date.now();
            this.statusMonitor.updateSessionId(this.sessionId);
            this.updateSessionId(this.sessionId);
            
            // Add start log
            this.addFriendlyLog('info', `Classification started in ${this.mode} mode`);
            
            // Reset pipeline flow
            ['initial_classifier', 'detail_extractor', 'damage_detector', 'final_compiler'].forEach(agent => {
                this.updatePipelineFlow(agent, '');
            });
            
            // Reset update counter
            const counterEl = document.getElementById('updateCounter');
            if (counterEl) counterEl.textContent = '0 inferences';
            
            // Initialize results table with placeholders
            this.initializeResultsTable();
            
        } catch (error) {
            console.error('[Integrated App] Failed to start monitoring:', error);
            this.showError('Failed to start monitoring: ' + error.message);
        }
    }

    /**
     * Stop monitoring
     */
    stopMonitoring() {
        console.log('[Integrated App] Stopping monitoring');
        
        this.monitoringActive = false;
        
        // Stop processing timer
        this.stopProcessingTimer();
        
        // Update UI state
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        if (startBtn) startBtn.disabled = false;
        if (stopBtn) stopBtn.disabled = true;
        
        // Send stop signal
        if (this.webrtcClient.dataChannel && this.webrtcClient.dataChannel.readyState === 'open') {
            this.webrtcClient.dataChannel.send(JSON.stringify({
                type: 'stop_session',
                session_id: this.sessionId || this.webrtcClient.sessionId
            }));
        }
        
        this.statusMonitor.stopMonitoring();
        this.streamViewer.showOverlay('Classification stopped');
        this.isPaused = false;
        this.updatePauseResumeButtons();
        this.updateSystemStatus('Stopped');
    }

    /**
     * Pause monitoring
     */
    pauseMonitoring() {
        if (this.mode !== 'automatic' || !this.monitoringActive || this.isPaused) {
            return;
        }
        
        console.log('[Integrated App] Pausing monitoring');
        this.isPaused = true;
        
        // Send pause signal
        if (this.webrtcClient.dataChannel && this.webrtcClient.dataChannel.readyState === 'open') {
            this.webrtcClient.dataChannel.send(JSON.stringify({
                type: 'pause_flow',
                session_id: this.sessionId
            }));
        }
        
        this.updatePauseResumeButtons();
        this.statusMonitor.updateSystemStatus('Paused');
        this.updateSystemStatus('Paused');
    }

    /**
     * Resume monitoring
     */
    resumeMonitoring() {
        if (this.mode !== 'automatic' || !this.monitoringActive || !this.isPaused) {
            return;
        }
        
        console.log('[Integrated App] Resuming monitoring');
        this.isPaused = false;
        
        // Send resume signal
        if (this.webrtcClient.dataChannel && this.webrtcClient.dataChannel.readyState === 'open') {
            this.webrtcClient.dataChannel.send(JSON.stringify({
                type: 'resume_flow',
                session_id: this.sessionId,
                restart_agent: false
            }));
        }
        
        this.updatePauseResumeButtons();
        this.statusMonitor.updateSystemStatus('Automatic Mode Active');
        this.updateSystemStatus('Processing - Automatic Mode');
    }

    /**
     * Handle agent started
     */
    handleAgentStarted(data) {
        console.log('[Integrated App] Agent started:', data.agent);
        
        // Update all monitoring components
        this.resultsDisplay.updateAgentProgress && this.resultsDisplay.updateAgentProgress(data.agent, 'Processing');
        this.statusMonitor.showProcessing(data.agent);
        this.agentMonitor.onAgentStarted(data);
        this.progressMonitor.onAgentStarted(data);
        
        this.updateSystemStatus(`Processing: ${data.agent}`);
        
        // Update pipeline flow visualization
        this.updatePipelineFlow(data.agent, 'active');
        
        // Update progressive updates panel
        this.updateAgentCard(data.agent, 'running');
        
        // Add user-friendly log
        const agentNames = {
            'initial_classifier': 'Initial Classifier',
            'detail_extractor': 'Detail Extractor',
            'damage_detector': 'Damage Detector',
            'final_compiler': 'Final Compiler'
        };
        this.addFriendlyLog('info', `${agentNames[data.agent] || data.agent} started analyzing clothing item`);
    }

    /**
     * Handle progress update
     */
    handleProgressUpdate(data) {
        console.log('[Integrated App] Progress update:', data.agent, data.progress);
        console.log('[FULL DATA]:', data);
        
        // Log progressive results
        if (data.partial_results) {
            console.log(`[PROGRESSIVE] Agent: ${data.agent}, Inference #${data.inference_num}`);
            console.log('[PROGRESSIVE] Results:', data.partial_results);
            
            // Update results display
            if (this.resultsDisplay && this.resultsDisplay.displayPartialResults) {
                this.resultsDisplay.displayPartialResults(data.agent, data.partial_results);
            }
            
            // Update agent card with progress - use the actual data
            this.updateAgentProgress(data.agent, data.inference_num, data.partial_results);
            
            // Also update the results table progressively
            this.updateResultsTablePartial(data.agent, data.partial_results);
        }
        
        // Update monitoring components
        if (this.agentMonitor) this.agentMonitor.onProgressUpdate(data);
        if (this.progressMonitor) this.progressMonitor.onProgressUpdate(data);
        
        // Update total inference counter
        const counterEl = document.getElementById('updateCounter');
        if (counterEl) {
            const current = parseInt(counterEl.textContent) || 0;
            counterEl.textContent = `${current + 1} inferences`;
        }
    }

    /**
     * Handle inference update (status only)
     */
    handleInferenceUpdate(data) {
        console.log('[Integrated App] Inference update:', data);
        // This just updates the counter and timer, no actual results yet
        this.updateAgentProgress(data.agent, data.inference_num, null);
        
        // Update total inference counter
        const counterEl = document.getElementById('updateCounter');
        if (counterEl) {
            const current = parseInt(counterEl.textContent) || 0;
            counterEl.textContent = `${current + 1} inferences`;
        }
    }
    
    /**
     * Handle inference result (actual attribute data)
     */
    handleInferenceResult(data) {
        console.log('[Integrated App] Inference result:', data);
        console.log('[PROGRESSIVE ATTRIBUTES]:', data.attributes);
        
        // Update agent progress with actual results
        if (data.attributes) {
            this.updateAgentProgress(data.agent, data.inference_num, data.attributes);
            
            // Update progress monitor if it exists
            if (this.progressMonitor && this.progressMonitor.onProgressUpdate) {
                this.progressMonitor.onProgressUpdate({
                    agent: data.agent,
                    inference_num: data.inference_num,
                    partial_results: data.attributes
                });
            }
            
            // Update results display with partial results
            if (this.resultsDisplay && this.resultsDisplay.displayPartialResults) {
                this.resultsDisplay.displayPartialResults(data.agent, data.attributes);
            }
            
            // Update the current row in the results table progressively
            this.updateResultsTablePartial(data.agent, data.attributes);
        }
    }

    /**
     * Handle agent completed
     */
    handleAgentCompleted(data) {
        console.log('[Integrated App] Agent completed:', data.agent);
        console.log('[FINALIZED] Agent results:', data.results);
        
        // Update all monitoring components
        this.resultsDisplay.updateAgentProgress && this.resultsDisplay.updateAgentProgress(data.agent, 'Completed');
        if (this.resultsDisplay.displayPartialResults && data.results) {
            this.resultsDisplay.displayPartialResults(data.agent, data.results);
        }
        
        this.agentMonitor.onAgentCompleted(data);
        this.progressMonitor.onAgentCompleted(data);
        
        // Update pipeline flow
        this.updatePipelineFlow(data.agent, 'complete');
        
        // Update agent card
        this.updateAgentCard(data.agent, 'complete');
        
        // Add friendly log
        const agentNames = {
            'initial_classifier': 'Initial Classifier',
            'detail_extractor': 'Detail Extractor',
            'damage_detector': 'Damage Detector',
            'final_compiler': 'Final Compiler'
        };
        this.addFriendlyLog('success', `${agentNames[data.agent] || data.agent} completed successfully`);
    }

    /**
     * Handle final results
     */
    handleFinalResults(data) {
        console.log('[Integrated App] Final classification results:', data);
        
        // Stop processing timer
        const processingTime = this.stopProcessingTimer();
        
        // Add metadata to results
        if (data.results) {
            data.results.processing_time = processingTime;
            data.results.session_id = this.sessionId;
            data.results.agents_completed = data.agents_completed;
        }
        
        // Display final results
        this.resultsDisplay.displayFinalResults(data.results);
        
        // Update the new results table
        this.updateResultsTable(data.results);
        
        // Add to CSV exporter
        this.csvExporter.addResult(data.results);
        
        // Update status
        this.statusMonitor.updateSystemStatus('Classification Complete');
        this.updateSystemStatus('Classification Complete');
        
        // Mark all pipeline nodes as complete
        ['initial_classifier', 'detail_extractor', 'damage_detector', 'final_compiler'].forEach(agent => {
            this.updatePipelineFlow(agent, 'complete');
        });
        
        // Add completion log
        this.addFriendlyLog('success', `Classification complete! Processed ${data.agents_completed || 4} agents successfully`);
        
        // Enable export buttons
        const exportBtn = document.getElementById('exportBtn');
        const exportCsvBtn = document.getElementById('exportCsvBtn');
        const exportJsonBtn = document.getElementById('exportJsonBtn');
        
        if (exportBtn) exportBtn.disabled = false;
        if (exportCsvBtn) exportCsvBtn.disabled = false;
        if (exportJsonBtn) exportJsonBtn.disabled = false;
    }

    /**
     * Handle status update
     */
    handleStatusUpdate(data) {
        if (data.session_id) {
            this.sessionId = data.session_id;
            this.statusMonitor.updateSessionId(data.session_id);
            this.updateSessionId(data.session_id);
        }
        
        if (data.status) {
            this.statusMonitor.updateSystemStatus(data.status);
            this.updateSystemStatus(data.status);
        }
    }

    /**
     * Handle flow paused
     */
    handleFlowPaused(data) {
        console.log('[Integrated App] Flow paused at:', data.paused_agent);
        this.isPaused = true;
        this.updatePauseResumeButtons();
        this.statusMonitor.updateSystemStatus(`Paused at ${data.paused_agent}`);
        this.updateSystemStatus(`Paused at ${data.paused_agent}`);
    }

    /**
     * Handle flow resumed
     */
    handleFlowResumed(data) {
        console.log('[Integrated App] Flow resumed at:', data.resuming_agent);
        this.isPaused = false;
        this.updatePauseResumeButtons();
        this.statusMonitor.updateSystemStatus('Automatic Mode Active');
        this.updateSystemStatus('Processing - Automatic Mode');
    }

    /**
     * Export results
     */
    exportResults() {
        console.log('[Integrated App] Exporting results');
        
        const results = this.resultsDisplay.getResults ? this.resultsDisplay.getResults() : this.resultsDisplay.currentResults;
        
        if (!results) {
            alert('No results to export');
            return;
        }
        
        const exportData = {
            session_id: this.sessionId,
            timestamp: new Date().toISOString(),
            mode: this.mode,
            results: results
        };
        
        // Create download link
        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `classification_${this.sessionId || 'results'}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    /**
     * Update pause/resume button visibility
     */
    updatePauseResumeButtons() {
        const pauseBtn = document.getElementById('pauseBtn');
        const resumeBtn = document.getElementById('resumeBtn');
        
        if (!pauseBtn || !resumeBtn) return;
        
        if (this.mode === 'automatic' && this.monitoringActive) {
            if (this.isPaused) {
                pauseBtn.style.display = 'none';
                resumeBtn.style.display = 'inline-flex';
                resumeBtn.disabled = false;
            } else {
                pauseBtn.style.display = 'inline-flex';
                pauseBtn.disabled = false;
                resumeBtn.style.display = 'none';
            }
        } else {
            pauseBtn.style.display = 'none';
            resumeBtn.style.display = 'none';
        }
    }

    /**
     * Update manual controls visibility
     */
    updateManualControls() {
        const manualControls = document.getElementById('manualControls');
        if (manualControls) {
            manualControls.style.display = 
                (this.mode === 'manual' && this.monitoringActive) ? 'block' : 'none';
        }
    }

    /**
     * Update indicator status
     */
    updateIndicator(indicatorId, active) {
        const indicator = document.getElementById(indicatorId);
        if (indicator) {
            if (active) {
                indicator.classList.add('active');
            } else {
                indicator.classList.remove('active');
            }
        }
    }

    /**
     * Update system status
     */
    updateSystemStatus(status) {
        const element = document.getElementById('systemStatus');
        if (element) {
            element.textContent = status;
        }
    }

    /**
     * Update session ID
     */
    updateSessionId(id) {
        const element = document.getElementById('sessionId');
        if (element) {
            element.textContent = id;
        }
    }

    /**
     * Start processing timer
     */
    startProcessingTimer() {
        this.startTime = Date.now();
        
        // Update every 100ms
        this.processingTimer = setInterval(() => {
            const elapsed = (Date.now() - this.startTime) / 1000;
            const element = document.getElementById('processingTime');
            if (element) {
                element.textContent = elapsed.toFixed(1) + 's';
            }
        }, 100);
    }

    /**
     * Stop processing timer
     */
    stopProcessingTimer() {
        if (this.processingTimer) {
            clearInterval(this.processingTimer);
            this.processingTimer = null;
        }
        
        const elapsed = this.startTime ? (Date.now() - this.startTime) / 1000 : 0;
        return elapsed.toFixed(1);
    }

    /**
     * Show error message
     */
    showError(message) {
        console.error('[Integrated App] Error:', message);
        this.streamViewer.showOverlay(`Error: ${message}`);
        this.statusMonitor.showError(message);
        this.updateSystemStatus('Error: ' + message);
        this.addFriendlyLog('error', message);
    }
    
    /**
     * Set up system monitoring
     */
    setupSystemMonitoring() {
        // Update uptime
        setInterval(() => {
            const uptime = Date.now() - this.systemUptime;
            const hours = Math.floor(uptime / 3600000);
            const minutes = Math.floor((uptime % 3600000) / 60000);
            const seconds = Math.floor((uptime % 60000) / 1000);
            const uptimeStr = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            const element = document.getElementById('systemUptime');
            if (element) element.textContent = uptimeStr;
        }, 1000);
        
        // Simulate system stats (in real app, these would come from backend)
        this.systemMonitor = setInterval(() => {
            this.updateSystemStats();
        }, 2000);
        
        // Set up logs panel
        const clearLogsBtn = document.getElementById('clearLogsBtn');
        if (clearLogsBtn) {
            clearLogsBtn.addEventListener('click', () => this.clearLogs());
        }
        
        // Add initial log
        this.addFriendlyLog('info', 'System ready - waiting for classification to start');
    }
    
    /**
     * Update system stats (simulated for demo)
     */
    updateSystemStats() {
        // Simulate CPU usage (20-60%)
        const cpuUsage = 20 + Math.random() * 40;
        const cpuElement = document.getElementById('cpuUsage');
        const cpuPercent = document.getElementById('cpuPercent');
        if (cpuElement) cpuElement.style.width = cpuUsage + '%';
        if (cpuPercent) cpuPercent.textContent = Math.round(cpuUsage) + '%';
        
        // Simulate GPU usage (higher when processing)
        const gpuUsage = this.monitoringActive ? 40 + Math.random() * 40 : 5 + Math.random() * 10;
        const gpuElement = document.getElementById('gpuUsage');
        const gpuPercent = document.getElementById('gpuPercent');
        if (gpuElement) gpuElement.style.width = gpuUsage + '%';
        if (gpuPercent) gpuPercent.textContent = Math.round(gpuUsage) + '%';
        
        // Simulate memory usage (30-70%)
        const memUsage = 30 + Math.random() * 40;
        const memElement = document.getElementById('memUsage');
        const memPercent = document.getElementById('memPercent');
        if (memElement) memElement.style.width = memUsage + '%';
        if (memPercent) memPercent.textContent = Math.round(memUsage) + '%';
        
        // Update processing speed
        const fps = this.monitoringActive ? 15 + Math.random() * 10 : 0;
        const speedElement = document.getElementById('processingSpeed');
        if (speedElement) speedElement.textContent = Math.round(fps);
    }
    
    /**
     * Add user-friendly log entry
     */
    addFriendlyLog(type, message) {
        const time = new Date().toLocaleTimeString('en-US', { hour12: false });
        const icons = {
            info: 'ℹ️',
            success: '✅',
            warning: '⚠️',
            error: '❌'
        };
        
        const logEntry = {
            time,
            type,
            icon: icons[type] || 'ℹ️',
            message
        };
        
        this.logsBuffer.push(logEntry);
        if (this.logsBuffer.length > this.maxLogs) {
            this.logsBuffer.shift();
        }
        
        // Add to UI
        const logsContainer = document.getElementById('friendlyLogs');
        if (logsContainer) {
            const entryEl = document.createElement('div');
            entryEl.className = `log-entry ${type}`;
            entryEl.innerHTML = `
                <span class="log-time">${time}</span>
                <span class="log-icon">${logEntry.icon}</span>
                <span class="log-message">${message}</span>
            `;
            logsContainer.appendChild(entryEl);
            
            // Auto-scroll to bottom
            logsContainer.scrollTop = logsContainer.scrollHeight;
            
            // Limit visible logs
            while (logsContainer.children.length > 50) {
                logsContainer.removeChild(logsContainer.firstChild);
            }
        }
    }
    
    /**
     * Clear logs
     */
    clearLogs() {
        this.logsBuffer = [];
        const logsContainer = document.getElementById('friendlyLogs');
        if (logsContainer) {
            logsContainer.innerHTML = `
                <div class="log-entry info">
                    <span class="log-time">${new Date().toLocaleTimeString('en-US', { hour12: false })}</span>
                    <span class="log-icon">ℹ️</span>
                    <span class="log-message">Logs cleared</span>
                </div>
            `;
        }
    }
    
    /**
     * Update pipeline flow visualization
     */
    updatePipelineFlow(agent, status) {
        const nodeMap = {
            'initial_classifier': 'flow-initial',
            'detail_extractor': 'flow-detail',
            'damage_detector': 'flow-damage',
            'final_compiler': 'flow-final',
            'initial': 'flow-initial',
            'detail': 'flow-detail',
            'damage': 'flow-damage',
            'final': 'flow-final'
        };
        
        const nodeId = nodeMap[agent];
        if (nodeId) {
            const node = document.getElementById(nodeId);
            if (node) {
                // Remove all status classes
                node.classList.remove('active', 'complete');
                
                // Add appropriate class
                if (status === 'active') {
                    node.classList.add('active');
                } else if (status === 'complete') {
                    node.classList.add('complete');
                }
            }
        }
    }
    
    /**
     * Update agent card in progressive updates panel
     */
    updateAgentCard(agent, status) {
        const agentMap = {
            'initial_classifier': 'initial',
            'detail_extractor': 'detail',
            'damage_detector': 'damage',
            'final_compiler': 'final',
            'initial': 'initial',
            'detail': 'detail',
            'damage': 'damage',
            'final': 'final'
        };
        
        const agentKey = agentMap[agent] || agent;
        const card = document.querySelector(`[data-agent="${agentKey}"]`);
        
        if (card) {
            // Update status
            const statusEl = document.getElementById(`${agentKey}-status`);
            if (statusEl) {
                statusEl.textContent = status === 'running' ? 'Processing' : 
                                      status === 'complete' ? 'Complete' : 'Waiting';
                statusEl.className = `update-status ${status}`;
            }
            
            // Add active class to card
            if (status === 'running') {
                card.classList.add('active');
            } else {
                card.classList.remove('active');
            }
        }
    }
    
    /**
     * Update agent progress in updates panel
     */
    updateAgentProgress(agent, inferenceNum, results) {
        // Map both long and short agent names
        const agentMap = {
            'initial_classifier': 'initial',
            'detail_extractor': 'detail',
            'damage_detector': 'damage',
            'final_compiler': 'final',
            'initial': 'initial',
            'detail': 'detail',
            'damage': 'damage',
            'final': 'final'
        };
        
        const agentKey = agentMap[agent] || agent;
        console.log(`[updateAgentProgress] Agent: ${agent} -> Key: ${agentKey}, Results:`, results);
        
        // Update inference count
        const countEl = document.getElementById(`${agentKey}-count`);
        if (countEl) {
            countEl.textContent = inferenceNum || 0;
        }
        
        // Update progress bar (assuming 4 seconds per agent, ~5 inferences)
        const progressEl = document.getElementById(`${agentKey}-progress`);
        if (progressEl && inferenceNum) {
            const progress = Math.min((inferenceNum / 5) * 100, 100);
            progressEl.style.width = `${progress}%`;
        }
        
        // Update timer
        const timerEl = document.getElementById(`${agentKey}-timer`);
        if (timerEl && this.startTime) {
            const elapsed = ((Date.now() - this.startTime) / 1000).toFixed(1);
            timerEl.textContent = `${elapsed}s`;
        }
        
        // Update preview with latest results
        if (results) {
            const previewEl = document.getElementById(`${agentKey}-preview`);
            if (previewEl) {
                const resultText = this.formatResultPreview(results);
                console.log(`[updateAgentProgress] Setting preview for ${agentKey}-preview:`, resultText);
                previewEl.textContent = resultText;
                // Also add a visual indicator that data was received
                previewEl.style.color = '#28a745';
                setTimeout(() => {
                    previewEl.style.color = '';
                }, 1000);
            } else {
                console.error(`[updateAgentProgress] Could not find element: ${agentKey}-preview`);
            }
        }
    }
    
    /**
     * Format result preview for display
     */
    formatResultPreview(results) {
        if (!results) return '';
        
        // Extract key attributes - handle various naming conventions from backend
        const items = [];
        
        // Type (item_type, clothing_type, or type)
        if (results.item_type || results.clothing_type || results.type) 
            items.push(`Type: ${results.item_type || results.clothing_type || results.type}`);
            
        // Color (color or primary_color)
        if (results.color || results.primary_color) 
            items.push(`Color: ${results.color || results.primary_color}`);
            
        // Pattern
        if (results.pattern) 
            items.push(`Pattern: ${results.pattern}`);
            
        // Damage
        if (results.has_damage !== undefined) 
            items.push(`Damage: ${results.has_damage ? 'Yes' : 'No'}`);
        if (results.damage_type) 
            items.push(`Damage Type: ${results.damage_type}`);
            
        // Other attributes
        if (results.brand) items.push(`Brand: ${results.brand}`);
        if (results.size) items.push(`Size: ${results.size}`);
        if (results.neckline) items.push(`Neckline: ${results.neckline}`);
        if (results.sleeve_type || results.sleeve_length) 
            items.push(`Sleeve: ${results.sleeve_type || results.sleeve_length}`);
        if (results.closure_type) items.push(`Closure: ${results.closure_type}`);
        
        return items.slice(0, 4).join(' | ');
    }
    
    /**
     * Update results table partially as data comes in
     */
    updateResultsTablePartial(agent, attributes) {
        const row = document.getElementById('current-classification-row');
        if (!row || !attributes) return;
        
        // Get cells
        const cells = row.getElementsByTagName('td');
        
        // Update cells based on agent and attributes
        const getValue = (attr) => {
            if (attr === null || attr === undefined || attr === '') return '-';
            return attr;
        };
        
        // Map agent to the attributes they provide - handle various attribute names
        if (agent === 'initial_classifier' || agent === 'initial') {
            // Type - could be item_type, clothing_type, or type
            if (attributes.item_type || attributes.clothing_type || attributes.type) {
                cells[0].textContent = getValue(attributes.item_type || attributes.clothing_type || attributes.type);
            }
            // Color - could be color or primary_color
            if (attributes.color || attributes.primary_color) {
                cells[1].textContent = getValue(attributes.color || attributes.primary_color);
            }
            if (attributes.pattern) cells[2].textContent = getValue(attributes.pattern);
        } else if (agent === 'detail_extractor' || agent === 'detail') {
            if (attributes.brand) cells[3].textContent = getValue(attributes.brand);
            if (attributes.size) cells[4].textContent = getValue(attributes.size);
            if (attributes.neckline) cells[5].textContent = getValue(attributes.neckline);
            if (attributes.sleeve_length) cells[6].textContent = getValue(attributes.sleeve_length);
            if (attributes.closure_type) cells[7].textContent = getValue(attributes.closure_type);
        } else if (agent === 'damage_detector' || agent === 'damage') {
            if (attributes.has_damage !== undefined) {
                cells[8].textContent = attributes.has_damage ? 'Yes' : 'No';
            }
            if (attributes.damage_type) cells[9].textContent = getValue(attributes.damage_type);
        }
        
        // Remove loading class from updated cells
        for (let cell of cells) {
            if (cell.textContent !== '-' && cell.textContent !== 'Processing...') {
                cell.classList.remove('result-loading');
            }
        }
    }
    
    /**
     * Initialize results table with placeholders
     */
    initializeResultsTable() {
        const tbody = document.getElementById('resultsTableBody');
        if (!tbody) return;
        
        // Clear any existing content
        tbody.innerHTML = '';
        
        // Create a row with loading placeholders
        const row = document.createElement('tr');
        row.id = 'current-classification-row';
        row.innerHTML = `
            <td class="result-loading">-</td>
            <td class="result-loading">-</td>
            <td class="result-loading">-</td>
            <td class="result-loading">-</td>
            <td class="result-loading">-</td>
            <td class="result-loading">-</td>
            <td class="result-loading">-</td>
            <td class="result-loading">-</td>
            <td class="result-loading">-</td>
            <td class="result-loading">-</td>
            <td class="result-loading">Processing...</td>
        `;
        
        tbody.appendChild(row);
    }
    
    /**
     * Update final results table
     */
    updateResultsTable(results) {
        const tbody = document.getElementById('resultsTableBody');
        if (!tbody) return;
        
        // First, update the current row's timestamp if it exists
        const currentRow = document.getElementById('current-classification-row');
        if (currentRow) {
            const cells = currentRow.getElementsByTagName('td');
            if (cells.length > 10) {
                cells[10].textContent = new Date().toLocaleTimeString('en-US', { hour12: false });
                cells[10].classList.remove('result-loading');
            }
            
            // Also update any missing cells with final data
            this.updateResultsTablePartial('final', results);
            
            // Remove the ID so the next classification gets a new row
            currentRow.id = '';
            return;
        }
        
        // Clear empty row
        if (tbody.querySelector('.empty-row')) {
            tbody.innerHTML = '';
        }
        
        // Create new row - handle both snake_case from backend and display names
        const row = document.createElement('tr');
        const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });
        
        // Map backend attribute names to display values
        const getValue = (attr) => {
            if (attr === null || attr === undefined || attr === '') return '-';
            return attr;
        };
        
        row.innerHTML = `
            <td>${getValue(results.item_type || results.clothing_type || results.type)}</td>
            <td>${getValue(results.color || results.primary_color)}</td>
            <td>${getValue(results.pattern)}</td>
            <td>${getValue(results.brand)}</td>
            <td>${getValue(results.size)}</td>
            <td>${getValue(results.neckline)}</td>
            <td>${getValue(results.sleeve_type || results.sleeve_length || results.sleeve)}</td>
            <td>${getValue(results.closure_type || results.closure)}</td>
            <td>${results.has_damage === true ? 'Yes' : results.has_damage === false ? 'No' : '-'}</td>
            <td>${getValue(results.damage_type)}</td>
            <td>${timestamp}</td>
        `;
        
        tbody.appendChild(row);
        
        // Enable export buttons
        const csvBtn = document.getElementById('exportCsvBtn');
        const jsonBtn = document.getElementById('exportJsonBtn');
        if (csvBtn) csvBtn.disabled = false;
        if (jsonBtn) jsonBtn.disabled = false;
    }
}

/**
 * Trigger agent manually (for compatibility)
 */
window.triggerAgent = function(agentType) {
    if (window.webrtcClient && window.webrtcClient.getConnectionStatus()) {
        window.webrtcClient.triggerAgent(agentType);
        
        // Update UI to show processing
        if (window.resultsDisplay) {
            window.resultsDisplay.updateAgentProgress && window.resultsDisplay.updateAgentProgress(agentType, 'Processing');
        }
        if (window.progressMonitor) {
            window.progressMonitor.onAgentStarted({ agent: agentType });
        }
    } else {
        console.error('WebRTC client not connected');
        alert('Please start monitoring first');
    }
};

/**
 * Initialize the integrated app after initialization completes
 */
window.initializeApp = function() {
    console.log('[Integrated App] Starting integrated application with professional UI');
    window.appController = new IntegratedApplicationController();
};

// Do NOT auto-initialize - let InitializationManager handle it