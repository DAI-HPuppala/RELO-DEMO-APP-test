/**
 * Main Application Controller
 * This is initialized AFTER the initialization manager completes
 */
class ApplicationController {
    constructor() {
        // Get the WebRTC client that was initialized by InitializationManager
        this.webrtcClient = window.webrtcClient;
        
        // Initialize components
        this.streamViewer = new StreamViewer();
        this.resultsDisplay = new ResultsDisplay();
        this.statusMonitor = new StatusMonitor();
        this.agentMonitor = new AgentMonitor();
        
        // Store globally for manual controls
        window.streamViewer = this.streamViewer;
        window.resultsDisplay = this.resultsDisplay;
        window.statusMonitor = this.statusMonitor;
        window.agentMonitor = this.agentMonitor;
        
        this.sessionId = null;
        this.monitoringActive = false;
        this.mode = 'automatic';
        this.isPaused = false;
        
        // If there's already a video stream from initialization, display it
        if (window.initVideoStream) {
            this.streamViewer.setStream(window.initVideoStream);
        }
        
        this.init();
    }

    /**
     * Initialize the application
     */
    async init() {
        console.log('Initializing main application after successful startup');
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Set up WebRTC callbacks
        this.setupWebRTCCallbacks();
        
        // Update initial status
        this.statusMonitor.updateConnectionStatus(true);
        this.statusMonitor.updateSystemStatus('Ready');
    }

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Start/Stop buttons
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        
        if (startBtn) {
            startBtn.addEventListener('click', () => this.startMonitoring());
        }
        
        if (stopBtn) {
            stopBtn.addEventListener('click', () => this.stopMonitoring());
        }
        
        // Mode switch
        const modeRadios = document.querySelectorAll('input[name="mode"]');
        modeRadios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.mode = e.target.value;
                console.log('[App] Mode changed to:', this.mode);
                this.resultsDisplay.setMode(this.mode);
                this.agentMonitor.setMode(this.mode);
                this.updateManualControls();
                this.updatePauseResumeButtons();
            });
        });
        
        // Pause/Resume buttons (only for automatic mode)
        const pauseBtn = document.getElementById('pauseBtn');
        const resumeBtn = document.getElementById('resumeBtn');
        
        if (pauseBtn) {
            pauseBtn.addEventListener('click', () => this.pauseMonitoring());
        }
        
        if (resumeBtn) {
            resumeBtn.addEventListener('click', () => this.resumeMonitoring());
        }
        
        // Export button
        const exportBtn = document.getElementById('exportBtn');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportResults());
        }
        
        // Manual mode controls (if they exist)
        this.setupManualControls();
    }

    /**
     * Set up WebRTC callbacks
     */
    setupWebRTCCallbacks() {
        // Set up callbacks using the WebRTC client's callback system
        this.webrtcClient.on('stream', (stream) => {
            console.log('[App] Received video stream');
            this.streamViewer.setStream(stream);
            this.statusMonitor.updateStreamStatus(true);
            this.statusMonitor.updateCameraStatus(true);
        });
        
        // Handle data channel messages through the callback
        this.webrtcClient.on('message', (data) => {
            this.handleDataChannelMessage(data);
        });
        
        // Handle connection events
        this.webrtcClient.on('connected', () => {
            console.log('[App] WebRTC connected');
            this.statusMonitor.updateConnectionStatus(true);
        });
        
        this.webrtcClient.on('disconnected', () => {
            console.log('[App] WebRTC disconnected');
            this.statusMonitor.updateConnectionStatus(false);
        });
        
        // Handle specific agent events
        this.webrtcClient.on('agentStarted', (data) => {
            this.handleAgentStarted(data);
        });
        
        this.webrtcClient.on('progressUpdate', (data) => {
            this.handleAgentProgress(data);
        });
        
        this.webrtcClient.on('agentCompleted', (data) => {
            this.handleAgentCompleted(data);
        });
        
        this.webrtcClient.on('finalResults', (data) => {
            this.handleFinalResults(data);
        });
    }
    
    /**
     * Handle data channel messages
     */
    handleDataChannelMessage(data) {
        console.log('[App] Data channel message:', data.type);
        
        switch(data.type) {
            case 'agent_started':
                this.handleAgentStarted(data);
                break;
            case 'progress_update':
                this.handleAgentProgress(data);
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
            default:
                console.log('Unknown message type:', data.type);
        }
    }


    /**
     * Start monitoring
     */
    async startMonitoring() {
        try {
            console.log('Starting monitoring...');
            
            // Reset UI
            this.resultsDisplay.reset();
            this.statusMonitor.startMonitoring();
            this.agentMonitor.reset();
            this.agentMonitor.setMode(this.mode);
            
            // Update UI state
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            if (startBtn) startBtn.disabled = true;
            if (stopBtn) stopBtn.disabled = false;
            
            this.monitoringActive = true;
            this.isPaused = false;
            this.streamViewer.hideOverlay();
            this.updateManualControls();
            this.updatePauseResumeButtons();
            
            // Check if we need to create a new session (after stop or if no session exists)
            if (!this.webrtcClient.sessionId) {
                console.log('[App] Creating new session for monitoring');
                await this.webrtcClient.startSession(this.mode, 'realsense');
                // Wait for session to be created
                await new Promise(resolve => setTimeout(resolve, 500));
            }
            
            // Start monitoring based on mode
            if (this.webrtcClient.sessionId) {
                // Get agent timers configuration
                const agentTimers = window.AGENT_CONFIG ? 
                    window.AGENT_CONFIG.getTimers() : 
                    {
                        initial_classifier: 4.0,
                        detail_extractor: 3.0,
                        damage_detector: 4.0,
                        final_compiler: 2.0
                    };
                
                if (this.mode === 'automatic') {
                    console.log('Starting automatic mode with agent timers');
                    console.log('[App] Starting automatic mode with timers:', agentTimers);
                    this.webrtcClient.startAutomatic(agentTimers, 'automatic');
                    this.statusMonitor.updateSystemStatus('Automatic Mode Active');
                } else if (this.mode === 'manual') {
                    console.log('Starting manual mode with agent timers');
                    console.log('[App] Starting manual mode with timers:', agentTimers);
                    // Start manual mode monitoring - this triggers the backend to initialize manual mode
                    this.webrtcClient.startAutomatic(agentTimers, 'manual');
                    this.statusMonitor.updateSystemStatus('Manual Mode - Waiting for Command');
                    
                    // Setup manual control button handlers if not already done
                    if (!this.manualHandlersSetup) {
                        this.setupManualButtonHandlers();
                        this.manualHandlersSetup = true;
                    }
                    
                    // Enable manual buttons
                    this.enableManualButtons();
                } else {
                    // Fallback for other modes
                    this.statusMonitor.updateSystemStatus('Monitoring Active');
                }
            } else {
                this.statusMonitor.updateSystemStatus('Session Not Ready');
            }
            
        } catch (error) {
            console.error('Failed to start monitoring:', error);
            this.showError('Failed to start monitoring');
        }
    }

    /**
     * Pause monitoring (only in automatic mode)
     */
    pauseMonitoring() {
        if (this.mode !== 'automatic' || !this.monitoringActive || this.isPaused) {
            console.log('Cannot pause - wrong mode or state');
            return;
        }
        
        console.log('Pausing monitoring...');
        this.isPaused = true;
        
        // Send pause signal via data channel
        if (this.webrtcClient.dataChannel && this.webrtcClient.dataChannel.readyState === 'open') {
            this.webrtcClient.dataChannel.send(JSON.stringify({
                type: 'pause_flow',
                session_id: this.sessionId
            }));
        }
        
        this.updatePauseResumeButtons();
        this.statusMonitor.updateSystemStatus('Paused');
    }
    
    /**
     * Resume monitoring (only in automatic mode)
     */
    resumeMonitoring() {
        if (this.mode !== 'automatic' || !this.monitoringActive || !this.isPaused) {
            console.log('Cannot resume - wrong mode or state');
            return;
        }
        
        console.log('Resuming monitoring...');
        this.isPaused = false;
        
        // Send resume signal via data channel
        if (this.webrtcClient.dataChannel && this.webrtcClient.dataChannel.readyState === 'open') {
            this.webrtcClient.dataChannel.send(JSON.stringify({
                type: 'resume_flow',
                session_id: this.sessionId,
                restart_agent: false  // Don't restart the current agent
            }));
        }
        
        this.updatePauseResumeButtons();
        this.statusMonitor.updateSystemStatus('Automatic Mode Active');
    }

    /**
     * Stop monitoring
     */
    stopMonitoring() {
        console.log('Stopping monitoring...');
        
        this.monitoringActive = false;
        
        // Update UI state
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        if (startBtn) startBtn.disabled = false;
        if (stopBtn) stopBtn.disabled = true;
        
        // Disable manual control buttons
        this.disableManualButtons();
        
        // Send stop signal via data channel with session_id
        if (this.webrtcClient.dataChannel && this.webrtcClient.dataChannel.readyState === 'open') {
            this.webrtcClient.dataChannel.send(JSON.stringify({
                type: 'stop_session',
                session_id: this.sessionId || this.webrtcClient.sessionId
            }));
        }
        
        // Clear the session ID to force new session creation on next start
        console.log('[App] Clearing session ID after stop');
        this.webrtcClient.sessionId = null;
        this.sessionId = null;
        
        this.statusMonitor.stopMonitoring();
        this.streamViewer.showOverlay('Monitoring stopped');
        this.isPaused = false;
        this.updatePauseResumeButtons();
        this.updateManualControls();
    }
    
    /**
     * Disable manual control buttons
     */
    disableManualButtons() {
        const buttons = ['prevBtn', 'nextBtn', 'redoBtn', 'triggerAllBtn'];
        buttons.forEach(btnId => {
            const btn = document.getElementById(btnId);
            if (btn) {
                btn.disabled = true;
            }
        });
    }
    
    /**
     * Enable manual control buttons
     */
    enableManualButtons() {
        const buttons = ['prevBtn', 'nextBtn', 'redoBtn', 'triggerAllBtn'];
        buttons.forEach(btnId => {
            const btn = document.getElementById(btnId);
            if (btn) {
                btn.disabled = false;
            }
        });
    }


    /**
     * Export results
     */
    async exportResults() {
        console.log('Exporting results');
        
        const results = this.resultsDisplay.getResults();
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
     * Handle status update
     */
    handleStatusUpdate(data) {
        if (data.session_id) {
            this.sessionId = data.session_id;
            this.statusMonitor.updateSessionId(data.session_id);
        }
        
        if (data.status) {
            this.statusMonitor.updateSystemStatus(data.status);
        }
    }

    /**
     * Handle agent started
     */
    handleAgentStarted(data) {
        console.log('[App] Agent started:', data.agent);
        this.resultsDisplay.updateAgentProgress(data.agent, 'Processing');
        this.statusMonitor.showProcessing(data.agent);
        this.agentMonitor.onAgentStarted(data);
    }
    
    /**
     * Handle agent progress
     */
    handleAgentProgress(data) {
        console.log('[App] Agent progress:', data.agent, data.progress);
        
        // Log detailed progressive classification results
        if (data.partial_results) {
            console.log(`[PROGRESSIVE RESULTS] Agent: ${data.agent}, Inference #${data.inference_num || data.progress}`);
            console.log('[PROGRESSIVE RESULTS] Attributes received:', data.partial_results);
            console.log('[PROGRESSIVE RESULTS] Full message:', JSON.stringify(data, null, 2));
            
            this.resultsDisplay.displayPartialResults(data.agent, data.partial_results);
        }
        
        this.agentMonitor.onProgressUpdate(data);
    }

    /**
     * Handle agent completed
     */
    handleAgentCompleted(data) {
        console.log('[App] Agent completed:', data.agent, data.results);
        
        // Log detailed finalized classification results for this agent
        console.log(`[AGENT FINALIZED] Agent: ${data.agent} has completed all inferences`);
        console.log('[AGENT FINALIZED] Final attributes for this agent:', data.results);
        console.log('[AGENT FINALIZED] Full message:', JSON.stringify(data, null, 2));
        
        this.resultsDisplay.updateAgentProgress(data.agent, 'Completed');
        this.resultsDisplay.displayPartialResults(data.agent, data.results);
        this.agentMonitor.onAgentCompleted(data);
    }

    /**
     * Handle flow paused
     */
    handleFlowPaused(data) {
        console.log('[App] Flow paused at agent:', data.paused_agent);
        this.isPaused = true;
        this.updatePauseResumeButtons();
        this.statusMonitor.updateSystemStatus(`Paused at ${data.paused_agent || 'agent'}`);
    }
    
    /**
     * Handle flow resumed
     */
    handleFlowResumed(data) {
        console.log('[App] Flow resumed at agent:', data.resuming_agent);
        this.isPaused = false;
        this.updatePauseResumeButtons();
        this.statusMonitor.updateSystemStatus('Automatic Mode Active');
    }

    /**
     * Handle final results
     */
    handleFinalResults(data) {
        console.log('[App] Final results received:', data);
        
        // Log detailed final classification results
        console.log('[FINAL CLASSIFICATION] Complete classification results from all agents');
        console.log('[FINAL CLASSIFICATION] Final attributes:', data.results);
        console.log('[FINAL CLASSIFICATION] Processing time:', data.processing_time);
        console.log('[FINAL CLASSIFICATION] Agents completed:', data.agents_completed);
        console.log('[FINAL CLASSIFICATION] Full message:', JSON.stringify(data, null, 2));
        
        this.resultsDisplay.displayFinalResults(data.results);
        this.statusMonitor.updateSystemStatus('Classification Complete');
        
        // Enable export button
        const exportBtn = document.getElementById('exportBtn');
        if (exportBtn) {
            exportBtn.disabled = false;
        }
    }


    /**
     * Set up manual mode controls
     */
    setupManualControls() {
        // Add click handlers to agent cards for manual triggering
        if (this.agentMonitor && this.agentMonitor.agents) {
            this.agentMonitor.agents.forEach(agent => {
                const card = document.getElementById(`agent-${agent.id}`);
                if (card) {
                    card.style.cursor = 'pointer';
                    card.addEventListener('click', () => {
                        if (this.mode === 'manual' && this.monitoringActive) {
                            console.log(`[App] Manual trigger for agent: ${agent.id}`);
                            this.webrtcClient.triggerAnalysis(agent.id);
                            // Update UI to show processing
                            this.agentMonitor.onAgentStarted({
                                agent: agent.id,
                                mode: 'manual'
                            });
                        }
                    });
                }
            });
        }
        
        // Add trigger all button for manual mode
        const triggerAllBtn = document.createElement('button');
        triggerAllBtn.id = 'triggerAllBtn';
        triggerAllBtn.className = 'btn btn-secondary';
        triggerAllBtn.innerHTML = 'Run All Agents';
        triggerAllBtn.style.display = 'none';
        
        const controlsSection = document.querySelector('.controls-section');
        if (controlsSection) {
            controlsSection.appendChild(triggerAllBtn);
            
            triggerAllBtn.addEventListener('click', () => {
                if (this.mode === 'manual' && this.monitoringActive) {
                    console.log('[App] Triggering all agents in manual mode');
                    this.webrtcClient.triggerAllAgents();
                }
            });
        }
    }
    
    /**
     * Show/hide manual controls based on mode
     */
    updateManualControls() {
        const triggerAllBtn = document.getElementById('triggerAllBtn');
        if (triggerAllBtn) {
            triggerAllBtn.style.display = 
                (this.mode === 'manual' && this.monitoringActive) ? 'inline-block' : 'none';
        }
    }
    
    /**
     * Setup manual button handlers for manual mode navigation
     */
    setupManualButtonHandlers() {
        console.log('[App] Setting up manual button handlers');
        
        // Previous button
        const prevBtn = document.getElementById('prevBtn');
        if (prevBtn && !prevBtn.hasAttribute('data-handler-set')) {
            prevBtn.addEventListener('click', () => {
                console.log('[App] Manual Previous clicked');
                this.webrtcClient.manualPrevious();
            });
            prevBtn.setAttribute('data-handler-set', 'true');
        }
        
        // Next button
        const nextBtn = document.getElementById('nextBtn');
        if (nextBtn && !nextBtn.hasAttribute('data-handler-set')) {
            nextBtn.addEventListener('click', () => {
                console.log('[App] Manual Next clicked');
                this.webrtcClient.manualNext();
            });
            nextBtn.setAttribute('data-handler-set', 'true');
        }
        
        // Redo button
        const redoBtn = document.getElementById('redoBtn');
        if (redoBtn && !redoBtn.hasAttribute('data-handler-set')) {
            redoBtn.addEventListener('click', () => {
                console.log('[App] Manual Redo clicked');
                this.webrtcClient.manualRedo();
            });
            redoBtn.setAttribute('data-handler-set', 'true');
        }
        
        // Trigger All button (for testing)
        const triggerAllBtn = document.getElementById('triggerAllBtn');
        if (triggerAllBtn && !triggerAllBtn.hasAttribute('data-handler-set')) {
            triggerAllBtn.addEventListener('click', () => {
                console.log('[App] Trigger All clicked - running through all agents');
                // This could sequentially trigger all agents for testing
                this.runAllAgentsManually();
            });
            triggerAllBtn.setAttribute('data-handler-set', 'true');
        }
    }
    
    /**
     * Run all agents manually in sequence (for testing)
     */
    async runAllAgentsManually() {
        console.log('[App] Running all agents manually');
        const agents = ['initial_classifier', 'detail_extractor', 'damage_detector'];
        
        for (const agent of agents) {
            console.log(`[App] Triggering ${agent}`);
            this.webrtcClient.manualSelectAgent(agent);
            // Wait a bit between agents
            await new Promise(resolve => setTimeout(resolve, 500));
        }
        
        // After all agents, confirm final compilation
        setTimeout(() => {
            console.log('[App] Confirming final compilation');
            this.webrtcClient.manualConfirmFinal();
        }, 2000);
    }
    
    /**
     * Update pause/resume button visibility and state
     */
    updatePauseResumeButtons() {
        const pauseBtn = document.getElementById('pauseBtn');
        const resumeBtn = document.getElementById('resumeBtn');
        
        if (!pauseBtn || !resumeBtn) return;
        
        // Only show pause/resume in automatic mode when monitoring is active
        if (this.mode === 'automatic' && this.monitoringActive) {
            if (this.isPaused) {
                pauseBtn.style.display = 'none';
                resumeBtn.style.display = 'inline-block';
                resumeBtn.disabled = false;
            } else {
                pauseBtn.style.display = 'inline-block';
                pauseBtn.disabled = false;
                resumeBtn.style.display = 'none';
            }
        } else {
            // Hide both buttons in manual mode or when not monitoring
            pauseBtn.style.display = 'none';
            resumeBtn.style.display = 'none';
        }
    }

    /**
     * Show error message
     */
    showError(message) {
        console.error('Error:', message);
        this.streamViewer.showOverlay(`Error: ${message}`);
        this.statusMonitor.showError(message);
    }
}

/**
 * Initialize the main app after initialization completes
 * This is called by the InitializationManager
 */
window.initializeApp = function() {
    console.log('Starting main application...');
    window.appController = new ApplicationController();
};

// Do NOT auto-initialize - let InitializationManager handle it
// The initialization manager will call window.initializeApp() when ready