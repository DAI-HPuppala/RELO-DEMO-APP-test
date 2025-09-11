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
            });
        });
        
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
            this.streamViewer.hideOverlay();
            this.updateManualControls();
            
            // The session is already started by initialization manager
            // Start automatic mode if selected
            if (this.webrtcClient.sessionId && this.mode === 'automatic') {
                console.log('Starting automatic mode with agent timers');
                // Start automatic processing with configured timers
                // Use shared agent configuration for timers
                const agentTimers = window.AGENT_CONFIG ? 
                    window.AGENT_CONFIG.getTimers() : 
                    {
                        initial_classifier: 4.0,
                        detail_extractor: 3.0,
                        damage_detector: 4.0,
                        final_compiler: 2.0
                    };
                console.log('[App] Starting automatic mode with timers:', agentTimers);
                this.webrtcClient.startAutomatic(agentTimers);
                this.statusMonitor.updateSystemStatus('Automatic Mode Active');
            } else if (this.webrtcClient.sessionId && this.mode !== 'automatic') {
                // Send mode change via data channel for manual mode
                if (this.webrtcClient.dataChannel && this.webrtcClient.dataChannel.readyState === 'open') {
                    this.webrtcClient.dataChannel.send(JSON.stringify({
                        type: 'mode_change',
                        mode: this.mode
                    }));
                }
                this.statusMonitor.updateSystemStatus('Manual Mode Active');
            } else {
                this.statusMonitor.updateSystemStatus('Monitoring Active');
            }
            
        } catch (error) {
            console.error('Failed to start monitoring:', error);
            this.showError('Failed to start monitoring');
        }
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
        
        // Send stop signal via data channel
        if (this.webrtcClient.dataChannel && this.webrtcClient.dataChannel.readyState === 'open') {
            this.webrtcClient.dataChannel.send(JSON.stringify({
                type: 'stop_session'
            }));
        }
        
        this.statusMonitor.stopMonitoring();
        this.streamViewer.showOverlay('Monitoring stopped');
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
        this.agentMonitor.onProgressUpdate(data);
        if (data.partial_results) {
            this.resultsDisplay.displayPartialResults(data.agent, data.partial_results);
        }
    }

    /**
     * Handle agent completed
     */
    handleAgentCompleted(data) {
        console.log('[App] Agent completed:', data.agent, data.results);
        this.resultsDisplay.updateAgentProgress(data.agent, 'Completed');
        this.resultsDisplay.displayPartialResults(data.agent, data.results);
        this.agentMonitor.onAgentCompleted(data);
    }

    /**
     * Handle final results
     */
    handleFinalResults(data) {
        console.log('[App] Final results received:', data);
        
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