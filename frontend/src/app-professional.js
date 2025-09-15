/**
 * Professional Application Controller
 * Enhanced version with progressive updates and persistence
 */
class ProfessionalApplicationController {
    constructor() {
        this.webrtcClient = window.webrtcClient;
        
        // Initialize components
        this.streamViewer = new StreamViewer();
        this.resultsDisplay = new ResultsDisplayProfessional();
        this.progressMonitor = new ProgressMonitor();
        
        // Store globally for access
        window.streamViewer = this.streamViewer;
        window.resultsDisplay = this.resultsDisplay;
        window.progressMonitor = this.progressMonitor;
        
        this.sessionId = null;
        this.monitoringActive = false;
        this.mode = 'auto';
        this.isPaused = false;
        this.startTime = null;
        this.processingTimer = null;
        this.currentAgent = 'initial_classifier';
        this.manualModeEnabled = false;
        this.lastSessionId = null;
        this.previousAgentStates = {};
        
        // Track completed agents in current cycle
        this.completedAgentsInCycle = new Set();
        
        // Agent timers and states
        this.agentTimers = {};
        this.agentStates = {
            'initial_classifier': 'inactive',
            'detail_extractor': 'inactive',
            'damage_detector': 'inactive',
            'initial': 'inactive',
            'detail': 'inactive',
            'damage': 'inactive'
        };
        this.agentStartTimes = {};
        
        // Track row index for results table
        this.currentRowIndex = 0;
        this.cycleStartTime = null;
        this.backendProcessingTime = null; // Store backend-reported processing time
        
        // Track session processing times for averaging
        this.sessionProcessingTimes = [];
        this.totalSessions = 0;
        
        // Processing FPS tracking
        this.processingFpsTracker = {
            inferences: [],
            currentFps: 0,
            updateInterval: null
        };
        
        // Manual mode node clicking restriction (controlled by backend)
        this.nodeClickingEnabled = false; // Disabled until all 3 agents complete in current cycle
        
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
        console.log('[Professional App] Initializing enhanced application');
        
        // Ensure sticky header on initialization
        setTimeout(() => this.ensureTableHeaderSticky(), 100);
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Set up WebRTC callbacks
        this.setupWebRTCCallbacks();
        
        // Update initial status
        this.updateConnectionStatus(true);
        this.updateSystemStatus('Ready');
        
        // Initialize node descriptions with default text
        this.initializeNodeDescriptions();
        
        // Check for existing sessions
        this.checkExistingSessions();
    }

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Control buttons
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const pauseResumeBtn = document.getElementById('pauseResumeBtn');
        const prevBtn = document.getElementById('prevBtn');
        const nextBtn = document.getElementById('nextBtn');
        const redoBtn = document.getElementById('redoBtn');
        const modeToggle = document.getElementById('modeToggle');
        
        if (startBtn) {
            startBtn.addEventListener('click', () => this.startMonitoring());
        }
        
        if (stopBtn) {
            stopBtn.addEventListener('click', () => this.stopMonitoring());
        }
        
        if (pauseResumeBtn) {
            pauseResumeBtn.addEventListener('click', () => this.togglePauseResume());
        }
        
        // Manual mode buttons
        if (prevBtn) {
            prevBtn.addEventListener('click', () => this.sendManualPrevious());
        }
        
        if (nextBtn) {
            nextBtn.addEventListener('click', () => this.sendManualNext());
        }
        
        if (redoBtn) {
            redoBtn.addEventListener('click', () => this.sendManualRedo());
        }
        
        // Mode toggle
        if (modeToggle) {
            modeToggle.addEventListener('change', (e) => this.handleModeToggle(e));
        }
        
        // Pipeline node clicks (for manual mode)
        const pipelineNodes = document.querySelectorAll('.pipeline-node');
        pipelineNodes.forEach(node => {
            node.addEventListener('click', (e) => this.handleNodeClick(e));
        })
        
        // Mode selector
        const modeRadios = document.querySelectorAll('input[name="mode"]');
        modeRadios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.mode = e.target.value;
                console.log('[Professional App] Mode changed to:', this.mode);
                this.updatePauseResumeButtons();
            });
        });
        
        // Clear progress button
        const clearProgressBtn = document.getElementById('clearProgressBtn');
        if (clearProgressBtn) {
            clearProgressBtn.addEventListener('click', () => {
                this.progressMonitor.clearLog();
            });
        }
        
        // Export buttons
        const exportCsvBtn = document.getElementById('exportCsvBtn');
        const exportJsonBtn = document.getElementById('exportJsonBtn');
        
        if (exportCsvBtn) {
            exportCsvBtn.addEventListener('click', () => this.exportResultsAsCSV());
        }
        
        if (exportJsonBtn) {
            exportJsonBtn.addEventListener('click', () => this.exportResultsAsJSON());
        }
        
        // Clear logs button
        const clearLogsBtn = document.getElementById('clearLogsBtn');
        if (clearLogsBtn) {
            clearLogsBtn.addEventListener('click', () => this.clearActivityLog());
        }
    }

    /**
     * Set up WebRTC callbacks
     */
    setupWebRTCCallbacks() {
        // Stream events
        this.webrtcClient.on('stream', (stream) => {
            console.log('[Professional App] Received video stream');
            this.streamViewer.setStream(stream);
            this.updateCameraStatus('Connected');
        });
        
        // Data channel messages
        this.webrtcClient.on('message', (data) => {
            this.handleDataChannelMessage(data);
        });
        
        // Manual mode status callback - ensure it's properly handled
        this.webrtcClient.on('manualModeStatus', (data) => {
            this.handleManualModeStatus(data);
        });
        
        // Connection events
        this.webrtcClient.on('connected', () => {
            console.log('[Professional App] WebRTC connected');
            this.updateConnectionStatus(true);
        });
        
        this.webrtcClient.on('disconnected', () => {
            console.log('[Professional App] WebRTC disconnected');
            this.updateConnectionStatus(false);
        });
    }

    /**
     * Handle data channel messages
     */
    handleDataChannelMessage(data) {
        console.log('[Professional App] Data channel message:', data.type);
        
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
            case 'manual_mode_status':
                this.handleManualModeStatus(data);
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
            console.log('[Professional App] Starting classification monitoring');
            
            // Reset components
            this.resultsDisplay.reset();
            this.progressMonitor.reset();
            
            // Initialize results table with placeholders
            this.initializeResultsTable();
            
            // Ensure table header stays sticky
            this.ensureTableHeaderSticky();
            
            // Force complete reset of all pipeline nodes
            this.forceResetAllNodes();
            
            // Clear completed agents tracking
            this.completedAgentsInCycle.clear();
            
            // Reset all agent states to inactive (not empty)
            this.agentStates = {
                'initial_classifier': 'inactive',
                'detail_extractor': 'inactive',
                'damage_detector': 'inactive',
                'initial': 'inactive',
                'detail': 'inactive',
                'damage': 'inactive'
            };
            this.previousAgentStates = {};
            
            // Reset all timers
            ['initial', 'detail', 'damage'].forEach(agent => {
                this.resetAgentTimer(agent);
            });
            
            // Clear all progressive attributes for new session
            this.resetProgressiveAttributes();
            
            // Reset node clicking for manual mode (backend will control this)
            if (this.mode === 'manual') {
                this.nodeClickingEnabled = false;
                console.log('[Professional App] Starting in manual mode - node clicking disabled until all agents complete');
            }
            
            // Reset update counter
            const counterEl = document.getElementById('updateCounter');
            if (counterEl) counterEl.textContent = '0 inferences';
            
            // Track cycle start time
            this.cycleStartTime = Date.now();
            this.backendProcessingTime = null; // Reset backend time for new cycle
            
            // Start processing timer
            this.startTime = Date.now();
            this.startProcessingTimer();
            
            // Start processing FPS monitoring
            this.startProcessingFpsMonitoring();
            
            // Add log entry
            this.addActivityLog(`Classification started in ${this.mode} mode`);
            
            // Update UI state
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const pauseResumeBtn = document.getElementById('pauseResumeBtn');
            const prevBtn = document.getElementById('prevBtn');
            const nextBtn = document.getElementById('nextBtn');
            const redoBtn = document.getElementById('redoBtn');
            const modeToggle = document.getElementById('modeToggle');
            
            if (startBtn) startBtn.disabled = true;
            if (stopBtn) stopBtn.disabled = false;
            if (modeToggle) modeToggle.disabled = true;  // Disable mode toggle during monitoring
            
            this.monitoringActive = true;
            this.isPaused = false;
            
            // Update buttons based on mode
            if (this.mode === 'auto') {
                // Show and enable pause/resume button for auto mode
                if (pauseResumeBtn) {
                    pauseResumeBtn.style.display = 'inline-flex';
                    pauseResumeBtn.disabled = false;
                    this.setPauseResumeButton(false); // Set to Pause state
                }
                // Hide manual mode buttons
                if (prevBtn) prevBtn.style.display = 'none';
                if (nextBtn) nextBtn.style.display = 'none';
                if (redoBtn) redoBtn.style.display = 'none';
            } else if (this.mode === 'manual') {
                // Hide pause/resume for manual mode
                if (pauseResumeBtn) pauseResumeBtn.style.display = 'none';
                
                // Show and enable manual mode buttons
                if (prevBtn) {
                    prevBtn.style.display = 'inline-flex';
                    prevBtn.disabled = false;
                }
                if (nextBtn) {
                    nextBtn.style.display = 'inline-flex';
                    nextBtn.disabled = false;
                }
                if (redoBtn) {
                    redoBtn.style.display = 'inline-flex';
                    redoBtn.disabled = false;
                }
                
                this.updateNodeDescriptions();
            }
            
            this.streamViewer.hideOverlay();
            this.updatePauseResumeButtons();
            
            // Check if we need to create a new session (after stop or if no session exists)
            if (!this.webrtcClient.sessionId) {
                console.log('[Professional App] Creating new session for monitoring');
                await this.webrtcClient.startSession(this.mode === 'manual' ? 'manual' : 'automatic', 'realsense');
                // Wait for session to be created
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
            
            // Start automatic mode if selected
            if (this.webrtcClient.sessionId && this.mode === 'auto') {
                console.log('[Professional App] Starting automatic mode');
                
                // Wait for data channel to be ready before starting
                const dataChannelReady = await this.webrtcClient.waitForDataChannel(3000);
                if (!dataChannelReady) {
                    console.warn('[Professional App] Data channel not ready, but proceeding with automatic mode');
                }
                
                const agentTimers = window.AGENT_CONFIG ? 
                    window.AGENT_CONFIG.getTimers() : 
                    {
                        initial_classifier: 4.0,
                        detail_extractor: 3.0,
                        damage_detector: 4.0,
                        final_compiler: 2.0
                    };
                
                this.webrtcClient.startAutomatic(agentTimers);
                this.updateSystemStatus('Processing - Automatic Mode');
            } else if (this.mode === 'manual') {
                console.log('[Professional App] Starting manual mode');
                
                // Wait for data channel to be ready
                const dataChannelReady = await this.webrtcClient.waitForDataChannel(3000);
                if (!dataChannelReady) {
                    console.warn('[Professional App] Data channel not ready, but proceeding with manual mode');
                }
                
                // Get agent timers (same as auto mode)
                const agentTimers = window.AGENT_CONFIG ? 
                    window.AGENT_CONFIG.getTimers() : 
                    {
                        initial_classifier: 4.0,
                        detail_extractor: 3.0,
                        damage_detector: 4.0,
                        final_compiler: 2.0
                    };
                
                // Start manual mode (pass 'manual' as second parameter)
                this.webrtcClient.startAutomatic(agentTimers, 'manual');
                this.updateSystemStatus('Ready - Manual Mode');
            }
            
            // Use the session ID from webrtcClient (which was just created or already exists)
            this.sessionId = this.webrtcClient.sessionId;
            if (!this.sessionId) {
                console.error('[Professional App] Failed to get session ID from WebRTC client');
                throw new Error('Failed to create session');
            }
            console.log('[Professional App] Using session ID:', this.sessionId);
            this.updateSessionId(this.sessionId);
            
            // Update session info display
            const sessionInfoEl = document.getElementById('sessionInfo');
            if (sessionInfoEl) {
                sessionInfoEl.textContent = this.sessionId.substring(0, 15) + '...';
            }
            
        } catch (error) {
            console.error('[Professional App] Failed to start monitoring:', error);
            this.showError('Failed to start monitoring: ' + error.message);
        }
    }

    /**
     * Stop monitoring
     */
    stopMonitoring() {
        console.log('[Professional App] Stopping monitoring');
        
        this.monitoringActive = false;
        
        // Stop processing timer
        this.stopProcessingTimer();
        
        // Stop all agent timers
        ['initial', 'detail', 'damage'].forEach(agent => {
            this.stopAgentTimer(agent);
        });
        
        // Force immediate visual reset of all pipeline nodes to gray
        this.forceResetAllNodes();
        
        // Reset agent states (also reset the short form aliases)
        ['initial_classifier', 'detail_extractor', 'damage_detector', 'initial', 'detail', 'damage'].forEach(agent => {
            this.agentStates[agent] = 'inactive';
        });
        
        // Reset timers
        ['initial', 'detail', 'damage'].forEach(agent => {
            this.resetAgentTimer(agent);
        });
        
        // Update UI state
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const pauseResumeBtn = document.getElementById('pauseResumeBtn');
        const prevBtn = document.getElementById('prevBtn');
        const nextBtn = document.getElementById('nextBtn');
        const redoBtn = document.getElementById('redoBtn');
        const modeToggle = document.getElementById('modeToggle');
        
        if (startBtn) startBtn.disabled = false;
        if (stopBtn) stopBtn.disabled = true;
        if (pauseResumeBtn) {
            pauseResumeBtn.disabled = true;
            pauseResumeBtn.style.display = 'inline-flex';
            this.setPauseResumeButton(false); // Reset to Pause state
        }
        
        // Disable manual mode buttons
        if (prevBtn) prevBtn.disabled = true;
        if (nextBtn) nextBtn.disabled = true;
        if (redoBtn) redoBtn.disabled = true;
        
        // Enable mode toggle
        if (modeToggle) modeToggle.disabled = false;
        
        // Clear node descriptions
        this.updateNodeDescriptions();
        
        // Send stop signal
        if (this.webrtcClient.dataChannel && this.webrtcClient.dataChannel.readyState === 'open') {
            this.webrtcClient.dataChannel.send(JSON.stringify({
                type: 'stop_session',
                session_id: this.sessionId
            }));
        }
        
        // Clear session ID to prevent sending commands with stale session
        console.log('[Professional App] Clearing session ID after stop');
        this.sessionId = null;
        this.webrtcClient.sessionId = null;
        
        this.streamViewer.showOverlay('Classification stopped');
        this.isPaused = false;
        this.updatePauseResumeButtons();
        
        // Stop processing FPS monitoring
        this.stopProcessingFpsMonitoring();
        
        this.updateSystemStatus('Stopped');
    }

    /**
     * Pause monitoring
     */
    pauseMonitoring() {
        if (this.mode !== 'auto' || !this.monitoringActive || this.isPaused) {
            return;
        }
        
        console.log('[Professional App] Pausing monitoring');
        this.isPaused = true;
        
        // Send pause signal
        if (this.webrtcClient.dataChannel && this.webrtcClient.dataChannel.readyState === 'open') {
            this.webrtcClient.dataChannel.send(JSON.stringify({
                type: 'pause_flow',
                session_id: this.sessionId
            }));
        }
        
        this.updatePauseResumeButtons();
        this.updateSystemStatus('Paused');
    }

    /**
     * Resume monitoring
     */
    resumeMonitoring() {
        if (this.mode !== 'auto' || !this.monitoringActive || !this.isPaused) {
            return;
        }
        
        console.log('[Professional App] Resuming monitoring');
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
        this.updateSystemStatus('Processing - Automatic Mode');
    }

    /**
     * Handle agent started
     */
    handleAgentStarted(data) {
        console.log('[Professional App] Agent started:', data.agent, 'Session:', data.session_id);
        this.progressMonitor.onAgentStarted(data);
        this.updateSystemStatus(`Processing: ${data.agent}`);
        
        // Check for session ID change which indicates a new cycle
        const sessionChanged = data.session_id && this.lastSessionId && (data.session_id !== this.lastSessionId);
        const isInitial = data.agent === 'initial_classifier' || data.agent === 'initial';
        
        // Detect new cycle: either session ID changed OR initial agent starting with completed previous agents
        const isNewCycle = sessionChanged || (isInitial && 
            (this.previousAgentStates['damage'] === 'completed' || 
             this.previousAgentStates['damage_detector'] === 'completed'));
        
        // Add more detailed logging to debug cycle detection
        console.log('[Professional App] Cycle detection:', {
            isInitial,
            sessionChanged,
            isNewCycle,
            previousStates: this.previousAgentStates,
            completedInCycle: Array.from(this.completedAgentsInCycle)
        });
        
        if (isNewCycle) {
            console.log('[Professional App] NEW CYCLE DETECTED - Resetting for new cycle');
            console.log('  - Session changed:', sessionChanged);
            console.log('  - Previous session:', this.lastSessionId);
            console.log('  - New session:', data.session_id);
            
            // Clear completedAgentsInCycle FIRST for new cycle
            this.completedAgentsInCycle.clear();
            
            // NOW reset detail and damage to gray (idle)
            this.setPipelineNodeState('detail', 'idle');
            this.setPipelineNodeState('damage', 'idle');
            
            // Reset all agent states for new cycle
            this.agentStates = {
                'initial_classifier': 'inactive',
                'detail_extractor': 'inactive',
                'damage_detector': 'inactive',
                'initial': 'inactive',
                'detail': 'inactive',
                'damage': 'inactive'
            };
            this.previousAgentStates = {};
            
            // Clear progressive attributes for new cycle
            this.resetProgressiveAttributes();
            
            // Initialize new results table row
            this.initializeResultsTable();
            
            // Reset update counter
            const counterEl = document.getElementById('updateCounter');
            if (counterEl) counterEl.textContent = '0 inferences';
            
            // Reset all agent timers
            ['initial', 'detail', 'damage'].forEach(agent => {
                this.resetAgentTimer(agent);
            });
        }
        
        // Update session ID tracking
        if (data.session_id) {
            this.lastSessionId = data.session_id;
        }
        
        // Update pipeline flow visualization for the current agent
        // This will set the agent to processing (orange)
        this.updatePipelineFlow(data.agent, 'active')
        
        // Don't reset attributes here - they should persist from previous agents
        // Only reset at the start of a new cycle
        
        // Update inference count and progress for the agent
        this.updateAgentProgress(data.agent, 0);
        
        // Add log entry
        const agentNames = {
            'initial_classifier': 'Initial Classifier',
            'detail_extractor': 'Detail Extractor',
            'damage_detector': 'Damage Detector',
            'final_compiler': 'Final Compiler'
        };
        this.addActivityLog(`${agentNames[data.agent] || data.agent} started`);
    }

    /**
     * Handle progress update
     */
    handleProgressUpdate(data) {
        console.log('[Professional App] Progress update:', data.agent, data.progress);
        console.log('[FULL DATA]:', data);
        
        // Log progressive results
        if (data.partial_results) {
            console.log(`[PROGRESSIVE] Agent: ${data.agent}, Inference #${data.inference_num}`);
            console.log('[PROGRESSIVE] Results:', data.partial_results);
            
            // Update progressive attributes display
            this.updateProgressiveAttributes(data.partial_results);
            
            // Update results table progressively
            this.updateResultsTablePartial(data.agent, data.partial_results);
        }
        
        this.progressMonitor.onProgressUpdate(data);
        
        // Update agent progress in pipeline node
        this.updateAgentProgress(data.agent, data.inference_num || 1);
        
        // Update total inference counter
        const counterEl = document.getElementById('updateCounter');
        if (counterEl) {
            const current = parseInt(counterEl.textContent) || 0;
            counterEl.textContent = `${current + 1} inferences`;
        }
        
        // Track processing FPS
        this.trackProcessingFps();
    }

    /**
     * Handle inference update
     */
    handleInferenceUpdate(data) {
        console.log('[Professional App] Inference update:', data);
        console.log('[PROGRESSIVE ATTRIBUTES]:', data.attributes);
        
        // Update progressive attributes if present
        if (data.attributes) {
            this.updateProgressiveAttributes(data.attributes);
            this.updateResultsTablePartial(data.agent, data.attributes);
        }
        
        this.progressMonitor.onProgressUpdate({
            agent: data.agent,
            inference_num: data.inference_num,
            partial_results: data.attributes || data.results
        });
        
        // Update agent progress in pipeline node
        this.updateAgentProgress(data.agent, data.inference_num || 1);
        
        // Update total inference counter
        const counterEl = document.getElementById('updateCounter');
        if (counterEl) {
            const current = parseInt(counterEl.textContent) || 0;
            counterEl.textContent = `${current + 1} inferences`;
        }
        
        // Track processing FPS
        this.trackProcessingFps();
    }

    /**
     * Handle manual mode status update
     */
    handleManualModeStatus(data) {
        console.log('[Professional App] Manual mode status:', data);
        console.log('[Professional App] handleManualModeStatus called with node_clicking_enabled:', data.node_clicking_enabled);
        
        // Check if all three main agents have been completed in the CURRENT cycle
        // Node clicking should ONLY be enabled when all 3 agents are completed in the current cycle
        let shouldEnableNodeClicking = false;
        
        if (data.completed_agents && Array.isArray(data.completed_agents)) {
            // Check if the current cycle has all 3 required agents completed
            const requiredAgents = ['initial_classifier', 'detail_extractor', 'damage_detector'];
            const currentCycleCompleted = requiredAgents.every(agent => data.completed_agents.includes(agent));
            
            // Enable node clicking ONLY if all 3 agents are completed in current cycle
            shouldEnableNodeClicking = currentCycleCompleted && data.completed_agents.length === 3;
            
            console.log(`[Professional App] Current cycle completed agents: ${data.completed_agents}, all 3 completed: ${shouldEnableNodeClicking}`);
        }
        
        // Update node clicking state
        const wasEnabled = this.nodeClickingEnabled;
        this.nodeClickingEnabled = shouldEnableNodeClicking;
        
        // If state changed from disabled to enabled
        if (!wasEnabled && this.nodeClickingEnabled) {
            console.log('[Professional App] All 3 agents completed in current cycle - node clicking now enabled');
            this.addActivityLog('All 3 agents completed - node clicking enabled for this cycle');
            
            // Update visual feedback for nodes
            const pipelineNodes = document.querySelectorAll('.pipeline-node');
            pipelineNodes.forEach(node => {
                node.classList.add('clicking-enabled');
            });
        }
        // If state changed from enabled to disabled (shouldn't happen unless new session)
        else if (wasEnabled && !this.nodeClickingEnabled) {
            console.log('[Professional App] Node clicking disabled (new session?)');
            
            // Remove visual feedback for nodes
            const pipelineNodes = document.querySelectorAll('.pipeline-node');
            pipelineNodes.forEach(node => {
                node.classList.remove('clicking-enabled');
            });
        }
        
        // Log completed and pending agents for debugging
        if (data.completed_agents) {
            console.log('[Professional App] Completed agents (current cycle):', data.completed_agents);
        }
        if (data.pending_agents) {
            console.log('[Professional App] Pending agents (current cycle):', data.pending_agents);
        }
    }

    /**
     * Handle agent completed
     */
    handleAgentCompleted(data) {
        console.log('[Professional App] Agent completed:', data.agent);
        console.log('[FINALIZED] Agent results:', data.results);
        
        this.progressMonitor.onAgentCompleted(data);
        
        // Update pipeline flow - this will set the node to green and track completion
        this.updatePipelineFlow(data.agent, 'complete');
        
        // Track completed state for cycle detection
        this.previousAgentStates[data.agent] = 'completed';
        
        // Log the current state of completedAgentsInCycle for debugging
        console.log('[Professional App] Completed agents in cycle:', Array.from(this.completedAgentsInCycle));
        
        // Update progressive attributes with final results
        if (data.results) {
            this.updateProgressiveAttributes(data.results);
            this.updateResultsTablePartial(data.agent, data.results);
        }
        
        // Add log entry
        const agentNames = {
            'initial_classifier': 'Initial Classifier',
            'detail_extractor': 'Detail Extractor',
            'damage_detector': 'Damage Detector',
            'final_compiler': 'Final Compiler'
        };
        this.addActivityLog(`${agentNames[data.agent] || data.agent} completed`);
        
        // IMPORTANT: Don't reset any nodes here - they should stay green until final_compiler triggers
        // The nodes will be reset in handleFinalResults when the cycle truly completes
    }

    /**
     * Handle final results
     */
    handleFinalResults(data) {
        console.log('[Professional App] Final classification results:', data);
        
        // Clear completedAgentsInCycle FIRST before resetting nodes
        // This ensures setPipelineNodeState will actually reset them to idle
        this.completedAgentsInCycle.clear();
        
        // When cycle completes (final compiler agent triggers):
        // NOW set detail and damage nodes to gray (idle)
        this.setPipelineNodeState('detail', 'idle');
        this.setPipelineNodeState('damage', 'idle');
        
        // Set initial node to orange (processing) when final compiler triggers
        // This indicates ready for next cycle
        this.setPipelineNodeState('initial', 'processing');
        
        // Mark all agents as completed for cycle detection
        this.previousAgentStates['initial'] = 'completed';
        this.previousAgentStates['initial_classifier'] = 'completed';
        this.previousAgentStates['detail'] = 'completed';
        this.previousAgentStates['detail_extractor'] = 'completed';
        this.previousAgentStates['damage'] = 'completed';
        this.previousAgentStates['damage_detector'] = 'completed';
        
        console.log('[Professional App] Cycle complete - nodes reset for next cycle');
        
        // Stop processing timer
        const processingTime = this.stopProcessingTimer();
        
        // Store backend processing time if provided
        if (data.processing_time !== undefined) {
            this.backendProcessingTime = data.processing_time;
            console.log('[Professional App] Backend reported processing time:', this.backendProcessingTime + 's');
        }
        
        // Add processing time to results (prefer backend time if available)
        if (data.results) {
            data.results.processing_time = this.backendProcessingTime || processingTime;
            data.results.session_id = this.sessionId;
            data.results.agents_completed = data.agents_completed;
        }
        
        // Display final results
        this.resultsDisplay.displayFinalResults(data.results);
        
        // Update the results table with final data
        this.updateResultsTable(data.results);
        
        this.updateSystemStatus('Classification Complete');
        
        // Don't reset nodes here - let the next cycle's initial agent handle it
        // The nodes should stay in their completed (green) state until the next cycle starts
        
        // Enable export buttons
        const exportCsvBtn = document.getElementById('exportCsvBtn');
        const exportJsonBtn = document.getElementById('exportJsonBtn');
        if (exportCsvBtn) exportCsvBtn.disabled = false;
        if (exportJsonBtn) exportJsonBtn.disabled = false;
        
        // Track session time and update average
        const duration = this.cycleStartTime ? 
            ((Date.now() - this.cycleStartTime) / 1000) : 0;
        
        if (duration > 0) {
            this.sessionProcessingTimes.push(duration);
            this.totalSessions++;
            this.updateAverageProcessingTime();
            
            // Update total sessions display
            const totalSessionsEl = document.getElementById('totalSessions');
            if (totalSessionsEl) {
                totalSessionsEl.textContent = this.totalSessions;
            }
        }
        
        // Add completion log
        this.addActivityLog(`Classification completed in ${duration.toFixed(1)}s`);
    }

    /**
     * Handle status update
     */
    handleStatusUpdate(data) {
        if (data.session_id) {
            // Check if session changed
            if (this.sessionId && data.session_id !== this.sessionId) {
                console.log('[Professional App] Session ID changed in status update - potential new cycle');
                this.lastSessionId = this.sessionId;
            }
            this.sessionId = data.session_id;
            this.updateSessionId(data.session_id);
        }
        
        if (data.status) {
            this.updateSystemStatus(data.status);
        }
    }

    /**
     * Handle flow paused
     */
    handleFlowPaused(data) {
        console.log('[Professional App] Flow paused at:', data.paused_agent);
        this.isPaused = true;
        this.updatePauseResumeButtons();
        this.updateSystemStatus(`Paused at ${data.paused_agent}`);
    }

    /**
     * Handle flow resumed
     */
    handleFlowResumed(data) {
        console.log('[Professional App] Flow resumed at:', data.resuming_agent);
        this.isPaused = false;
        this.updatePauseResumeButtons();
        this.updateSystemStatus('Processing - Automatic Mode');
    }

    /**
     * Update pause/resume button visibility
     */
    updatePauseResumeButtons() {
        if (this.mode === 'auto' && this.monitoringActive) {
            this.setPauseResumeButton(this.isPaused);
        }
    }
    
    /**
     * Toggle between pause and resume
     */
    togglePauseResume() {
        if (this.isPaused) {
            this.resumeMonitoring();
        } else {
            this.pauseMonitoring();
        }
    }
    
    /**
     * Set pause/resume button state
     */
    setPauseResumeButton(isPaused) {
        const btn = document.getElementById('pauseResumeBtn');
        if (!btn) return;
        
        const icon = btn.querySelector('.btn-icon');
        const text = btn.querySelector('.btn-text');
        
        if (isPaused) {
            // Change to Resume button
            btn.className = 'btn btn-info auto-only';
            btn.setAttribute('data-tooltip', 'Resume agent processing');
            if (icon) icon.textContent = '▶';
            if (text) text.textContent = 'Resume';
        } else {
            // Change to Pause button
            btn.className = 'btn btn-warning auto-only';
            btn.setAttribute('data-tooltip', 'Pause agent processing');
            if (icon) icon.textContent = '⏸';
            if (text) text.textContent = 'Pause';
        }
    }

    /**
     * Update connection status
     */
    updateConnectionStatus(connected) {
        const indicator = document.getElementById('connectionStatus');
        if (indicator) {
            if (connected) {
                indicator.classList.add('connected');
                indicator.querySelector('.status-text').textContent = 'Connected';
            } else {
                indicator.classList.remove('connected');
                indicator.querySelector('.status-text').textContent = 'Disconnected';
            }
        }
    }

    /**
     * Update camera status
     */
    updateCameraStatus(status) {
        const element = document.getElementById('cameraStatus');
        if (element) {
            element.querySelector('span:last-child').textContent = status;
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
     * Check for existing sessions
     */
    checkExistingSessions() {
        const sessions = this.resultsDisplay.getAllResults();
        console.log(`[Professional App] Found ${sessions.length} existing sessions`);
    }

    /**
     * Show error message
     */
    showError(message) {
        console.error('[Professional App] Error:', message);
        this.updateSystemStatus('Error: ' + message);
    }
    
    /**
     * Update pipeline flow visualization
     */
    updatePipelineFlow(agent, status) {
        // Skip final_compiler as it's not displayed in the UI
        if (agent === 'final_compiler' || agent === 'final') {
            return;
        }
        
        console.log(`[Pipeline Update] Agent: ${agent}, Status: ${status}`);
        
        // Map agent names to node IDs
        const agentToNodeMap = {
            'initial_classifier': 'initial',
            'detail_extractor': 'detail',
            'damage_detector': 'damage',
            'initial': 'initial',
            'detail': 'detail',
            'damage': 'damage'
        };
        
        const nodeId = agentToNodeMap[agent];
        if (!nodeId) return;
        
        // Clean, simple logic:
        if (status === 'active') {
            // Orange (processing)
            this.setPipelineNodeState(nodeId, 'processing');
            this.startAgentTimer(agent);
        } else if (status === 'complete') {
            // Green (completed) + track in completedAgentsInCycle
            this.setPipelineNodeState(nodeId, 'completed');
            this.completedAgentsInCycle.add(nodeId);
            this.stopAgentTimer(agent);
        } else if (status === 'inactive') {
            // Check if completed (green) or not (gray)
            if (this.completedAgentsInCycle.has(nodeId)) {
                this.setPipelineNodeState(nodeId, 'completed');
            } else {
                this.setPipelineNodeState(nodeId, 'idle');
            }
            this.stopAgentTimer(agent);
        }
    }
    
    /**
     * Start timer for an agent
     */
    startAgentTimer(agent) {
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
        const maxTimes = {
            'initial': 4.0,
            'detail': 3.0,
            'damage': 4.0,
            'final': 2.0
        };
        
        // Clear existing timer
        if (this.agentTimers[agentKey]) {
            clearInterval(this.agentTimers[agentKey]);
        }
        
        // Reset start time
        this.agentStartTimes[agentKey] = Date.now();
        
        // Start new timer
        this.agentTimers[agentKey] = setInterval(() => {
            const elapsed = (Date.now() - this.agentStartTimes[agentKey]) / 1000;
            const timerEl = document.getElementById(`${agentKey}-timer`);
            if (timerEl) {
                const maxTime = maxTimes[agentKey];
                timerEl.textContent = `${elapsed.toFixed(1)}s / ${maxTime}s`;
            }
        }, 100);
    }
    
    /**
     * Stop timer for an agent
     */
    stopAgentTimer(agent) {
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
        
        if (this.agentTimers[agentKey]) {
            clearInterval(this.agentTimers[agentKey]);
            this.agentTimers[agentKey] = null;
        }
    }
    
    /**
     * Reset timer display for an agent
     */
    resetAgentTimer(agent) {
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
        const maxTimes = {
            'initial': 4.0,
            'detail': 3.0,
            'damage': 4.0,
            'final': 2.0
        };
        
        // Reset timer display
        const timerEl = document.getElementById(`${agentKey}-timer`);
        if (timerEl) {
            const maxTime = maxTimes[agentKey];
            timerEl.textContent = `0.0s / ${maxTime}s`;
        }
        
        // Reset progress bar
        const progressEl = document.getElementById(`${agentKey}-progress`);
        if (progressEl) {
            progressEl.style.width = '0%';
        }
        
        // Reset inference count
        const countEl = document.getElementById(`${agentKey}-count`);
        if (countEl) {
            countEl.textContent = '0 inferences';
        }
    }
    
    /**
     * Reset progressive attributes display
     */
    resetProgressiveAttributes() {
        const attributes = ['color', 'type', 'pattern', 'neckline', 'closure', 'brand', 'size', 'sleeve', 'damage', 'damage-type'];
        attributes.forEach(attr => {
            const element = document.getElementById(`prog-${attr}`);
            if (element) {
                element.textContent = '-';
                element.classList.remove('updated');
            }
        });
    }
    
    /**
     * Update progressive attributes display
     */
    updateProgressiveAttributes(results) {
        if (!results) return;
        
        // Update each attribute if present (using normalized keys)
        const attributeMap = {
            'color': results.color,
            'type': results.item_type,
            'pattern': results.pattern,
            'neckline': results.neckline,
            'closure': results.closure_type,
            'brand': results.brand,
            'size': results.size,
            'sleeve': results.sleeve_type,
            'damage': results.is_damaged !== undefined ? (results.is_damaged ? 'Yes' : 'No') : null,
            'damage-type': results.damage_type
        };
        
        Object.entries(attributeMap).forEach(([key, value]) => {
            if (value !== null && value !== undefined && value !== '') {
                const element = document.getElementById(`prog-${key}`);
                if (element) {
                    element.textContent = value;
                    element.classList.add('updated');
                    // Add visual feedback
                    element.style.color = '#28a745';
                    setTimeout(() => {
                        element.style.color = '';
                    }, 1000);
                }
            }
        });
    }
    
    /**
     * Ensure table header stays sticky
     */
    ensureTableHeaderSticky() {
        const container = document.querySelector('.results-table-container-scrollable');
        const thead = container?.querySelector('.classification-table thead');
        
        if (container && thead) {
            // Ensure the container allows sticky positioning
            if (container.style.overflow !== 'auto') {
                container.style.overflow = 'auto';
            }
            
            // Ensure thead has sticky positioning
            thead.style.position = 'sticky';
            thead.style.top = '0';
            thead.style.zIndex = '10';
        }
    }
    
    /**
     * Initialize results table with placeholders
     */
    initializeResultsTable() {
        const tbody = document.getElementById('resultsTableBody');
        if (!tbody) return;
        
        // Don't clear existing rows, just add new one
        // Remove empty row if it exists
        const emptyRow = tbody.querySelector('.empty-row');
        if (emptyRow) {
            tbody.innerHTML = '';
        }
        
        // Increment row index
        this.currentRowIndex++;
        
        // Create a row with loading placeholders
        const row = document.createElement('tr');
        row.id = 'current-classification-row';
        row.innerHTML = `
            <td data-value="${this.currentRowIndex}">${this.currentRowIndex}</td>
            <td data-value="N/A">-</td>
            <td data-value="N/A">-</td>
            <td data-value="N/A">-</td>
            <td data-value="N/A">-</td>
            <td data-value="N/A">-</td>
            <td data-value="N/A">-</td>
            <td data-value="N/A">-</td>
            <td data-value="N/A">-</td>
            <td data-value="N/A">-</td>
            <td data-value="N/A">-</td>
            <td data-value="Processing">Processing...</td>
        `;
        
        tbody.appendChild(row);
        
        // Track cycle start time and reset backend time
        this.cycleStartTime = Date.now();
        this.backendProcessingTime = null;
    }
    
    /**
     * Update results table partially as data comes in
     */
    updateResultsTablePartial(agent, attributes) {
        const row = document.getElementById('current-classification-row');
        if (!row || !attributes) return;
        
        // Get cells (skip index column at position 0)
        const cells = row.getElementsByTagName('td');
        
        // Helper function to format values professionally
        const formatValue = (attr) => {
            if (attr === null || attr === undefined || attr === '') return 'N/A';
            // Capitalize first letter and clean up underscores
            return String(attr).replace(/_/g, ' ')
                .replace(/\b\w/g, l => l.toUpperCase());
        };
        
        // Helper to set cell content with data attribute
        const setCell = (index, value) => {
            const formattedValue = formatValue(value);
            cells[index].textContent = formattedValue;
            cells[index].setAttribute('data-value', formattedValue);
        };
        
        // Map agent to the attributes they provide (using normalized keys)
        if (agent === 'initial_classifier' || agent === 'initial') {
            // Initial agent handles: type, color, pattern, neckline, sleeve, closure
            if (attributes.item_type !== undefined) setCell(1, attributes.item_type);
            if (attributes.color !== undefined) setCell(2, attributes.color);
            if (attributes.pattern !== undefined) setCell(3, attributes.pattern);
            if (attributes.neckline !== undefined) setCell(6, attributes.neckline);
            if (attributes.sleeve_type !== undefined) setCell(7, attributes.sleeve_type);
            if (attributes.closure_type !== undefined) setCell(8, attributes.closure_type);
        } else if (agent === 'detail_extractor' || agent === 'detail') {
            // Detail agent handles: brand and size only
            if (attributes.brand !== undefined) setCell(4, attributes.brand);
            if (attributes.size !== undefined) setCell(5, attributes.size);
        } else if (agent === 'damage_detector' || agent === 'damage') {
            // Damage agent handles: is_damaged and damage_type
            if (attributes.is_damaged !== undefined) {
                const damageValue = attributes.is_damaged ? 'Yes' : 'No';
                cells[9].textContent = damageValue;
                cells[9].setAttribute('data-value', damageValue);
            }
            if (attributes.damage_type !== undefined) setCell(10, attributes.damage_type);
        }
    }
    
    /**
     * Update final results table
     */
    updateResultsTable(results) {
        const tbody = document.getElementById('resultsTableBody');
        if (!tbody) return;
        
        // Use backend processing time if available, otherwise calculate from frontend
        const duration = this.backendProcessingTime !== null ? 
            this.backendProcessingTime.toFixed(1) : 
            (this.cycleStartTime ? 
                ((Date.now() - this.cycleStartTime) / 1000).toFixed(1) : '0.0');
        
        const timestamp = new Date().toLocaleTimeString('en-US', { 
            hour12: true,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
        const timestampWithDuration = `${timestamp} (${duration}s)`;
        
        // Update the current row's timestamp if it exists
        const currentRow = document.getElementById('current-classification-row');
        if (currentRow) {
            const cells = currentRow.getElementsByTagName('td');
            if (cells.length > 11) {
                cells[11].textContent = timestampWithDuration;
            }
            
            // Also update any missing cells with final data from all agents
            // Format function
            const formatValue = (attr) => {
                if (attr === null || attr === undefined || attr === '') return '-';
                return String(attr).replace(/_/g, ' ')
                    .replace(/\b\w/g, l => l.toUpperCase());
            };
            
            // Handle initial classifier attributes
            if (results.item_type || results.type) {
                cells[1].textContent = formatValue(results.item_type || results.type);
            }
            if (results.color) cells[2].textContent = formatValue(results.color);
            if (results.pattern) cells[3].textContent = formatValue(results.pattern);
            
            // Handle detail extractor attributes
            if (results.brand) cells[4].textContent = formatValue(results.brand);
            if (results.size) cells[5].textContent = formatValue(results.size);
            if (results.neckline) cells[6].textContent = formatValue(results.neckline);
            if (results.sleeve_type || results.sleeve_length || results.sleeve) {
                cells[7].textContent = formatValue(results.sleeve_type || results.sleeve_length || results.sleeve);
            }
            if (results.closure_type || results.closure) {
                cells[8].textContent = formatValue(results.closure_type || results.closure);
            }
            
            // Handle damage detector attributes
            if (results.has_damage !== undefined) {
                cells[9].textContent = results.has_damage ? 'Yes' : 'No';
            }
            if (results.damage_type) cells[10].textContent = formatValue(results.damage_type);
            
            // Remove the ID so the next classification gets a new row
            currentRow.id = '';
            return;
        }
        
        // This shouldn't normally happen, but create a new row if needed
        this.currentRowIndex++;
        const row = document.createElement('tr');
        
        // Helper function to format values professionally
        const formatValue = (attr) => {
            if (attr === null || attr === undefined || attr === '') return '-';
            return String(attr).replace(/_/g, ' ')
                .replace(/\b\w/g, l => l.toUpperCase());
        };
        
        row.innerHTML = `
            <td>${this.currentRowIndex}</td>
            <td>${formatValue(results.item_type || results.clothing_type || results.type)}</td>
            <td>${formatValue(results.color || results.primary_color)}</td>
            <td>${formatValue(results.pattern)}</td>
            <td>${formatValue(results.brand)}</td>
            <td>${formatValue(results.size)}</td>
            <td>${formatValue(results.neckline)}</td>
            <td>${formatValue(results.sleeve_type || results.sleeve_length || results.sleeve)}</td>
            <td>${formatValue(results.closure_type || results.closure)}</td>
            <td>${results.has_damage === true ? 'Yes' : results.has_damage === false ? 'No' : '-'}</td>
            <td>${formatValue(results.damage_type)}</td>
            <td>${timestampWithDuration}</td>
        `;
        
        tbody.appendChild(row);
    }
    
    /**
     * Update average processing time display
     */
    updateAverageProcessingTime() {
        if (this.sessionProcessingTimes.length === 0) return;
        
        const average = this.sessionProcessingTimes.reduce((a, b) => a + b, 0) / this.sessionProcessingTimes.length;
        const avgElement = document.getElementById('processingTime');
        if (avgElement) {
            avgElement.textContent = `${average.toFixed(1)}s`;
        }
    }
    
    /**
     * Add entry to activity log
     */
    addActivityLog(message) {
        const logContainer = document.getElementById('activityLog');
        if (!logContainer) return;
        
        const timestamp = new Date().toLocaleTimeString('en-US', {
            hour12: false,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
        
        const logEntry = document.createElement('div');
        logEntry.className = 'log-entry';
        logEntry.innerHTML = `
            <span class="log-timestamp">${timestamp}</span>
            <span class="log-message">${message}</span>
        `;
        
        logContainer.appendChild(logEntry);
        
        // Auto-scroll to bottom
        logContainer.scrollTop = logContainer.scrollHeight;
        
        // Limit to 50 entries
        while (logContainer.children.length > 50) {
            logContainer.removeChild(logContainer.firstChild);
        }
    }
    
    /**
     * Clear activity log
     */
    clearActivityLog() {
        const logContainer = document.getElementById('activityLog');
        if (logContainer) {
            logContainer.innerHTML = `
                <div class="log-entry">
                    <span class="log-timestamp">-</span>
                    <span class="log-message">Log cleared</span>
                </div>
            `;
        }
    }
    
    /**
     * Update agent progress in pipeline node
     */
    updateAgentProgress(agent, inferenceNum) {
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
        
        // Update inference count
        const countEl = document.getElementById(`${agentKey}-count`);
        if (countEl) {
            countEl.textContent = `${inferenceNum} inferences`;
        }
        
        // Update progress bar (assuming ~5 inferences per agent)
        const progressEl = document.getElementById(`${agentKey}-progress`);
        if (progressEl && inferenceNum) {
            const progress = Math.min((inferenceNum / 5) * 100, 100);
            progressEl.style.width = `${progress}%`;
        }
    }
    
    /**
     * Reset for new classification cycle
     */
    resetForNewCycle() {
        console.log('[Professional App] Resetting for new classification cycle');
        
        // This function is now mostly handled by handleAgentStarted when initial_classifier starts
        // We keep it for manual resets and stop operations
        
        // Reset all nodes to gray
        this.forceResetAllNodes();
        
        // Reset node clicking for new cycle in manual mode
        if (this.manualModeEnabled) {
            this.nodeClickingEnabled = false;
            console.log('[Professional App] New cycle started - node clicking disabled until all agents complete');
            
            // Remove visual feedback for enabled clicking
            const pipelineNodes = document.querySelectorAll('.pipeline-node');
            pipelineNodes.forEach(node => {
                node.classList.remove('clicking-enabled');
            });
        }
        
        // Reset agent states and timers (including short form aliases)
        ['initial_classifier', 'detail_extractor', 'damage_detector', 'initial', 'detail', 'damage'].forEach(agent => {
            this.agentStates[agent] = 'inactive';
        });
        
        // Reset timers
        ['initial', 'detail', 'damage'].forEach(agent => {
            this.resetAgentTimer(agent);
        });
        
        // Clear descriptions
        this.updateNodeDescriptions();
        
        // Clear progressive attributes
        this.resetProgressiveAttributes();
        
        // Initialize new results table row
        this.initializeResultsTable();
        
        // Reset update counter
        const counterEl = document.getElementById('updateCounter');
        if (counterEl) counterEl.textContent = '0 inferences';
        
        // Automatically start first agent if in automatic mode
        if (this.mode === 'auto' && this.monitoringActive) {
            // The backend will automatically start the next cycle
            console.log('[Professional App] Ready for next automatic cycle');
        }
    }
    
    /**
     * Export results as CSV
     */
    exportResultsAsCSV() {
        const tbody = document.getElementById('resultsTableBody');
        if (!tbody || tbody.querySelector('.empty-row')) {
            alert('No results to export');
            return;
        }
        
        // Create CSV content with exact table headers
        const headers = ['#', 'Type', 'Color', 'Pattern', 'Brand', 'Size', 'Neckline', 'Sleeve', 'Closure', 'Damage', 'Damage Type', 'Timestamp'];
        const rows = [];
        
        // Add headers
        rows.push(headers.join(','));
        
        // Add data rows - export exactly what's visible in the table
        const dataRows = tbody.querySelectorAll('tr:not(.empty-row)');
        dataRows.forEach(row => {
            const cells = row.querySelectorAll('td');
            const rowData = Array.from(cells).map(cell => {
                // Get the exact text content from the table
                let content = cell.textContent.trim();
                // Escape commas and quotes in cell content
                if (content.includes(',') || content.includes('"')) {
                    content = `"${content.replace(/"/g, '""')}"`;
                }
                return content;
            });
            rows.push(rowData.join(','));
        });
        
        // Create download with today's date
        const csvContent = rows.join('\n');
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `classification_results_${new Date().toISOString().split('T')[0]}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    /**
     * Set pipeline node visual state
     */
    setPipelineNodeState(nodeId, state) {
        const node = document.getElementById(`node-${nodeId}`);
        if (!node) {
            console.warn(`[Node State] Node element not found: node-${nodeId}`);
            return;
        }
        
        // Don't reset completed nodes to idle unless explicitly clearing the cycle
        const currentlyCompleted = node.classList.contains('completed');
        if (currentlyCompleted && state === 'idle' && this.completedAgentsInCycle.has(nodeId)) {
            console.log(`[Node State] Preserving completed state for ${nodeId} (not resetting to idle)`);
            return;
        }
        
        // Remove all state classes
        node.classList.remove('idle', 'processing', 'completed');
        
        // Add new state class
        node.classList.add(state);
        
        console.log(`[Node State] ${nodeId} -> ${state}`);
    }
    
    /**
     * Force reset all pipeline nodes to idle state
     */
    forceResetAllNodes() {
        // Clear tracking FIRST to allow nodes to be reset
        this.completedAgentsInCycle.clear();
        
        // Now reset all nodes to idle
        ['initial', 'detail', 'damage'].forEach(nodeId => {
            this.setPipelineNodeState(nodeId, 'idle');
        });
    }
    
    /**
     * Export results as JSON
     */
    exportResultsAsJSON() {
        const tbody = document.getElementById('resultsTableBody');
        if (!tbody || tbody.querySelector('.empty-row')) {
            alert('No results to export');
            return;
        }
        
        const results = [];
        const dataRows = tbody.querySelectorAll('tr:not(.empty-row)');
        
        // Export exactly what's visible in the table
        dataRows.forEach(row => {
            const cells = row.querySelectorAll('td');
            if (cells.length >= 12) {
                results.push({
                    index: cells[0].textContent.trim(),
                    type: cells[1].textContent.trim(),
                    color: cells[2].textContent.trim(),
                    pattern: cells[3].textContent.trim(),
                    brand: cells[4].textContent.trim(),
                    size: cells[5].textContent.trim(),
                    neckline: cells[6].textContent.trim(),
                    sleeve: cells[7].textContent.trim(),
                    closure: cells[8].textContent.trim(),
                    damage: cells[9].textContent.trim(),
                    damage_type: cells[10].textContent.trim(),
                    timestamp: cells[11].textContent.trim()
                });
            }
        });
        
        const exportData = {
            export_date: new Date().toISOString(),
            total_items: results.length,
            classifications: results
        };
        
        // Create download
        const jsonContent = JSON.stringify(exportData, null, 2);
        const blob = new Blob([jsonContent], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `classification_results_${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    /**
     * Track processing FPS based on inference updates
     */
    trackProcessingFps() {
        const now = Date.now();
        this.processingFpsTracker.inferences.push(now);
        
        // Keep only last 10 seconds of data
        const cutoff = now - 10000;
        this.processingFpsTracker.inferences = this.processingFpsTracker.inferences.filter(time => time > cutoff);
        
        // Calculate FPS based on inferences in last 10 seconds
        if (this.processingFpsTracker.inferences.length > 1) {
            const timeSpan = (now - this.processingFpsTracker.inferences[0]) / 1000;
            this.processingFpsTracker.currentFps = Math.round(this.processingFpsTracker.inferences.length / timeSpan);
        } else {
            this.processingFpsTracker.currentFps = 0;
        }
        
        this.updateProcessingFpsDisplay();
    }
    
    /**
     * Update processing FPS display
     */
    updateProcessingFpsDisplay() {
        const processingFpsElement = document.getElementById('processingFPS');
        if (processingFpsElement) {
            const fps = this.processingFpsTracker.currentFps;
            processingFpsElement.textContent = fps;
            
            // Apply color coding based on processing performance
            processingFpsElement.className = 'stat-value';
            if (fps >= 5) {
                processingFpsElement.classList.add('fps-good');
            } else if (fps >= 2) {
                processingFpsElement.classList.add('fps-medium');
            } else if (fps > 0) {
                processingFpsElement.classList.add('fps-poor');
            }
        }
    }
    
    /**
     * Start processing FPS monitoring
     */
    startProcessingFpsMonitoring() {
        // Reset tracker
        this.processingFpsTracker.inferences = [];
        this.processingFpsTracker.currentFps = 0;
        
        // Update display immediately
        this.updateProcessingFpsDisplay();
        
        // Start periodic updates to clear old data and update display
        this.processingFpsTracker.updateInterval = setInterval(() => {
            this.trackProcessingFps();
        }, 1000);
    }
    
    /**
     * Stop processing FPS monitoring
     */
    stopProcessingFpsMonitoring() {
        if (this.processingFpsTracker.updateInterval) {
            clearInterval(this.processingFpsTracker.updateInterval);
            this.processingFpsTracker.updateInterval = null;
        }
        
        this.processingFpsTracker.inferences = [];
        this.processingFpsTracker.currentFps = 0;
        this.updateProcessingFpsDisplay();
    }
    
    /**
     * Handle mode toggle between auto and manual
     */
    handleModeToggle(event) {
        const isManual = !event.target.checked;
        
        // Only allow mode change when monitoring is stopped
        if (this.monitoringActive) {
            event.target.checked = !isManual;
            this.addActivityLog('Mode cannot be changed during monitoring. Stop monitoring to switch modes.');
            return;
        }
        
        this.mode = isManual ? 'manual' : 'auto';
        this.manualModeEnabled = isManual;
        
        console.log('[Professional App] Mode changed to:', this.mode);
        this.addActivityLog(`Switched to ${this.mode} mode`);
        
        // Update UI based on mode
        this.updateModeUI(isManual);
        
        // Update node descriptions
        this.updateNodeDescriptions();
    }
    
    /**
     * Update UI elements based on mode
     */
    updateModeUI(isManual) {
        const autoOnlyButtons = document.querySelectorAll('.auto-only');
        const manualOnlyButtons = document.querySelectorAll('.manual-only');
        const pipelineNodes = document.querySelectorAll('.pipeline-node');
        
        if (isManual) {
            // Hide auto buttons, show manual buttons
            autoOnlyButtons.forEach(btn => btn.style.display = 'none');
            manualOnlyButtons.forEach(btn => btn.style.display = 'inline-flex');
            
            // Make nodes clickable
            pipelineNodes.forEach(node => node.classList.add('clickable'));
        } else {
            // Show auto buttons, hide manual buttons
            autoOnlyButtons.forEach(btn => btn.style.display = 'inline-flex');
            manualOnlyButtons.forEach(btn => btn.style.display = 'none');
            
            // Remove clickable from nodes
            pipelineNodes.forEach(node => node.classList.remove('clickable'));
        }
    }
    
    /**
     * Handle pipeline node click in manual mode
     */
    handleNodeClick(event) {
        if (!this.manualModeEnabled || !this.monitoringActive) {
            return;
        }
        
        // Check if node clicking is enabled (all agents completed at least once)
        if (!this.nodeClickingEnabled) {
            // Provide feedback that node clicking is disabled
            console.log('[Professional App] Node clicking disabled until all agents complete once');
            this.addActivityLog('Node clicking disabled - complete all agents first');
            
            // Visual feedback - brief highlight animation
            const node = event.currentTarget;
            node.style.animation = 'shake 0.3s';
            setTimeout(() => {
                node.style.animation = '';
            }, 300);
            
            return;
        }
        
        const node = event.currentTarget;
        const agentName = node.dataset.agent;
        
        console.log('[Professional App] Manual node selection:', agentName);
        this.sendManualSelectAgent(agentName);
    }
    
    /**
     * Send manual previous agent command
     */
    sendManualPrevious() {
        if (!this.manualModeEnabled || !this.monitoringActive || !this.sessionId) {
            console.log('[Professional App] Cannot send manual command - no active session');
            return;
        }
        
        const command = {
            type: 'manual_previous',
            session_id: this.sessionId,
            timestamp: Date.now()
        };
        
        console.log('[Professional App] Sending manual previous command with session:', this.sessionId);
        this.sendDataChannelMessage(command);
        this.addActivityLog('Manual: Previous agent');
    }
    
    /**
     * Send manual next agent command
     */
    sendManualNext() {
        if (!this.manualModeEnabled || !this.monitoringActive || !this.sessionId) {
            console.log('[Professional App] Cannot send manual command - no active session');
            return;
        }
        
        const command = {
            type: 'manual_next',
            session_id: this.sessionId,
            timestamp: Date.now()
        };
        
        console.log('[Professional App] Sending manual next command with session:', this.sessionId);
        this.sendDataChannelMessage(command);
        this.addActivityLog('Manual: Next agent');
    }
    
    /**
     * Send manual redo current agent command
     */
    sendManualRedo() {
        if (!this.manualModeEnabled || !this.monitoringActive || !this.sessionId) {
            console.log('[Professional App] Cannot send manual command - no active session');
            return;
        }
        
        const command = {
            type: 'manual_redo',
            session_id: this.sessionId,
            agent: this.currentAgent,
            timestamp: Date.now()
        };
        
        console.log('[Professional App] Sending manual redo command for:', this.currentAgent, 'with session:', this.sessionId);
        this.sendDataChannelMessage(command);
        this.addActivityLog(`Manual: Redo ${this.currentAgent}`);
    }
    
    /**
     * Send manual select specific agent command
     */
    sendManualSelectAgent(agentName) {
        if (!this.manualModeEnabled || !this.monitoringActive || !this.sessionId) {
            console.log('[Professional App] Cannot send manual command - no active session');
            return;
        }
        
        const command = {
            type: 'manual_select_agent',
            session_id: this.sessionId,
            agent: agentName,  // Changed from target_agent to agent to match backend
            timestamp: Date.now()
        };
        
        console.log('[Professional App] Sending manual select agent command:', agentName, 'with session:', this.sessionId);
        this.sendDataChannelMessage(command);
        this.addActivityLog(`Manual: Jump to ${agentName}`);
        
        // Update current agent (visual feedback will come from backend status update)
        this.currentAgent = agentName;
    }
    
    /**
     * Send message through data channel
     */
    sendDataChannelMessage(message) {
        if (this.webrtcClient && this.webrtcClient.dataChannel && 
            this.webrtcClient.dataChannel.readyState === 'open') {
            this.webrtcClient.dataChannel.send(JSON.stringify(message));
        } else {
            console.warn('[Professional App] Data channel not ready for message:', message);
        }
    }
    
    /**
     * Initialize node descriptions on page load
     */
    initializeNodeDescriptions() {
        const currentDesc = document.getElementById('currentDescription');
        const nextDesc = document.getElementById('nextDescription');
        
        if (currentDesc && nextDesc) {
            currentDesc.innerHTML = '<strong>Ready:</strong> Click Start to begin classification';
            nextDesc.innerHTML = 'Auto mode will process through all agents automatically';
        }
    }
    
    /**
     * Update node descriptions based on current mode and state
     */
    updateNodeDescriptions() {
        const descriptionsContainer = document.getElementById('nodeDescriptions');
        const currentDesc = document.getElementById('currentDescription');
        const nextDesc = document.getElementById('nextDescription');
        
        if (!descriptionsContainer) return;
        
        if (this.mode === 'manual' && this.monitoringActive) {
            currentDesc.innerHTML = '<strong>Manual Mode:</strong> Click on nodes to select agent';
            nextDesc.innerHTML = 'Use Previous/Next buttons or click nodes to navigate';
        } else if (this.mode === 'auto' && this.monitoringActive) {
            // Will be updated by agent status messages
            currentDesc.innerHTML = '<strong>Auto Mode:</strong> Processing automatically';
            nextDesc.innerHTML = 'Agents will process in sequence: Initial → Details → Damage';
        } else if (!this.monitoringActive) {
            currentDesc.innerHTML = '<strong>Ready:</strong> Click Start to begin classification';
            nextDesc.innerHTML = this.mode === 'manual' ? 'Manual mode - control agent progression' : 'Auto mode - automatic agent progression';
        } else {
            currentDesc.innerHTML = '';
            nextDesc.innerHTML = '';
        }
    }
}

/**
 * Initialize the professional app after initialization completes
 */
window.initializeApp = function() {
    console.log('[Professional App] Starting professional application');
    window.appController = new ProfessionalApplicationController();
};

// Do NOT auto-initialize - let InitializationManager handle it