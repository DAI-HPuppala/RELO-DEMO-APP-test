/**
 * WebRTC Client V2 - Enhanced client with stateful orchestration support
 * Supports pause/resume, multi-inference tracking, and checkpoint recovery
 */

class WebRTCClientV2 {
    constructor(wsUrl) {
        this.ws = null;
        this.pc = null;
        this.localStream = null;
        this.sessionId = null;
        this.isConnected = false;
        
        // State tracking
        this.currentAgent = null;
        this.agentTimers = {};
        this.inferenceCounters = {};
        this.sessionState = 'IDLE';
        
        // Callbacks
        this.onStateChange = null;
        this.onAgentStarted = null;
        this.onAgentCompleted = null;
        this.onInferenceUpdate = null;
        this.onSessionPaused = null;
        this.onSessionResumed = null;
        this.onError = null;
        this.onconnected = null;
        this.onstream = null;
        
        // WebSocket URL (use provided URL or default)
        this.wsUrl = wsUrl || 'ws://localhost:8000/ws/v2/stream';
    }
    
    /**
     * Initialize WebSocket connection
     */
    async connect() {
        return new Promise((resolve, reject) => {
            console.log('Connecting to WebSocket:', this.wsUrl);
            this.ws = new WebSocket(this.wsUrl);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.isConnected = true;
                
                // Trigger onconnected callback
                if (this.onconnected) {
                    this.onconnected();
                }
                
                // Resolve immediately after connection
                resolve();
            };
            
            this.ws.onmessage = (event) => {
                this.handleMessage(JSON.parse(event.data));
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.isConnected = false;
                if (this.onError) {
                    this.onError(error);
                }
                reject(error);
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.isConnected = false;
            };
        });
    }
    
    /**
     * Start camera and create WebRTC offer
     */
    async startCamera() {
        try {
            // Get user media
            this.localStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    frameRate: { ideal: 30 }
                },
                audio: false
            });
            
            // Create peer connection - OFFLINE MODE: empty ICE servers for local network
            this.pc = new RTCPeerConnection({
                iceServers: []  // Empty = works offline on local network
            });
            
            // Add local stream tracks
            this.localStream.getTracks().forEach(track => {
                this.pc.addTrack(track, this.localStream);
            });
            
            // Handle ICE candidates
            this.pc.onicecandidate = (event) => {
                if (event.candidate) {
                    console.log('ICE candidate:', event.candidate);
                }
            };
            
            // Create offer
            const offer = await this.pc.createOffer();
            await this.pc.setLocalDescription(offer);
            
            // Generate session ID
            this.sessionId = this.generateSessionId();
            
            // Send offer to server
            this.sendMessage({
                type: 'offer',
                session_id: this.sessionId,
                sdp: offer.sdp
            });
            
            return this.localStream;
            
        } catch (error) {
            console.error('Failed to start camera:', error);
            throw error;
        }
    }
    
    /**
     * Start monitoring flow
     */
    async startMonitoring(mode = 'automatic') {
        if (!this.sessionId) {
            throw new Error('No active session');
        }
        
        this.sendMessage({
            type: 'start_monitoring',
            session_id: this.sessionId,
            mode: mode
        });
    }
    
    /**
     * Pause the current flow
     */
    async pauseFlow() {
        if (!this.sessionId) {
            throw new Error('No active session');
        }
        
        this.sendMessage({
            type: 'pause_flow',
            session_id: this.sessionId
        });
    }
    
    /**
     * Resume the paused flow
     */
    async resumeFlow(restartAgent = false) {
        if (!this.sessionId) {
            throw new Error('No active session');
        }
        
        this.sendMessage({
            type: 'resume_flow',
            session_id: this.sessionId,
            restart_agent: restartAgent
        });
    }
    
    /**
     * Get current session state
     */
    async getSessionState() {
        if (!this.sessionId) {
            throw new Error('No active session');
        }
        
        this.sendMessage({
            type: 'get_session_state',
            session_id: this.sessionId
        });
    }
    
    /**
     * Recover session from checkpoint
     */
    async recoverSession() {
        if (!this.sessionId) {
            throw new Error('No active session');
        }
        
        this.sendMessage({
            type: 'recover_session',
            session_id: this.sessionId
        });
    }
    
    /**
     * Get checkpoint information
     */
    async getCheckpointInfo() {
        if (!this.sessionId) {
            throw new Error('No active session');
        }
        
        this.sendMessage({
            type: 'get_checkpoint_info',
            session_id: this.sessionId
        });
    }
    
    /**
     * Set custom timers for testing
     */
    async setCustomTimers(timers) {
        if (!this.sessionId) {
            throw new Error('No active session');
        }
        
        this.sendMessage({
            type: 'set_custom_timers',
            session_id: this.sessionId,
            timers: timers
        });
    }
    
    /**
     * Handle incoming WebSocket messages
     */
    handleMessage(message) {
        console.log('Received message:', message.type, message);
        
        switch (message.type) {
            case 'answer':
                this.handleAnswer(message);
                break;
                
            case 'agent_started':
                this.handleAgentStarted(message);
                break;
                
            case 'agent_completed':
                this.handleAgentCompleted(message);
                break;
                
            case 'inference_update':
                this.handleInferenceUpdate(message);
                break;
                
            case 'flow_paused':
                this.handleFlowPaused(message);
                break;
                
            case 'flow_resumed':
                this.handleFlowResumed(message);
                break;
                
            case 'classification_complete':
                this.handleClassificationComplete(message);
                break;
                
            case 'session_state':
                this.handleSessionState(message);
                break;
                
            case 'session_recovered':
                this.handleSessionRecovered(message);
                break;
                
            case 'checkpoint_info':
                this.handleCheckpointInfo(message);
                break;
                
            case 'error':
                this.handleError(message);
                break;
                
            case 'agent_error':
                this.handleAgentError(message);
                break;
                
            default:
                console.warn('Unknown message type:', message.type);
        }
    }
    
    /**
     * Handle WebRTC answer
     */
    async handleAnswer(message) {
        try {
            const answer = new RTCSessionDescription({
                type: 'answer',
                sdp: message.sdp
            });
            await this.pc.setRemoteDescription(answer);
            console.log('WebRTC connection established');
        } catch (error) {
            console.error('Failed to set remote description:', error);
        }
    }
    
    /**
     * Handle agent started event
     */
    handleAgentStarted(message) {
        this.currentAgent = message.agent;
        this.agentTimers[message.agent] = {
            timerSeconds: message.timer_seconds,
            startTime: Date.now()
        };
        this.inferenceCounters[message.agent] = 0;
        
        if (this.onAgentStarted) {
            this.onAgentStarted(message);
        }
    }
    
    /**
     * Handle agent completed event
     */
    handleAgentCompleted(message) {
        const agent = message.agent;
        if (this.agentTimers[agent]) {
            this.agentTimers[agent].endTime = Date.now();
            this.agentTimers[agent].duration = 
                (this.agentTimers[agent].endTime - this.agentTimers[agent].startTime) / 1000;
        }
        
        if (this.onAgentCompleted) {
            this.onAgentCompleted(message);
        }
    }
    
    /**
     * Handle inference update
     */
    handleInferenceUpdate(message) {
        const agent = message.agent;
        if (agent && this.inferenceCounters[agent] !== undefined) {
            this.inferenceCounters[agent]++;
        }
        
        if (this.onInferenceUpdate) {
            this.onInferenceUpdate(message);
        }
    }
    
    /**
     * Handle flow paused
     */
    handleFlowPaused(message) {
        this.sessionState = 'PAUSED';
        if (this.onSessionPaused) {
            this.onSessionPaused(message);
        }
    }
    
    /**
     * Handle flow resumed
     */
    handleFlowResumed(message) {
        this.sessionState = 'RUNNING';
        if (this.onSessionResumed) {
            this.onSessionResumed(message);
        }
    }
    
    /**
     * Handle classification complete
     */
    handleClassificationComplete(message) {
        this.sessionState = 'COMPLETED';
        console.log('Classification complete:', message.final_result);
        
        if (this.onStateChange) {
            this.onStateChange({
                state: 'COMPLETED',
                result: message.final_result
            });
        }
    }
    
    /**
     * Handle session state update
     */
    handleSessionState(message) {
        this.sessionState = message.status;
        console.log('Session state:', message);
    }
    
    /**
     * Handle session recovered
     */
    handleSessionRecovered(message) {
        console.log('Session recovered from checkpoint:', message);
    }
    
    /**
     * Handle checkpoint info
     */
    handleCheckpointInfo(message) {
        console.log('Checkpoint info:', message);
    }
    
    /**
     * Handle error message
     */
    handleError(message) {
        console.error('Error:', message.message);
        if (this.onError) {
            this.onError(message);
        }
    }
    
    /**
     * Handle agent error
     */
    handleAgentError(message) {
        console.error(`Agent ${message.agent} error:`, message.error);
        console.log('Action:', message.action);
        
        if (this.onError) {
            this.onError(message);
        }
    }
    
    /**
     * Send message via WebSocket
     */
    sendMessage(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        } else {
            console.error('WebSocket not connected');
        }
    }
    
    /**
     * Generate unique session ID
     */
    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    /**
     * Get agent statistics
     */
    getAgentStats() {
        const stats = {};
        
        Object.keys(this.agentTimers).forEach(agent => {
            stats[agent] = {
                timer: this.agentTimers[agent],
                inferences: this.inferenceCounters[agent] || 0
            };
        });
        
        return stats;
    }
    
    /**
     * Disconnect and cleanup
     */
    disconnect() {
        if (this.localStream) {
            this.localStream.getTracks().forEach(track => track.stop());
            this.localStream = null;
        }
        
        if (this.pc) {
            this.pc.close();
            this.pc = null;
        }
        
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        
        this.isConnected = false;
        this.sessionId = null;
        this.currentAgent = null;
        this.agentTimers = {};
        this.inferenceCounters = {};
        this.sessionState = 'IDLE';
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = WebRTCClientV2;
}