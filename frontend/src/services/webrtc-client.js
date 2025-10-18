/**
 * WebRTC Client Service for handling video streaming and data channels
 */
class WebRTCClient {
    constructor() {
        this.pc = null;
        this.ws = null;
        this.dataChannel = null;
        this.remoteStream = null;
        this.sessionId = null;
        this.tableRowLimit = 200; // Default limit, updated from backend
        this.isConnected = false;
        this.wsUrl = 'ws://localhost:8000/ws/stream';
        // OFFLINE MODE: Empty ICE servers for local network operation without internet
        this.configuration = {
            iceServers: []  // Empty = works offline on local network
        };
        this.callbacks = {
            onConnected: null,
            onDisconnected: null,
            onStream: null,
            onMessage: null,
            onError: null,
            onProgressUpdate: null,
            onStatusUpdate: null,
            onAgentStarted: null,
            onAgentCompleted: null,
            onFinalResults: null
        };
    }

    /**
     * Connect to WebSocket server
     */
    async connect() {
        return new Promise((resolve, reject) => {
            try {
                this.ws = new WebSocket(this.wsUrl);
                
                this.ws.onopen = () => {
                    this.isConnected = true;
                    if (this.callbacks.onConnected) {
                        this.callbacks.onConnected();
                    }
                    resolve();
                };

                this.ws.onmessage = async (event) => {
                    const message = JSON.parse(event.data);
                    await this.handleWebSocketMessage(message);
                };

                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    if (this.callbacks.onError) {
                        this.callbacks.onError(error);
                    }
                    reject(error);
                };

                this.ws.onclose = () => {
                    this.isConnected = false;
                    if (this.callbacks.onDisconnected) {
                        this.callbacks.onDisconnected();
                    }
                };
            } catch (error) {
                console.error('Failed to connect WebSocket:', error);
                reject(error);
            }
        });
    }

    /**
     * Start a new session
     */
    async startSession(mode = 'automatic', cameraSource = 'realsense') {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            throw new Error('WebSocket not connected');
        }

        const message = {
            type: 'start_session',
            mode: mode,
            camera_config: {
                source: cameraSource,
                resolution: '640x480',
                fps: 30
            }
        };

        this.ws.send(JSON.stringify(message));
    }

    /**
     * Handle incoming WebSocket messages
     */
    async handleWebSocketMessage(message) {
        switch (message.type) {
            case 'session_created':
                this.sessionId = message.session_id;
                this.tableRowLimit = message.table_row_limit || 200;
                await this.initializeWebRTC();
                break;

            case 'answer':
                await this.handleAnswer(message);
                break;

            case 'ice_candidate':
                await this.handleRemoteIceCandidate(message);
                break;

            case 'automatic_started':
                if (this.callbacks.onMessage) {
                    this.callbacks.onMessage(message);
                }
                break;
                
            case 'agent_result':
                if (this.callbacks.onMessage) {
                    this.callbacks.onMessage(message);
                }
                break;

            case 'session_stopped':
                // Clear session ID when session is stopped
                if (message.session_id === this.sessionId) {
                    this.sessionId = null;
                }
                if (this.callbacks.onMessage) {
                    this.callbacks.onMessage(message);
                }
                break;
                
            case 'manual_mode_started':
                if (this.callbacks.onMessage) {
                    this.callbacks.onMessage(message);
                }
                break;

            case 'auto_export_success':
                console.log('[WebRTC Client] Auto-export successful:', message.files);
                // Dispatch event for app-professional.js to listen to
                window.dispatchEvent(new CustomEvent('websocket_message', { detail: message }));
                if (this.callbacks.onMessage) {
                    this.callbacks.onMessage(message);
                }
                break;

            case 'error':
                console.error('Server error:', message);
                // Clear session on certain errors
                if (message.error_code === 'SESSION_NOT_FOUND') {
                    this.sessionId = null;
                }
                if (this.callbacks.onError) {
                    this.callbacks.onError(message);
                }
                break;

            default:
                break;
        }
    }

    /**
     * Initialize WebRTC peer connection
     */
    async initializeWebRTC() {
        try {
            // Create peer connection
            this.pc = new RTCPeerConnection(this.configuration);

            // Set up event handlers
            this.pc.onicecandidate = (event) => {
                if (event.candidate) {
                    this.sendIceCandidate(event.candidate);
                }
            };

            this.pc.ontrack = (event) => {
                this.remoteStream = event.streams[0];
                if (this.callbacks.onStream) {
                    this.callbacks.onStream(this.remoteStream);
                }
            };

            this.pc.onconnectionstatechange = () => {
                if (this.pc.connectionState === 'failed') {
                    console.error('WebRTC connection failed');
                }
            };

            // Create data channel on the client side BEFORE creating offer
            this.dataChannel = this.pc.createDataChannel('results', {
                ordered: true
            });
            this.setupDataChannel();
            
            // Also listen for data channel from backend (if created there)
            this.pc.ondatachannel = (event) => {
                // Use backend channel if we don't have one
                if (!this.dataChannel || this.dataChannel.readyState !== 'open') {
                    this.dataChannel = event.channel;
                    this.setupDataChannel();
                }
            };

            // Create offer
            const offer = await this.pc.createOffer({
                offerToReceiveVideo: true,
                offerToReceiveAudio: false
            });
            
            await this.pc.setLocalDescription(offer);

            // Send offer to server
            this.ws.send(JSON.stringify({
                type: 'offer',
                sdp: offer.sdp,
                session_id: this.sessionId
            }));

        } catch (error) {
            console.error('Failed to initialize WebRTC:', error);
            if (this.callbacks.onError) {
                this.callbacks.onError(error);
            }
        }
    }

    /**
     * Set up data channel event handlers
     */
    setupDataChannel() {
        this.dataChannel.onopen = () => {
            // Data channel ready
        };

        this.dataChannel.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                
                // Route all messages through the callback
                if (this.callbacks.onMessage) {
                    this.callbacks.onMessage(data);
                }
                
                // Also handle specific message types
                switch(data.type) {
                    case 'agent_started':
                        if (this.callbacks.onAgentStarted) {
                            this.callbacks.onAgentStarted(data);
                        }
                        break;
                    case 'progress_update':
                        if (this.callbacks.onProgressUpdate) {
                            this.callbacks.onProgressUpdate(data);
                        }
                        break;
                    case 'agent_completed':
                        if (this.callbacks.onAgentCompleted) {
                            this.callbacks.onAgentCompleted(data);
                        }
                        break;
                    case 'final_results':
                        if (this.callbacks.onFinalResults) {
                            this.callbacks.onFinalResults(data);
                        }
                        break;
                    case 'status_update':
                        if (this.callbacks.onStatusUpdate) {
                            this.callbacks.onStatusUpdate(data);
                        }
                        break;
                    case 'inference_update':
                        // Just status, no results yet
                        if (this.callbacks.onInferenceUpdate) {
                            this.callbacks.onInferenceUpdate(data);
                        }
                        break;
                    case 'inference_result':
                        // Actual attribute results
                        if (this.callbacks.onInferenceResult) {
                            this.callbacks.onInferenceResult(data);
                        }
                        break;
                    case 'manual_mode_status':
                        // Manual mode status update
                        if (this.callbacks.onManualModeStatus) {
                            this.callbacks.onManualModeStatus(data);
                        }
                        break;
                    case 'manual_mode_started':
                        // Manual mode started confirmation
                        if (this.callbacks.onManualModeStarted) {
                            this.callbacks.onManualModeStarted(data);
                        }
                        break;
                    case 'error':
                        // Handle error messages
                        console.error('[WebRTC Client] Error from backend:', data.message);
                        if (data.error_code === 'SESSION_NOT_FOUND' || 
                            data.error_code === 'MANUAL_MODE_NOT_ACTIVE') {
                            // Session is no longer valid, clean up
                            console.warn('[WebRTC Client] Session terminated, cleaning up');
                            this.sessionId = null;
                        }
                        if (this.callbacks.onError) {
                            this.callbacks.onError(data);
                        }
                        break;
                }
            } catch (error) {
                console.error('Failed to parse data channel message:', error);
            }
        };

        this.dataChannel.onerror = (error) => {
            console.error('Data channel error:', error);
        };

        this.dataChannel.onclose = () => {
            // Data channel closed
        };
    }

    /**
     * Handle WebRTC answer from server
     */
    async handleAnswer(message) {
        try {
            const answer = new RTCSessionDescription({
                type: 'answer',
                sdp: message.sdp
            });
            await this.pc.setRemoteDescription(answer);
        } catch (error) {
            console.error('Failed to handle answer:', error);
        }
    }

    /**
     * Handle remote ICE candidate
     */
    async handleRemoteIceCandidate(message) {
        try {
            if (message.candidate) {
                await this.pc.addIceCandidate(new RTCIceCandidate(message.candidate));
            }
        } catch (error) {
            console.error('Failed to add ICE candidate:', error);
        }
    }

    /**
     * Send ICE candidate to server
     */
    sendIceCandidate(candidate) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'ice_candidate',
                candidate: candidate,
                session_id: this.sessionId
            }));
        }
    }

    /**
     * Stop the current session
     */
    stopSession() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN && this.sessionId) {
            this.ws.send(JSON.stringify({
                type: 'stop_session',
                session_id: this.sessionId
            }));
        }

        this.cleanup();
    }

    /**
     * Switch operating mode
     */
    switchMode(mode) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN && this.sessionId) {
            this.ws.send(JSON.stringify({
                type: 'switch_mode',
                session_id: this.sessionId,
                mode: mode
            }));
        }
    }

    /**
     * Check if WebRTC connection is established
     */
    getConnectionStatus() {
        return this.isConnected;
    }
    
    /**
     * Check if data channel is ready
     */
    isDataChannelReady() {
        return this.dataChannel && this.dataChannel.readyState === 'open';
    }
    
    /**
     * Wait for data channel to be ready
     */
    async waitForDataChannel(timeout = 5000) {
        const startTime = Date.now();
        
        while (Date.now() - startTime < timeout) {
            if (this.isDataChannelReady()) {
                return true;
            }
            await new Promise(resolve => setTimeout(resolve, 100));
        }

        console.warn('Data channel timeout');
        return false;
    }
    
    /**
     * Trigger manual analysis for a specific agent
     */
    triggerAnalysis(agent = null) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN && this.sessionId) {
            this.ws.send(JSON.stringify({
                type: 'manual_trigger',
                session_id: this.sessionId,
                agent: agent
            }));
        }
    }
    
    /**
     * Trigger all agents in manual mode
     */
    /**
     * Trigger a specific agent (alias for triggerAnalysis)
     */
    triggerAgent(agentType) {
        // Map friendly names to actual agent IDs
        const agentMap = {
            'initial': 'initial_classifier',
            'detail': 'detail_extractor',
            'damage': 'damage_detector',
            'final': 'final_compiler'
        };
        const agentId = agentMap[agentType] || agentType;
        this.triggerAnalysis(agentId);
    }
    
    triggerAllAgents() {
        const agents = ['initial_classifier', 'detail_extractor', 'damage_detector', 'final_compiler'];
        agents.forEach((agent, index) => {
            setTimeout(() => {
                this.triggerAnalysis(agent);
            }, index * 1000); // Stagger the triggers
        });
    }
    
    /**
     * Start automatic processing with agents
     */
    startAutomatic(agentTimers = null, mode = 'automatic') {
        if (this.ws && this.ws.readyState === WebSocket.OPEN && this.sessionId) {
            const message = {
                type: 'start_automatic',
                session_id: this.sessionId,
                manual_mode: mode === 'manual'
            };
            
            // Use custom timers or defaults
            if (!agentTimers) {
                agentTimers = {
                    initial_classifier: 4.0,
                    detail_extractor: 3.0,
                    damage_detector: 4.0,
                    final_compiler: 2.0
                };
            }
            message.agent_timers = agentTimers;
            
            console.log('[WebRTC Client] Starting monitoring in', mode, 'mode');
            this.ws.send(JSON.stringify(message));
        } else {
            console.error('[WebRTC Client] Cannot start monitoring:', {
                ws: !!this.ws,
                wsReady: this.ws?.readyState === WebSocket.OPEN,
                sessionId: this.sessionId
            });
        }
    }

    /**
     * Send manual mode command via data channel
     */
    sendManualCommand(commandType, data = {}) {
        // Check if session is still active
        if (!this.sessionId) {
            console.warn('[WebRTC Client] No active session for manual command');
            if (this.callbacks.onError) {
                this.callbacks.onError({
                    type: 'error',
                    error_code: 'NO_SESSION',
                    message: 'No active session. Please start a new session.'
                });
            }
            return;
        }
        
        if (this.dataChannel && this.dataChannel.readyState === 'open') {
            const message = {
                type: commandType,
                session_id: this.sessionId,
                timestamp: Date.now(),
                ...data
            };
            this.dataChannel.send(JSON.stringify(message));
        } else {
            console.warn('[WebRTC Client] Data channel not ready for manual command');
            if (this.callbacks.onError) {
                this.callbacks.onError({
                    type: 'error',
                    error_code: 'CHANNEL_NOT_READY',
                    message: 'Connection not ready. Please wait or restart the session.'
                });
            }
        }
    }
    
    /**
     * Manual mode navigation commands
     */
    manualNext() {
        this.sendManualCommand('manual_next');
    }
    
    manualPrevious() {
        this.sendManualCommand('manual_previous');
    }
    
    manualRedo() {
        this.sendManualCommand('manual_redo');
    }
    
    manualSelectAgent(agentName) {
        this.sendManualCommand('manual_select_agent', { agent: agentName });
    }
    
    manualConfirmFinal() {
        this.sendManualCommand('manual_confirm_final');
    }

    /**
     * Tap-to-focus: Send focus coordinates to camera
     * @param {number} x - Normalized x coordinate (0-1)
     * @param {number} y - Normalized y coordinate (0-1)
     */
    tapToFocus(x, y) {
        this.sendManualCommand('tap_to_focus', { x, y });
    }

    /**
     * Send a generic message via WebSocket
     * @param {Object} message - Message object to send
     */
    sendMessage(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        } else {
            console.warn('[WebRTC Client] WebSocket not ready to send message');
        }
    }

    /**
     * Export results
     */
    exportResults(format = 'json') {
        if (this.ws && this.ws.readyState === WebSocket.OPEN && this.sessionId) {
            this.ws.send(JSON.stringify({
                type: 'export_results',
                session_id: this.sessionId,
                format: format
            }));
        }
    }

    /**
     * Clean up connections
     */
    cleanup() {
        if (this.dataChannel) {
            this.dataChannel.close();
            this.dataChannel = null;
        }

        if (this.pc) {
            this.pc.close();
            this.pc = null;
        }

        this.remoteStream = null;
        this.sessionId = null;
    }

    /**
     * Disconnect completely
     */
    disconnect() {
        this.cleanup();

        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }

        this.isConnected = false;
    }

    /**
     * Set callback functions
     */
    on(event, callback) {
        if (this.callbacks.hasOwnProperty('on' + event.charAt(0).toUpperCase() + event.slice(1))) {
            this.callbacks['on' + event.charAt(0).toUpperCase() + event.slice(1)] = callback;
        }
    }
}

// Export for use in other modules
window.WebRTCClient = WebRTCClient;