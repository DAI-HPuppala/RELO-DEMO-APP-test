/**
 * Initialization Manager
 * Handles the initialization sequence for all system components
 */
class InitializationManager {
    constructor() {
        this.steps = [
            { id: 'backend', name: 'Backend Server', status: 'pending' },
            { id: 'camera', name: 'RealSense Camera', status: 'pending' },
            { id: 'websocket', name: 'WebSocket Connection', status: 'pending' },
            { id: 'webrtc', name: 'WebRTC Stream', status: 'pending' }
        ];
        this.currentStep = 0;
        this.webrtcClient = null;
        this.initPromise = null;
    }

    /**
     * Start the initialization process
     */
    async initialize() {
        // Prevent multiple initialization
        if (this.initPromise) {
            return this.initPromise;
        }

        this.initPromise = this._runInitialization();
        return this.initPromise;
    }

    async _runInitialization() {
        try {
            console.log('Starting initialization sequence...');
            
            // Step 1: Check backend server
            await this.initializeBackend();
            
            // Step 2: Initialize camera
            await this.initializeCamera();
            
            // Step 3: Connect WebSocket
            await this.initializeWebSocket();
            
            // Step 4: Setup WebRTC
            await this.initializeWebRTC();
            
            // All steps completed
            this.updateProgress(100);
            console.log('Initialization completed successfully!');
            
            // Wait a moment for the animation
            await this.delay(500);
            
            // Transition to main app
            this.transitionToMainApp();
            
            return { success: true };
            
        } catch (error) {
            console.error('Initialization failed:', error);
            this.showError(error.message);
            return { success: false, error: error.message };
        }
    }

    /**
     * Initialize backend server connection
     */
    async initializeBackend() {
        this.updateStepStatus('backend', 'active', 'Connecting to server...');
        
        try {
            // Check if backend is running
            const response = await fetch('http://localhost:8000/api/health', {
                method: 'GET',
                mode: 'cors'
            });
            
            if (response.ok) {
                const data = await response.json();
                console.log('Backend health check:', data);
                
                await this.delay(500); // Show progress
                this.updateStepStatus('backend', 'completed', 'Connected');
                this.updateProgress(25);
            } else {
                throw new Error('Backend server is not responding');
            }
        } catch (error) {
            this.updateStepStatus('backend', 'error', 'Connection failed');
            throw new Error(`Backend initialization failed: ${error.message}`);
        }
    }

    /**
     * Initialize camera
     */
    async initializeCamera() {
        this.updateStepStatus('camera', 'active', 'Initializing camera...');
        
        try {
            // Check camera status via API
            const response = await fetch('http://localhost:8000/api/camera/status', {
                method: 'GET',
                mode: 'cors'
            });
            
            if (response.ok) {
                const data = await response.json();
                console.log('Camera status:', data);
                
                if (data.available) {
                    await this.delay(800); // Simulate camera initialization
                    this.updateStepStatus('camera', 'completed', `${data.device_name || 'RealSense'} ready`);
                    this.updateProgress(50);
                    
                    // Update camera status indicator
                    this.updateStatusIndicators(true, false, false);
                } else {
                    // Camera not available but not critical
                    this.updateStepStatus('camera', 'completed', 'Using test pattern');
                    this.updateProgress(50);
                    
                    // Camera not actually active, but mark step as complete
                    this.updateStatusIndicators(false, false, false);
                }
            } else {
                // If endpoint doesn't exist, assume camera will be initialized on demand
                await this.delay(500);
                this.updateStepStatus('camera', 'completed', 'Ready');
                this.updateProgress(50);
            }
        } catch (error) {
            console.warn('Camera check failed, will initialize on demand:', error);
            this.updateStepStatus('camera', 'completed', 'Will initialize on demand');
            this.updateProgress(50);
        }
    }

    /**
     * Initialize WebSocket connection
     */
    async initializeWebSocket() {
        this.updateStepStatus('websocket', 'active', 'Establishing connection...');
        
        try {
            // Create WebRTC client (which includes WebSocket)
            this.webrtcClient = new WebRTCClient();
            
            // Set up stream callback to handle video early
            this.webrtcClient.on('stream', (stream) => {
                console.log('Video stream received during initialization');
                // Store stream for the main app to use
                window.initVideoStream = stream;
                
                // Show video in preview during initialization
                const videoElement = document.getElementById('videoPreview');
                if (videoElement) {
                    videoElement.srcObject = stream;
                    const overlay = document.getElementById('videoOverlay');
                    if (overlay) {
                        overlay.style.display = 'none';
                    }
                }
                
                // Update status indicators immediately when stream is received
                this.updateStatusIndicators(true, true, true);
            });
            
            // Connect WebSocket
            await this.webrtcClient.connect();
            
            await this.delay(300);
            this.updateStepStatus('websocket', 'completed', 'Connected');
            this.updateProgress(75);
            
            // Update connection status indicator
            this.updateStatusIndicators(false, true, false);
            
        } catch (error) {
            this.updateStepStatus('websocket', 'error', 'Connection failed');
            throw new Error(`WebSocket initialization failed: ${error.message}`);
        }
    }

    /**
     * Initialize WebRTC
     */
    async initializeWebRTC() {
        this.updateStepStatus('webrtc', 'active', 'Setting up video stream...');
        
        try {
            // Start a session to initialize WebRTC
            await this.webrtcClient.startSession('automatic', 'realsense');
            
            // Wait for WebRTC to be ready
            await this.waitForWebRTC();
            
            await this.delay(500);
            this.updateStepStatus('webrtc', 'completed', 'Stream ready');
            this.updateProgress(100);
            
            // Store the client globally for the app to use
            window.webrtcClient = this.webrtcClient;
            
        } catch (error) {
            this.updateStepStatus('webrtc', 'error', 'Stream setup failed');
            throw new Error(`WebRTC initialization failed: ${error.message}`);
        }
    }

    /**
     * Wait for WebRTC connection to be established
     */
    async waitForWebRTC() {
        return new Promise((resolve, reject) => {
            let attempts = 0;
            const maxAttempts = 100; // 10 seconds timeout
            
            const checkInterval = setInterval(() => {
                if (this.webrtcClient && this.webrtcClient.pc) {
                    const state = this.webrtcClient.pc.connectionState;
                    console.log('WebRTC connection state:', state);
                    
                    // Accept both 'connected' and 'connecting' as success for initialization
                    if (state === 'connected' || state === 'connecting') {
                        clearInterval(checkInterval);
                        resolve();
                    } else if (state === 'failed' || state === 'closed') {
                        clearInterval(checkInterval);
                        reject(new Error(`WebRTC connection ${state}`));
                    }
                }
                
                // Also check if we have a video stream, which indicates success
                if (this.webrtcClient && this.webrtcClient.remoteStream) {
                    console.log('WebRTC stream detected, initialization complete');
                    clearInterval(checkInterval);
                    resolve();
                    return;
                }
                
                attempts++;
                if (attempts >= maxAttempts) {
                    console.log('WebRTC timeout, but proceeding anyway as backend is connected');
                    clearInterval(checkInterval);
                    resolve(); // Don't reject, just proceed
                }
            }, 100);
        });
    }

    /**
     * Update step status in UI
     */
    updateStepStatus(stepId, status, statusText) {
        const stepElement = document.getElementById(`step-${stepId}`);
        if (!stepElement) return;
        
        // Remove all status classes
        stepElement.classList.remove('active', 'completed', 'error');
        
        // Add new status class
        if (status !== 'pending') {
            stepElement.classList.add(status);
        }
        
        // Update status text
        const statusElement = stepElement.querySelector('.step-status');
        if (statusElement && statusText) {
            statusElement.textContent = statusText;
        }
        
        // Update icon
        const spinner = stepElement.querySelector('.spinner');
        const checkmark = stepElement.querySelector('.checkmark');
        
        if (status === 'completed') {
            if (spinner) spinner.style.display = 'none';
            if (checkmark) checkmark.style.display = 'block';
        } else if (status === 'active') {
            if (spinner) spinner.style.display = 'block';
            if (checkmark) checkmark.style.display = 'none';
        }
    }

    /**
     * Update overall progress
     */
    updateProgress(percentage) {
        const progressFill = document.getElementById('initProgress');
        const progressText = document.getElementById('initProgressText');
        
        if (progressFill) {
            progressFill.style.width = `${percentage}%`;
        }
        
        if (progressText) {
            progressText.textContent = `${percentage}% Complete`;
        }
    }

    /**
     * Show error message
     */
    showError(message) {
        const errorElement = document.getElementById('initError');
        if (errorElement) {
            const messageElement = errorElement.querySelector('.error-message');
            if (messageElement) {
                messageElement.textContent = message;
            }
            errorElement.style.display = 'block';
        }
    }

    /**
     * Transition to main application
     */
    transitionToMainApp() {
        const initScreen = document.getElementById('initScreen');
        const mainApp = document.getElementById('mainApp');
        
        if (initScreen && mainApp) {
            // Fade out init screen
            initScreen.style.opacity = '0';
            
            setTimeout(() => {
                initScreen.style.display = 'none';
                mainApp.style.display = 'block';
                
                // Update status indicators
                this.updateMainAppIndicators();
                
                // Initialize main app
                if (window.initializeApp) {
                    window.initializeApp();
                }
            }, 500);
        }
    }

    /**
     * Update status indicators during initialization
     */
    updateStatusIndicators(camera, connection, stream) {
        // Update camera indicator
        const cameraIndicator = document.getElementById('cameraIndicator');
        if (cameraIndicator) {
            const dot = cameraIndicator.querySelector('.indicator-dot');
            if (camera) {
                cameraIndicator.classList.add('active');
                if (dot) dot.classList.add('active');
            } else {
                cameraIndicator.classList.remove('active');
                if (dot) dot.classList.remove('active');
            }
        }
        
        // Update connection indicator
        const connectionIndicator = document.getElementById('connectionIndicator');
        if (connectionIndicator) {
            const dot = connectionIndicator.querySelector('.indicator-dot');
            if (connection) {
                connectionIndicator.classList.add('active');
                if (dot) dot.classList.add('active');
            } else {
                connectionIndicator.classList.remove('active');
                if (dot) dot.classList.remove('active');
            }
        }
        
        // Update stream indicator
        const streamIndicator = document.getElementById('streamIndicator');
        if (streamIndicator) {
            const dot = streamIndicator.querySelector('.indicator-dot');
            if (stream) {
                streamIndicator.classList.add('active');
                if (dot) dot.classList.add('active');
            } else {
                streamIndicator.classList.remove('active');
                if (dot) dot.classList.remove('active');
            }
        }
        
        // Also use the status monitor if available
        if (window.statusMonitor) {
            window.statusMonitor.updateCameraStatus(camera);
            window.statusMonitor.updateConnectionStatus(connection);
            window.statusMonitor.updateStreamStatus(stream);
        }
    }

    /**
     * Update main app status indicators
     */
    updateMainAppIndicators() {
        // Update camera indicator
        const cameraIndicator = document.getElementById('cameraIndicator');
        if (cameraIndicator) {
            cameraIndicator.classList.add('active');
        }
        
        // Update connection indicator
        const connectionIndicator = document.getElementById('connectionIndicator');
        if (connectionIndicator) {
            connectionIndicator.classList.add('active');
        }
        
        // Update stream indicator
        const streamIndicator = document.getElementById('streamIndicator');
        if (streamIndicator) {
            streamIndicator.classList.add('active');
        }
    }

    /**
     * Utility function to add delay
     */
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

/**
 * Retry initialization
 */
window.retryInitialization = async function() {
    // Hide error
    const errorElement = document.getElementById('initError');
    if (errorElement) {
        errorElement.style.display = 'none';
    }
    
    // Reset all steps
    const manager = window.initManager;
    if (manager) {
        manager.steps.forEach(step => {
            manager.updateStepStatus(step.id, 'pending', 'Waiting...');
        });
        manager.updateProgress(0);
        manager.currentStep = 0;
        manager.initPromise = null;
        
        // Restart initialization
        await manager.initialize();
    }
};

// Start initialization when page loads
document.addEventListener('DOMContentLoaded', async () => {
    console.log('Page loaded, starting initialization...');
    
    // Create initialization manager
    window.initManager = new InitializationManager();
    
    // Start initialization after a short delay for smooth animation
    setTimeout(async () => {
        await window.initManager.initialize();
    }, 500);
});