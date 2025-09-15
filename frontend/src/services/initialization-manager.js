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
            { id: 'webrtc', name: 'WebRTC Stream', status: 'pending' },
            { id: 'gpu', name: 'GPU Initialization', status: 'pending' },
            { id: 'vlm', name: 'VLM Engine', status: 'pending' }
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
            
            // Step 5: GPU Initialization
            await this.initializeGPU();
            
            // Step 6: VLM Engine Warmup
            await this.initializeVLM();
            
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
                this.updateProgress(17);  // 1/6 steps
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
                    this.updateProgress(34);  // 2/6 steps
                    
                    // Update camera status indicator
                    this.updateStatusIndicators(true, false, false);
                } else {
                    // Camera not available but not critical
                    this.updateStepStatus('camera', 'completed', 'Using test pattern');
                    this.updateProgress(34);  // 2/6 steps
                    
                    // Camera not actually active, but mark step as complete
                    this.updateStatusIndicators(false, false, false);
                }
            } else {
                // If endpoint doesn't exist, assume camera will be initialized on demand
                await this.delay(500);
                this.updateStepStatus('camera', 'completed', 'Ready');
                this.updateProgress(34);  // 2/6 steps
            }
        } catch (error) {
            console.warn('Camera check failed, will initialize on demand:', error);
            this.updateStepStatus('camera', 'completed', 'Will initialize on demand');
            this.updateProgress(34);  // 2/6 steps
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
            this.updateProgress(50);  // 3/6 steps
            
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
            this.updateProgress(67);  // 4/6 steps
            
            // Store the client globally for the app to use
            window.webrtcClient = this.webrtcClient;
            
        } catch (error) {
            this.updateStepStatus('webrtc', 'error', 'Stream setup failed');
            throw new Error(`WebRTC initialization failed: ${error.message}`);
        }
    }

    /**
     * Initialize GPU
     */
    async initializeGPU() {
        this.updateStepStatus('gpu', 'active', 'Detecting GPU...');
        
        try {
            // Check GPU status from backend
            const response = await fetch('http://localhost:8000/api/vlm/status', {
                method: 'GET',
                mode: 'cors'
            });
            
            if (response.ok) {
                const data = await response.json();
                let statusMessage = 'GPU Ready';
                
                if (data.gpu_info && data.gpu_info.name) {
                    statusMessage = `${data.gpu_info.name} Ready`;
                    console.log(`GPU: ${data.gpu_info.name}`);
                    console.log(`Memory: ${data.gpu_info.memory_free}MB free / ${data.gpu_info.memory_total}MB total`);
                }
                
                await this.delay(500);
                this.updateStepStatus('gpu', 'completed', statusMessage);
                this.updateProgress(84);  // 5/6 steps
            } else {
                // GPU not critical, continue
                this.updateStepStatus('gpu', 'completed', 'CPU Mode');
                this.updateProgress(84);  // 5/6 steps
            }
        } catch (error) {
            console.warn('GPU check failed:', error);
            this.updateStepStatus('gpu', 'completed', 'CPU Fallback');
            this.updateProgress(84);  // 5/6 steps
        }
    }

    /**
     * Initialize VLM with GPU warmup
     */
    async initializeVLM() {
        this.updateStepStatus('vlm', 'active', 'Checking VLM readiness...');
        
        try {
            // Quick check if VLM is already warmed up
            const quickCheck = await fetch('http://localhost:8000/api/vlm/quick-check', {
                method: 'GET',
                mode: 'cors'
            });
            
            if (quickCheck.ok) {
                const quickData = await quickCheck.json();
                console.log('VLM quick check:', quickData);
                
                if (quickData.ready) {
                    // VLM already ready
                    await this.delay(300);
                    this.updateStepStatus('vlm', 'completed', '✅ Already optimized (no warmup needed)');
                    this.updateProgress(100);  // 6/6 steps - complete
                    
                    // Show GPU info if available
                    if (quickData.status?.gpu_info) {
                        const gpu = quickData.status.gpu_info;
                        console.log(`GPU: ${gpu.name} | Memory: ${gpu.memory_free}MB free`);
                    }
                    return;
                }
            }
            
            // Create VLM details container for sub-steps
            this.showVLMDetails();
            
            // Trigger VLM warmup
            this.updateStepStatus('vlm', 'active', 'Initializing VLM Engine...');
            
            const warmupResponse = await fetch('http://localhost:8000/api/vlm/warmup', {
                method: 'POST',
                mode: 'cors',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (warmupResponse.ok) {
                const warmupData = await warmupResponse.json();
                console.log('VLM warmup result:', warmupData);
                
                if (warmupData.success) {
                    // Show detailed progress through warmup steps
                    const progressUpdates = warmupData.progress_updates || [];
                    let lastProgress = 80;
                    
                    for (const update of progressUpdates) {
                        // Parse the message to extract step type
                        const message = update.message;
                        let stepClass = 'vlm-step';
                        let icon = '⚙️';
                        
                        if (message.includes('GPU') || message.includes('CUDA')) {
                            stepClass = 'vlm-step-gpu';
                            icon = '🎮';
                        } else if (message.includes('Loading') || message.includes('model')) {
                            stepClass = 'vlm-step-model';
                            icon = '📥';
                        } else if (message.includes('Memory') || message.includes('Allocating')) {
                            stepClass = 'vlm-step-memory';
                            icon = '💾';
                        } else if (message.includes('Compiling') || message.includes('optimization')) {
                            stepClass = 'vlm-step-compile';
                            icon = '⚡';
                        } else if (message.includes('Warmup') || message.includes('inference')) {
                            stepClass = 'vlm-step-warmup';
                            icon = '🔥';
                        } else if (message.includes('complete') || message.includes('ready')) {
                            stepClass = 'vlm-step-complete';
                            icon = '✨';
                        }
                        
                        // Update main status
                        this.updateStepStatus('vlm', 'active', message);
                        
                        // Add sub-step to details
                        this.addVLMSubStep(icon, message, stepClass);
                        
                        // Update overall progress based on VLM progress
                        const vlmProgress = update.progress || 0;
                        const overallProgress = 84 + Math.floor(vlmProgress * 0.16); // 84-100% range
                        if (overallProgress > lastProgress) {
                            this.updateProgress(overallProgress);
                            lastProgress = overallProgress;
                        }
                        
                        await this.delay(300); // Slightly longer delay for visibility
                    }
                    
                    // Extract performance metrics
                    const result = warmupData.result || {};
                    const duration = result.duration;
                    const gpuInfo = result.gpu_info || {};
                    const warmupResults = result.warmup_results || {};
                    
                    let statusText = '✅ VLM Engine Optimized';
                    if (duration) {
                        statusText += ` (${duration.toFixed(1)}s)`;
                    }
                    if (warmupResults.average_time_ms) {
                        statusText += ` | Avg: ${Math.round(warmupResults.average_time_ms)}ms`;
                    }
                    
                    await this.delay(500);
                    this.updateStepStatus('vlm', 'completed', statusText);
                    this.updateProgress(100);  // 6/6 steps - complete
                    
                    // Show final GPU stats
                    if (gpuInfo.name) {
                        console.log(`🎮 GPU: ${gpuInfo.name}`);
                        console.log(`💾 Memory: ${gpuInfo.memory_free}MB free / ${gpuInfo.memory_total}MB total`);
                    }
                    
                    console.log('🚀 VLM initialization completed with single warmup!');
                    
                    // Hide VLM details after success
                    setTimeout(() => this.hideVLMDetails(), 2000);
                    
                } else {
                    // Warmup failed but not critical - continue
                    console.warn('VLM warmup failed:', warmupData);
                    this.updateStepStatus('vlm', 'completed', 'Cold inference mode');
                    this.updateProgress(100);  // 6/6 steps - complete
                    this.hideVLMDetails();
                }
            } else {
                // API call failed but not critical
                console.warn('VLM warmup API failed');
                this.updateStepStatus('vlm', 'completed', 'Cold inference mode');
                this.updateProgress(100);  // 6/6 steps - complete
                this.hideVLMDetails();
            }
            
        } catch (error) {
            // VLM warmup failed but not critical for overall initialization
            console.warn('VLM initialization failed, will use cold inference:', error);
            this.updateStepStatus('vlm', 'completed', 'Using cold inference');
            this.updateProgress(100);  // 6/6 steps - complete
            this.hideVLMDetails();
        }
    }
    
    /**
     * Show VLM initialization details container
     */
    showVLMDetails() {
        // Check if details container already exists
        let detailsContainer = document.getElementById('vlmDetails');
        if (detailsContainer) {
            detailsContainer.innerHTML = ''; // Clear existing content
            return;
        }
        
        // Create details container
        const vlmStep = document.getElementById('step-vlm');
        if (vlmStep) {
            detailsContainer = document.createElement('div');
            detailsContainer.id = 'vlmDetails';
            detailsContainer.className = 'vlm-details';
            detailsContainer.innerHTML = `
                <div class="vlm-details-header">VLM Initialization Steps:</div>
                <div class="vlm-substeps" id="vlmSubsteps"></div>
            `;
            
            // Add some CSS for the details
            if (!document.getElementById('vlmDetailsStyle')) {
                const style = document.createElement('style');
                style.id = 'vlmDetailsStyle';
                style.textContent = `
                    .vlm-details {
                        margin-top: 10px;
                        margin-left: 40px;
                        padding: 10px;
                        background: rgba(0, 0, 0, 0.2);
                        border-radius: 8px;
                        font-size: 0.9em;
                    }
                    .vlm-details-header {
                        color: #64b5f6;
                        margin-bottom: 8px;
                        font-weight: 500;
                    }
                    .vlm-substeps {
                        display: flex;
                        flex-direction: column;
                        gap: 4px;
                    }
                    .vlm-substep {
                        display: flex;
                        align-items: center;
                        gap: 8px;
                        padding: 4px 8px;
                        background: rgba(0, 0, 0, 0.3);
                        border-radius: 4px;
                        animation: slideIn 0.3s ease;
                    }
                    .vlm-substep-icon {
                        font-size: 1.1em;
                    }
                    .vlm-substep-text {
                        flex: 1;
                        color: #e0e0e0;
                    }
                    .vlm-step-gpu { border-left: 3px solid #4caf50; }
                    .vlm-step-model { border-left: 3px solid #2196f3; }
                    .vlm-step-memory { border-left: 3px solid #ff9800; }
                    .vlm-step-compile { border-left: 3px solid #9c27b0; }
                    .vlm-step-warmup { border-left: 3px solid #f44336; }
                    .vlm-step-complete { border-left: 3px solid #00e676; }
                    
                    @keyframes slideIn {
                        from {
                            opacity: 0;
                            transform: translateX(-20px);
                        }
                        to {
                            opacity: 1;
                            transform: translateX(0);
                        }
                    }
                `;
                document.head.appendChild(style);
            }
            
            vlmStep.appendChild(detailsContainer);
        }
    }
    
    /**
     * Add a sub-step to VLM details
     */
    addVLMSubStep(icon, text, className = 'vlm-step') {
        const substepsContainer = document.getElementById('vlmSubsteps');
        if (substepsContainer) {
            const substep = document.createElement('div');
            substep.className = `vlm-substep ${className}`;
            substep.innerHTML = `
                <span class="vlm-substep-icon">${icon}</span>
                <span class="vlm-substep-text">${text}</span>
            `;
            substepsContainer.appendChild(substep);
            
            // Limit to last 5 substeps for visibility
            const substeps = substepsContainer.querySelectorAll('.vlm-substep');
            if (substeps.length > 5) {
                substeps[0].remove();
            }
        }
    }
    
    /**
     * Hide VLM details container
     */
    hideVLMDetails() {
        const detailsContainer = document.getElementById('vlmDetails');
        if (detailsContainer) {
            detailsContainer.style.opacity = '0';
            setTimeout(() => {
                if (detailsContainer.parentNode) {
                    detailsContainer.parentNode.removeChild(detailsContainer);
                }
            }, 500);
        }
    }

    /**
     * Wait for WebRTC connection to be established
     */
    async waitForWebRTC() {
        return new Promise((resolve, reject) => {
            let attempts = 0;
            const maxAttempts = 100; // 10 seconds timeout
            
            const checkInterval = setInterval(async () => {
                if (this.webrtcClient && this.webrtcClient.pc) {
                    const state = this.webrtcClient.pc.connectionState;
                    console.log('WebRTC connection state:', state);
                    
                    // Check if connection is established
                    if (state === 'connected') {
                        // Also wait for data channel to be ready
                        console.log('WebRTC connected, waiting for data channel...');
                        const dataChannelReady = await this.webrtcClient.waitForDataChannel(3000);
                        
                        if (dataChannelReady) {
                            console.log('Data channel is ready!');
                            clearInterval(checkInterval);
                            resolve();
                        } else {
                            console.warn('Data channel not ready, but proceeding...');
                            clearInterval(checkInterval);
                            resolve();
                        }
                    } else if (state === 'failed' || state === 'closed') {
                        clearInterval(checkInterval);
                        reject(new Error(`WebRTC connection ${state}`));
                    }
                }
                
                // Also check if we have a video stream, which indicates success
                if (this.webrtcClient && this.webrtcClient.remoteStream) {
                    console.log('WebRTC stream detected');
                    // Still wait a bit for data channel
                    if (!this.webrtcClient.isDataChannelReady()) {
                        await this.webrtcClient.waitForDataChannel(1000);
                    }
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