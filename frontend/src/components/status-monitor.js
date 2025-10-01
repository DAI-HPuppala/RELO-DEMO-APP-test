/**
 * Status Monitor Component
 * Manages status indicators and system monitoring
 */
class StatusMonitor {
    constructor() {
        this.sessionIdElement = document.getElementById('sessionId');
        this.fpsElement = document.getElementById('fpsCounter');
        this.framesElement = document.getElementById('framesCounter');
        this.statusElement = document.getElementById('systemStatus');
        
        this.cameraIndicator = document.getElementById('cameraIndicator');
        this.connectionIndicator = document.getElementById('connectionIndicator');
        this.streamIndicator = document.getElementById('streamIndicator');
        
        this.frameCount = 0;
        this.sessionId = null;
    }

    /**
     * Update session ID
     */
    updateSessionId(sessionId) {
        this.sessionId = sessionId;
        if (this.sessionIdElement) {
            this.sessionIdElement.textContent = sessionId ? sessionId.substring(0, 8) : '-';
        }
    }

    /**
     * Update FPS counter
     */
    updateFps(fps) {
        if (this.fpsElement) {
            this.fpsElement.textContent = fps;
        }
    }

    /**
     * Update frames counter
     */
    updateFrameCount(count) {
        this.frameCount = count;
        if (this.framesElement) {
            this.framesElement.textContent = count;
        }
    }

    /**
     * Increment frame count
     */
    incrementFrameCount() {
        this.frameCount++;
        this.updateFrameCount(this.frameCount);
    }

    /**
     * Update system status
     */
    updateSystemStatus(status) {
        if (this.statusElement) {
            this.statusElement.textContent = status;
            
            // Update status color based on status
            this.statusElement.className = 'status-value';
            if (status.toLowerCase().includes('error')) {
                this.statusElement.classList.add('status-error');
            } else if (status.toLowerCase().includes('processing')) {
                this.statusElement.classList.add('status-processing');
            } else if (status.toLowerCase().includes('ready')) {
                this.statusElement.classList.add('status-ready');
            }
        }
    }

    /**
     * Update camera indicator
     */
    updateCameraStatus(active) {
        if (this.cameraIndicator) {
            const dot = this.cameraIndicator.querySelector('.indicator-dot');
            if (active) {
                this.cameraIndicator.classList.add('active');
                if (dot) dot.classList.add('active');
            } else {
                this.cameraIndicator.classList.remove('active');
                if (dot) dot.classList.remove('active');
            }
        }
    }

    /**
     * Update connection indicator
     */
    updateConnectionStatus(connected) {
        if (this.connectionIndicator) {
            const dot = this.connectionIndicator.querySelector('.indicator-dot');
            if (connected) {
                this.connectionIndicator.classList.add('active');
                if (dot) dot.classList.add('active');
            } else {
                this.connectionIndicator.classList.remove('active');
                if (dot) dot.classList.remove('active');
            }
        }
    }

    /**
     * Update stream indicator
     */
    updateStreamStatus(active) {
        if (this.streamIndicator) {
            const dot = this.streamIndicator.querySelector('.indicator-dot');
            if (active) {
                this.streamIndicator.classList.add('active');
                if (dot) dot.classList.add('active');
            } else {
                this.streamIndicator.classList.remove('active');
                if (dot) dot.classList.remove('active');
            }
        }
    }

    /**
     * Reset all counters and status
     */
    reset() {
        this.frameCount = 0;
        this.sessionId = null;
        
        this.updateSessionId(null);
        this.updateFps(0);
        this.updateFrameCount(0);
        this.updateSystemStatus('Ready');
        
        this.updateCameraStatus(false);
        this.updateStreamStatus(false);
    }

    /**
     * Start monitoring mode
     */
    startMonitoring() {
        this.reset();
        this.updateSystemStatus('Initializing...');
    }

    /**
     * Stop monitoring mode
     */
    stopMonitoring() {
        this.updateSystemStatus('Stopped');
        this.updateCameraStatus(false);
        this.updateStreamStatus(false);
    }

    /**
     * Update from WebRTC stats
     */
    updateFromStats(stats) {
        // Parse WebRTC stats for monitoring
        if (stats.video) {
            if (stats.video.framesPerSecond) {
                this.updateFps(Math.round(stats.video.framesPerSecond));
            }
            if (stats.video.framesReceived) {
                this.updateFrameCount(stats.video.framesReceived);
            }
        }
    }

    /**
     * Show error state
     */
    showError(message) {
        this.updateSystemStatus(`Error: ${message}`);
    }

    /**
     * Show processing state
     */
    showProcessing(agentName) {
        this.updateSystemStatus(`Processing: ${agentName}`);
    }

    /**
     * Show ready state
     */
    showReady() {
        this.updateSystemStatus('Ready');
    }
}