/**
 * Video Display Component
 */
class VideoDisplay {
    constructor() {
        // Support both remoteVideo (simple) and videoPreview (professional) IDs
        this.videoElement = document.getElementById('videoPreview') || document.getElementById('remoteVideo');
        this.overlayElement = document.getElementById('videoOverlay');
        this.fpsElement = document.getElementById('fpsValue') || document.getElementById('fpsDisplay');
        this.fpsUpdateInterval = null;
        this.frameCount = 0;
        this.lastFpsUpdate = Date.now();
        this.webrtcClient = null;
        this.tapToFocusEnabled = true;

        // Setup tap-to-focus
        this.setupTapToFocus();
    }

    /**
     * Setup tap-to-focus click handler
     */
    setupTapToFocus() {
        if (!this.videoElement) {
            console.warn('VideoDisplay: No video element found for tap-to-focus');
            return;
        }

        this.videoElement.addEventListener('click', (event) => {
            if (!this.tapToFocusEnabled) {
                return;
            }

            if (!this.webrtcClient) {
                console.warn('No WebRTC client set for tap-to-focus');
                return;
            }

            const rect = this.videoElement.getBoundingClientRect();
            const x = (event.clientX - rect.left) / rect.width;
            const y = (event.clientY - rect.top) / rect.height;

            // Send normalized coordinates (0-1)
            this.webrtcClient.tapToFocus(x, y);

            // Visual feedback - show focus indicator
            this.showFocusIndicator(event.clientX - rect.left, event.clientY - rect.top);
        });

        // Add cursor pointer to indicate clickable
        this.videoElement.style.cursor = 'crosshair';
    }

    /**
     * Show visual focus indicator at tap location
     */
    showFocusIndicator(x, y) {
        // Remove existing indicator
        const existing = document.querySelector('.focus-indicator');
        if (existing) existing.remove();

        // Create new indicator
        const indicator = document.createElement('div');
        indicator.className = 'focus-indicator';
        indicator.style.cssText = `
            position: absolute;
            left: ${x}px;
            top: ${y}px;
            width: 60px;
            height: 60px;
            margin-left: -30px;
            margin-top: -30px;
            border: 2px solid #00ff00;
            border-radius: 50%;
            pointer-events: none;
            animation: focus-pulse 0.6s ease-out;
        `;

        const container = this.videoElement.parentElement;
        container.style.position = 'relative';
        container.appendChild(indicator);

        // Remove after animation
        setTimeout(() => indicator.remove(), 600);
    }

    /**
     * Set WebRTC client for tap-to-focus
     */
    setWebRTCClient(client) {
        this.webrtcClient = client;
    }

    /**
     * Enable/disable tap-to-focus
     */
    setTapToFocusEnabled(enabled) {
        this.tapToFocusEnabled = enabled;
        if (this.videoElement) {
            this.videoElement.style.cursor = enabled ? 'crosshair' : 'default';
        }
    }

    /**
     * Set video stream
     */
    setStream(stream) {
        if (this.videoElement && stream) {
            this.videoElement.srcObject = stream;
            this.hideOverlay();
            this.startFpsMonitoring();
        }
    }

    /**
     * Show overlay with message
     */
    showOverlay(message = 'Connecting to camera...') {
        if (this.overlayElement) {
            this.overlayElement.classList.remove('hidden');
            const messageElement = this.overlayElement.querySelector('.overlay-message p');
            if (messageElement) {
                messageElement.textContent = message;
            }
        }
    }

    /**
     * Hide overlay
     */
    hideOverlay() {
        if (this.overlayElement) {
            this.overlayElement.classList.add('hidden');
        }
    }

    /**
     * Start FPS monitoring
     */
    startFpsMonitoring() {
        this.stopFpsMonitoring();
        
        // Monitor video frame updates
        const countFrame = () => {
            if (this.videoElement && !this.videoElement.paused && !this.videoElement.ended) {
                this.frameCount++;
                requestAnimationFrame(countFrame);
            }
        };
        requestAnimationFrame(countFrame);

        // Update FPS display every second
        this.fpsUpdateInterval = setInterval(() => {
            const now = Date.now();
            const elapsed = (now - this.lastFpsUpdate) / 1000;
            const fps = Math.round(this.frameCount / elapsed);
            
            this.updateFps(fps);
            
            this.frameCount = 0;
            this.lastFpsUpdate = now;
        }, 1000);
    }

    /**
     * Stop FPS monitoring
     */
    stopFpsMonitoring() {
        if (this.fpsUpdateInterval) {
            clearInterval(this.fpsUpdateInterval);
            this.fpsUpdateInterval = null;
        }
        this.frameCount = 0;
    }

    /**
     * Update FPS display
     */
    updateFps(fps) {
        if (this.fpsElement) {
            this.fpsElement.textContent = fps;
            
            // Color code based on FPS
            if (fps >= 25) {
                this.fpsElement.style.color = '#4caf50';
            } else if (fps >= 15) {
                this.fpsElement.style.color = '#ff9800';
            } else {
                this.fpsElement.style.color = '#f44336';
            }
        }
    }

    /**
     * Clear video stream
     */
    clearStream() {
        if (this.videoElement) {
            this.videoElement.srcObject = null;
        }
        this.stopFpsMonitoring();
        this.updateFps(0);
    }

    /**
     * Capture current frame
     */
    captureFrame() {
        if (!this.videoElement || !this.videoElement.srcObject) {
            return null;
        }

        const canvas = document.createElement('canvas');
        canvas.width = this.videoElement.videoWidth;
        canvas.height = this.videoElement.videoHeight;
        
        const ctx = canvas.getContext('2d');
        ctx.drawImage(this.videoElement, 0, 0);
        
        return canvas.toDataURL('image/jpeg', 0.95);
    }
}

// Export for use in other modules
window.VideoDisplay = VideoDisplay;