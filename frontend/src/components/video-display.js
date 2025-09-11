/**
 * Video Display Component
 */
class VideoDisplay {
    constructor() {
        this.videoElement = document.getElementById('remoteVideo');
        this.overlayElement = document.getElementById('videoOverlay');
        this.fpsElement = document.getElementById('fpsValue');
        this.fpsUpdateInterval = null;
        this.frameCount = 0;
        this.lastFpsUpdate = Date.now();
    }

    /**
     * Set video stream
     */
    setStream(stream) {
        if (this.videoElement && stream) {
            this.videoElement.srcObject = stream;
            this.hideOverlay();
            this.startFpsMonitoring();
            console.log('Video stream set');
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