/**
 * Stream Viewer Component
 * Manages the video stream display and WebRTC connection
 */
class StreamViewer {
    constructor() {
        this.videoElement = document.getElementById('videoPreview');
        this.overlayElement = document.getElementById('videoOverlay');
        this.stream = null;
        this.frameCount = 0;
        this.lastFpsTime = Date.now();
        this.fps = 0;
    }

    /**
     * Set the video stream
     */
    setStream(stream) {
        console.log('Setting video stream');
        this.stream = stream;
        
        if (this.videoElement) {
            this.videoElement.srcObject = stream;
            
            // Configure for low-latency playback
            this.videoElement.muted = true;
            this.videoElement.playsInline = true;
            this.videoElement.disablePictureInPicture = true;
            
            // Set buffering to minimum for real-time streaming
            if ('buffered' in this.videoElement) {
                // Try to minimize buffering for real-time streaming
                this.videoElement.currentTime = this.videoElement.buffered.length > 0 ? 
                    this.videoElement.buffered.end(this.videoElement.buffered.length - 1) : 0;
            }
            
            this.hideOverlay();
            this.startFpsCounter();
        }
    }

    /**
     * Clear the video stream
     */
    clearStream() {
        if (this.videoElement) {
            this.videoElement.srcObject = null;
        }
        this.stream = null;
        this.showOverlay('No video stream');
        this.stopFpsCounter();
    }

    /**
     * Show overlay message
     */
    showOverlay(message) {
        if (this.overlayElement) {
            const messageElement = this.overlayElement.querySelector('.overlay-message');
            if (messageElement) {
                messageElement.textContent = message;
            }
            this.overlayElement.style.display = 'flex';
        }
    }

    /**
     * Hide overlay
     */
    hideOverlay() {
        if (this.overlayElement) {
            this.overlayElement.style.display = 'none';
        }
    }

    /**
     * Capture current frame
     */
    captureFrame() {
        if (!this.videoElement || !this.stream) {
            return null;
        }

        const canvas = document.createElement('canvas');
        canvas.width = this.videoElement.videoWidth;
        canvas.height = this.videoElement.videoHeight;
        
        const ctx = canvas.getContext('2d');
        ctx.drawImage(this.videoElement, 0, 0);
        
        return canvas.toDataURL('image/jpeg');
    }

    /**
     * Start FPS counter
     */
    startFpsCounter() {
        this.fpsInterval = setInterval(() => {
            const now = Date.now();
            const delta = now - this.lastFpsTime;
            
            if (delta >= 1000) {
                this.fps = Math.round((this.frameCount * 1000) / delta);
                this.updateFpsDisplay();
                
                this.frameCount = 0;
                this.lastFpsTime = now;
            }
            
            this.frameCount++;
        }, 1000 / 30); // Assuming 30 fps
    }

    /**
     * Stop FPS counter
     */
    stopFpsCounter() {
        if (this.fpsInterval) {
            clearInterval(this.fpsInterval);
            this.fpsInterval = null;
        }
        this.fps = 0;
        this.updateFpsDisplay();
    }

    /**
     * Update FPS display
     */
    updateFpsDisplay() {
        // Update old fps counter if exists
        const fpsElement = document.getElementById('fpsCounter');
        if (fpsElement) {
            fpsElement.textContent = this.fps;
        }
        
        // Update new streaming FPS display
        const streamingFpsElement = document.getElementById('streamingFPS');
        if (streamingFpsElement) {
            streamingFpsElement.textContent = this.fps;
            
            // Apply color coding based on FPS performance
            streamingFpsElement.className = 'stat-value';
            if (this.fps >= 25) {
                streamingFpsElement.classList.add('fps-good');
            } else if (this.fps >= 15) {
                streamingFpsElement.classList.add('fps-medium');
            } else if (this.fps > 0) {
                streamingFpsElement.classList.add('fps-poor');
            }
        }
    }

    /**
     * Check if stream is active
     */
    isStreamActive() {
        return this.stream && this.stream.active;
    }
}