/**
 * Barcode Retry Button Component
 * Allows retry of barcode detection after failure or timeout
 */
class BarcodeRetryButton {
    constructor() {
        this.button = null;
        this.isVisible = false;
        this.retryCallback = null;

        this.createButton();
        this.setupEventListeners();
    }

    /**
     * Create retry button HTML
     */
    createButton() {
        // Create button element
        this.button = document.createElement('button');
        this.button.id = 'barcodeRetryBtn';
        this.button.className = 'barcode-retry-btn';
        this.button.style.display = 'none';
        this.button.setAttribute('data-tooltip', 'Retry barcode detection');

        this.button.innerHTML = `
            <span class="retry-icon">↻</span>
            <span class="retry-text">Retry Barcode</span>
        `;

        // Add to video container or control panel
        const videoContainer = document.getElementById('videoContainer');
        if (videoContainer) {
            videoContainer.appendChild(this.button);
        } else {
            document.body.appendChild(this.button);
        }
    }

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        if (this.button) {
            this.button.addEventListener('click', () => this.handleRetry());
        }
    }

    /**
     * Handle retry button click
     */
    async handleRetry() {
        console.log('Barcode retry button clicked');

        // Disable button during retry
        this.disable();

        try {
            // Call backend API to retry initialization
            const response = await fetch('/api/barcode/retry-init', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const result = await response.json();

            if (result.success) {
                // Hide button on successful retry
                this.hide();

                // Show success notification
                this.showNotification('Barcode detection reinitialized', 'success');

                // Trigger callback if set
                if (this.retryCallback) {
                    this.retryCallback();
                }
            } else {
                // Re-enable button on failure
                this.enable();

                // Show error notification
                this.showNotification('Failed to reinitialize. Please try again.', 'error');
            }
        } catch (error) {
            console.error('Failed to retry barcode detection:', error);

            // Re-enable button on error
            this.enable();

            // Show error notification
            this.showNotification('Network error. Please check your connection.', 'error');
        }
    }

    /**
     * Show retry button
     */
    show() {
        if (this.button) {
            this.button.style.display = 'flex';
            this.isVisible = true;
        }
    }

    /**
     * Hide retry button
     */
    hide() {
        if (this.button) {
            this.button.style.display = 'none';
            this.isVisible = false;
        }
    }

    /**
     * Enable button
     */
    enable() {
        if (this.button) {
            this.button.disabled = false;
            this.button.classList.remove('disabled');
        }
    }

    /**
     * Disable button
     */
    disable() {
        if (this.button) {
            this.button.disabled = true;
            this.button.classList.add('disabled');
        }
    }

    /**
     * Set retry callback
     */
    setRetryCallback(callback) {
        this.retryCallback = callback;
    }

    /**
     * Show notification
     */
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `barcode-notification ${type}`;
        notification.textContent = message;
        document.body.appendChild(notification);

        setTimeout(() => {
            notification.classList.add('show');
        }, 10);

        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    /**
     * Check if button is visible
     */
    isButtonVisible() {
        return this.isVisible;
    }
}

// Export for use in other modules
window.BarcodeRetryButton = BarcodeRetryButton;
