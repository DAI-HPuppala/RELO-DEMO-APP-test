/**
 * Barcode Manual Entry Modal Component
 * Allows manual barcode entry when automatic detection fails or times out
 */
class BarcodeManualEntryModal {
    constructor() {
        this.modal = null;
        this.backdrop = null;
        this.input = null;
        this.submitBtn = null;
        this.cancelBtn = null;
        this.skipBtn = null;
        this.errorMessage = null;

        this.isOpen = false;
        this.resolveCallback = null;
        this.rejectCallback = null;

        this.createModal();
        this.setupEventListeners();
    }

    /**
     * Create modal HTML structure
     */
    createModal() {
        // Create backdrop
        this.backdrop = document.createElement('div');
        this.backdrop.className = 'barcode-modal-backdrop';
        this.backdrop.style.display = 'none';

        // Create modal container
        this.modal = document.createElement('div');
        this.modal.className = 'barcode-modal';

        // Modal content
        this.modal.innerHTML = `
            <div class="barcode-modal-header">
                <h3 class="barcode-modal-title">Manual Barcode Entry</h3>
                <button class="barcode-modal-close" aria-label="Close">×</button>
            </div>
            <div class="barcode-modal-body">
                <p class="barcode-modal-message">
                    Automatic barcode detection failed. Please enter the barcode manually or skip this step.
                </p>
                <div class="barcode-input-group">
                    <label for="barcodeInput" class="barcode-input-label">Barcode:</label>
                    <input
                        type="text"
                        id="barcodeInput"
                        class="barcode-input"
                        placeholder="Enter barcode value..."
                        autocomplete="off"
                    />
                    <div class="barcode-error-message" style="display:none;">
                        Please enter a valid barcode
                    </div>
                </div>
            </div>
            <div class="barcode-modal-footer">
                <button class="barcode-modal-btn barcode-btn-skip">Skip Barcode</button>
                <button class="barcode-modal-btn barcode-btn-cancel">Cancel</button>
                <button class="barcode-modal-btn barcode-btn-submit">Submit</button>
            </div>
        `;

        this.backdrop.appendChild(this.modal);
        document.body.appendChild(this.backdrop);

        // Get references to elements
        this.input = this.modal.querySelector('#barcodeInput');
        this.submitBtn = this.modal.querySelector('.barcode-btn-submit');
        this.cancelBtn = this.modal.querySelector('.barcode-btn-cancel');
        this.skipBtn = this.modal.querySelector('.barcode-btn-skip');
        this.closeBtn = this.modal.querySelector('.barcode-modal-close');
        this.errorMessage = this.modal.querySelector('.barcode-error-message');
    }

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Submit button
        if (this.submitBtn) {
            this.submitBtn.addEventListener('click', () => this.handleSubmit());
        }

        // Cancel button
        if (this.cancelBtn) {
            this.cancelBtn.addEventListener('click', () => this.handleCancel());
        }

        // Skip button
        if (this.skipBtn) {
            this.skipBtn.addEventListener('click', () => this.handleSkip());
        }

        // Close button (X)
        if (this.closeBtn) {
            this.closeBtn.addEventListener('click', () => this.handleCancel());
        }

        // Enter key to submit
        if (this.input) {
            this.input.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.handleSubmit();
                }
            });

            // Clear error on input
            this.input.addEventListener('input', () => {
                this.hideError();
            });
        }

        // Close on backdrop click
        this.backdrop.addEventListener('click', (e) => {
            if (e.target === this.backdrop) {
                this.handleCancel();
            }
        });

        // Escape key to close
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isOpen) {
                this.handleCancel();
            }
        });
    }

    /**
     * Open modal and return a promise
     */
    open(options = {}) {
        return new Promise((resolve, reject) => {
            this.resolveCallback = resolve;
            this.rejectCallback = reject;

            // Update message if provided
            if (options.message) {
                const messageEl = this.modal.querySelector('.barcode-modal-message');
                if (messageEl) {
                    messageEl.textContent = options.message;
                }
            }

            // Show modal
            this.backdrop.style.display = 'flex';
            this.isOpen = true;

            // Focus input
            setTimeout(() => {
                if (this.input) {
                    this.input.focus();
                }
            }, 100);
        });
    }

    /**
     * Close modal
     */
    close() {
        this.backdrop.style.display = 'none';
        this.isOpen = false;
        this.input.value = '';
        this.hideError();
        this.resolveCallback = null;
        this.rejectCallback = null;
    }

    /**
     * Handle submit
     */
    async handleSubmit() {
        const value = this.input.value.trim();

        // Validate input
        if (!value) {
            this.showError('Please enter a barcode value');
            return;
        }

        if (value.length < 3) {
            this.showError('Barcode must be at least 3 characters');
            return;
        }

        try {
            // Submit to backend
            const sessionId = window.appController?.sessionId;
            if (!sessionId) {
                throw new Error('No active session');
            }

            const response = await fetch('/api/barcode/manual-entry', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: sessionId,
                    barcode_data: value
                })
            });

            const result = await response.json();

            if (result.success) {
                // Resolve promise with entered value
                if (this.resolveCallback) {
                    this.resolveCallback({
                        action: 'manual_entry',
                        barcode_data: value,
                        barcode_type: 'MANUAL'
                    });
                }
                this.close();
            } else {
                this.showError('Failed to submit barcode. Please try again.');
            }
        } catch (error) {
            console.error('Failed to submit manual barcode:', error);
            this.showError('Network error. Please check your connection.');
        }
    }

    /**
     * Handle cancel
     */
    handleCancel() {
        if (this.rejectCallback) {
            this.rejectCallback({ action: 'cancel' });
        }
        this.close();
    }

    /**
     * Handle skip
     */
    async handleSkip() {
        try {
            // Skip barcode detection
            const sessionId = window.appController?.sessionId;
            if (!sessionId) {
                throw new Error('No active session');
            }

            const response = await fetch('/api/barcode/skip', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: sessionId
                })
            });

            const result = await response.json();

            if (result.success) {
                // Resolve promise with skip action
                if (this.resolveCallback) {
                    this.resolveCallback({
                        action: 'skip',
                        barcode_data: null,
                        barcode_type: null
                    });
                }
                this.close();
            } else {
                this.showError('Failed to skip barcode. Please try again.');
            }
        } catch (error) {
            console.error('Failed to skip barcode:', error);
            this.showError('Network error. Please check your connection.');
        }
    }

    /**
     * Show error message
     */
    showError(message) {
        if (this.errorMessage) {
            this.errorMessage.textContent = message;
            this.errorMessage.style.display = 'block';
        }

        // Add error class to input
        if (this.input) {
            this.input.classList.add('error');
        }
    }

    /**
     * Hide error message
     */
    hideError() {
        if (this.errorMessage) {
            this.errorMessage.style.display = 'none';
        }

        // Remove error class from input
        if (this.input) {
            this.input.classList.remove('error');
        }
    }

    /**
     * Check if modal is open
     */
    isModalOpen() {
        return this.isOpen;
    }
}

// Export for use in other modules
window.BarcodeManualEntryModal = BarcodeManualEntryModal;
