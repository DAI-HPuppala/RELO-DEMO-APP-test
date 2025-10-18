/**
 * CleanupManager - Handles aggressive cleanup on page refresh/close
 *
 * Triggers backend cleanup via /api/session/force-reset before page unload.
 * Works in tandem with WebSocket disconnect cleanup for redundant safety.
 */
class CleanupManager {
    constructor() {
        this.apiBaseUrl = window.location.origin;
        this.cleanupTriggered = false;
        this.setupListeners();
        console.log('[CleanupManager] Initialized - ready to handle page unload');
    }

    /**
     * Setup beforeunload listener
     */
    setupListeners() {
        window.addEventListener('beforeunload', (event) => {
            this.triggerCleanup();
        });
    }

    /**
     * Trigger backend cleanup using sendBeacon (reliable during page unload)
     */
    triggerCleanup() {
        if (this.cleanupTriggered) {
            return; // Only trigger once
        }

        this.cleanupTriggered = true;

        const cleanupUrl = `${this.apiBaseUrl}/api/session/force-reset`;

        // Use sendBeacon for reliability during page unload
        // Unlike fetch(), sendBeacon() is guaranteed to complete even as page unloads
        const success = navigator.sendBeacon(cleanupUrl, JSON.stringify({}));

        if (success) {
            console.log('[CleanupManager] Backend cleanup signal sent successfully');
        } else {
            console.warn('[CleanupManager] Failed to send cleanup signal (browser may have blocked)');
        }
    }
}

// Initialize immediately when script loads
window.cleanupManager = new CleanupManager();
