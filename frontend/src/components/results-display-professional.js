/**
 * Professional Results Display Component
 * Manages classification results with persistence and history
 */
class ResultsDisplayProfessional {
    constructor() {
        this.currentResults = null;
        this.sessionHistory = [];
        this.allSessions = this.loadFromLocalStorage() || [];
        this.csvExporter = new CSVExporter();
        this.initializeElements();

        // Restore last current result on page load
        this.restoreCurrentResult();
    }

    /**
     * Initialize DOM elements
     */
    initializeElements() {
        this.summaryContainer = document.getElementById('resultsSummary');
        this.historyList = document.getElementById('historyList');
        this.tableContainer = document.getElementById('resultsTableContainer');
        this.tableBody = document.getElementById('resultsTableBody');
        
        // Export buttons
        this.csvBtn = document.getElementById('exportCsvBtn');
        this.jsonBtn = document.getElementById('exportJsonBtn');
        
        // Clear buttons
        this.clearHistoryBtn = document.getElementById('clearHistoryBtn');
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Load existing history
        this.displayHistory();
    }

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        if (this.csvBtn) {
            this.csvBtn.addEventListener('click', () => this.exportToCSV());
        }
        
        if (this.jsonBtn) {
            this.jsonBtn.addEventListener('click', () => this.exportToJSON());
        }
        
        if (this.clearHistoryBtn) {
            this.clearHistoryBtn.addEventListener('click', () => this.clearAllHistory());
        }
    }

    /**
     * Reset display for new session
     */
    reset() {
        this.currentResults = null;
        this.showNoResults();
        this.disableExportButtons();

        // Clear saved current result when starting new classification
        try {
            localStorage.removeItem('relo_current_result');
        } catch (e) {
            console.error('Failed to clear current result:', e);
        }
    }

    /**
     * Show no results message
     */
    showNoResults() {
        if (this.summaryContainer) {
            this.summaryContainer.innerHTML = `
                <div class="no-results">
                    <span class="no-results-icon">📊</span>
                    <p>No classification results yet</p>
                    <p class="hint-text">Results will appear here after classification</p>
                </div>
            `;
        }
        
        if (this.tableContainer) {
            this.tableContainer.style.display = 'none';
        }
    }

    /**
     * Display final results
     */
    displayFinalResults(results) {
        if (!results) return;
        
        this.currentResults = results;
        
        // Add timestamp and session ID
        results.timestamp = new Date().toISOString();
        results.session_id = results.session_id || this.generateSessionId();
        
        // Display in summary
        this.displayResultsSummary(results);
        
        // Display in table
        this.displayResultsTable(results);
        
        // Add to history
        this.addToHistory(results);
        
        // Add to CSV exporter
        this.csvExporter.addResult(results);
        
        // Enable export buttons
        this.enableExportButtons();

        // Save to localStorage for persistence
        this.saveToLocalStorage();

        // Save current result separately for restoration
        this.saveCurrentResult(results);
    }

    /**
     * Display results summary card
     */
    displayResultsSummary(results) {
        if (!this.summaryContainer) return;
        
        const confidence = results.overall_confidence || results.confidence || 0;
        const confidencePercent = Math.round(confidence * 100);
        
        this.summaryContainer.innerHTML = `
            <div class="final-result-card">
                <div class="final-result-header">
                    <h3 class="final-result-title">Classification Complete</h3>
                    <div class="confidence-badge">
                        <span>Confidence:</span>
                        <strong>${confidencePercent}%</strong>
                    </div>
                </div>
                
                <div class="result-grid">
                    ${this.createResultItem('Item Type', results.item_type || 'Unknown')}
                    ${this.createResultItem('Brand', results.brand || 'Unknown')}
                    ${this.createResultItem('Color', results.color || 'Unknown')}
                    ${this.createResultItem('Size', results.size || 'Unknown')}
                    ${this.createResultItem('Pattern', results.pattern || 'Unknown')}
                    ${this.createResultItem('Neckline', results.neckline || 'Unknown')}
                    ${this.createResultItem('Sleeve Type', results.sleeve_type || 'Unknown')}
                    ${this.createResultItem('Closure Type', results.closure_type || 'Unknown')}
                </div>
                
                ${results.is_damaged ? `
                    <div class="damage-section">
                        <h4>Damage Assessment</h4>
                        <div class="damage-indicator damage-found">
                            ⚠️ Damage Detected
                        </div>
                        ${results.damage_type ? `
                            <div class="damage-details">
                                <div>• Type: ${results.damage_type}</div>
                                ${results.damage_severity ? `<div>• Severity: ${results.damage_severity}</div>` : ''}
                            </div>
                        ` : ''}
                    </div>
                ` : `
                    <div class="damage-section">
                        <h4>Damage Assessment</h4>
                        <div class="damage-indicator no-damage">
                            ✅ No Damage Found
                        </div>
                    </div>
                `}
                
                <div class="return-status">
                    <div class="status-badge ${results.return_valid ? 'valid' : 'invalid'}">
                        ${results.return_valid ? '✅ Return Valid' : '❌ Return Invalid'}
                    </div>
                    ${results.reason ? `<p class="return-reason">${results.reason}</p>` : ''}
                </div>
                
                <div class="result-metadata">
                    <span class="meta-item">
                        <strong>Session:</strong> ${results.session_id}
                    </span>
                    <span class="meta-item">
                        <strong>Time:</strong> ${new Date(results.timestamp).toLocaleString()}
                    </span>
                    <span class="meta-item">
                        <strong>Processing:</strong> ${results.processing_time || '0'}s
                    </span>
                </div>
            </div>
        `;
    }

    /**
     * Create result item HTML
     */
    createResultItem(label, value) {
        return `
            <div class="result-item">
                <span class="result-key">${label}:</span>
                <span class="result-value">${value}</span>
            </div>
        `;
    }

    /**
     * Display results in table format
     */
    displayResultsTable(results) {
        if (!this.tableContainer || !this.tableBody) return;
        
        // Clear existing rows
        this.tableBody.innerHTML = '';
        
        // Create rows for each attribute (using normalized keys)
        const attributes = [
            { key: 'item_type', label: 'Item Type' },
            { key: 'color', label: 'Color' },
            { key: 'pattern', label: 'Pattern' },
            { key: 'brand', label: 'Brand' },
            { key: 'size', label: 'Size' },
            { key: 'neckline', label: 'Neckline' },
            { key: 'sleeve_type', label: 'Sleeve Type' },
            { key: 'closure_type', label: 'Closure Type' },
            { key: 'is_damaged', label: 'Damage Detected' },
            { key: 'damage_type', label: 'Damage Type' },
            { key: 'damage_severity', label: 'Damage Severity' }
        ];
        
        attributes.forEach(attr => {
            const value = results[attr.key];
            if (value !== undefined && value !== null) {
                const row = document.createElement('tr');
                
                // Format value
                let displayValue = value;
                let confidence = '';
                
                if (typeof value === 'boolean') {
                    displayValue = value ? 'Yes' : 'No';
                }
                
                // Get confidence if available
                const confidenceKey = `${attr.key}_confidence`;
                if (results[confidenceKey]) {
                    confidence = `${Math.round(results[confidenceKey] * 100)}%`;
                }
                
                row.innerHTML = `
                    <td>${attr.label}</td>
                    <td><strong>${displayValue}</strong></td>
                    <td>${confidence || '-'}</td>
                `;
                
                this.tableBody.insertBefore(row, this.tableBody.firstChild);
            }
        });
        
        // Show table container
        this.tableContainer.style.display = 'block';
    }

    /**
     * Add results to history
     */
    addToHistory(results) {
        // Add to session history
        this.sessionHistory.push(results);
        
        // Add to all sessions
        this.allSessions.push(results);
        
        // Display in history list
        this.addHistoryItem(results);
        
        // Save to localStorage
        this.saveToLocalStorage();
    }

    /**
     * Display history list
     */
    displayHistory() {
        if (!this.historyList) return;
        
        // Clear existing
        this.historyList.innerHTML = '';
        
        // Display all sessions
        this.allSessions.forEach(session => {
            this.addHistoryItem(session);
        });
    }

    /**
     * Add history item to list
     */
    addHistoryItem(results) {
        if (!this.historyList) return;
        
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        historyItem.dataset.sessionId = results.session_id;
        
        const timestamp = new Date(results.timestamp).toLocaleString();
        
        historyItem.innerHTML = `
            <div class="history-header">
                <strong>${results.item_type || 'Unknown Item'}</strong>
                <span class="history-timestamp">${timestamp}</span>
            </div>
            <div class="history-details">
                <span>${results.brand || 'N/A'}</span> • 
                <span>${results.color || 'N/A'}</span> • 
                <span>${results.condition || 'N/A'}</span>
            </div>
            <div class="history-status">
                ${results.return_valid ? '✅ Valid' : '❌ Invalid'}
            </div>
        `;
        
        // Add click handler to view details
        historyItem.addEventListener('click', () => {
            this.displayResultsSummary(results);
            this.displayResultsTable(results);
        });
        
        // Add to top of list
        this.historyList.insertBefore(historyItem, this.historyList.firstChild);
    }

    /**
     * Export results to CSV
     */
    exportToCSV() {
        this.csvExporter.exportCurrentSession();
    }

    /**
     * Export results to JSON
     */
    exportToJSON() {
        this.csvExporter.exportJSON(this.currentResults ? [this.currentResults] : this.sessionHistory);
    }

    /**
     * Clear all history
     */
    clearAllHistory() {
        if (confirm('Are you sure you want to clear all history? This cannot be undone.')) {
            this.sessionHistory = [];
            this.allSessions = [];
            this.csvExporter.clearAllHistory();
            
            if (this.historyList) {
                this.historyList.innerHTML = '';
            }
            
            this.clearLocalStorage();
        }
    }

    /**
     * Enable export buttons
     */
    enableExportButtons() {
        if (this.csvBtn) this.csvBtn.disabled = false;
        if (this.jsonBtn) this.jsonBtn.disabled = false;
    }

    /**
     * Disable export buttons
     */
    disableExportButtons() {
        if (this.csvBtn) this.csvBtn.disabled = true;
        if (this.jsonBtn) this.jsonBtn.disabled = true;
    }

    /**
     * Generate session ID
     */
    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substring(2, 11);
    }

    /**
     * Save to localStorage
     */
    saveToLocalStorage() {
        try {
            localStorage.setItem('relo_classification_history', JSON.stringify(this.allSessions));
        } catch (e) {
            console.error('Failed to save to localStorage:', e);
        }
    }

    /**
     * Load from localStorage
     */
    loadFromLocalStorage() {
        try {
            const stored = localStorage.getItem('relo_classification_history');
            return stored ? JSON.parse(stored) : [];
        } catch (e) {
            console.error('Failed to load from localStorage:', e);
            return [];
        }
    }

    /**
     * Clear localStorage
     */
    clearLocalStorage() {
        try {
            localStorage.removeItem('relo_classification_history');
            localStorage.removeItem('relo_current_result');
        } catch (e) {
            console.error('Failed to clear localStorage:', e);
        }
    }

    /**
     * Save current result for restoration after refresh
     */
    saveCurrentResult(results) {
        try {
            localStorage.setItem('relo_current_result', JSON.stringify(results));
        } catch (e) {
            console.error('Failed to save current result:', e);
        }
    }

    /**
     * Restore current result from localStorage on page load
     */
    restoreCurrentResult() {
        try {
            const stored = localStorage.getItem('relo_current_result');
            if (stored) {
                const results = JSON.parse(stored);
                this.currentResults = results;

                // Display restored results
                this.displayResultsSummary(results);
                this.displayResultsTable(results);

                // Add to CSV exporter for export functionality
                this.csvExporter.addResult(results);

                // Enable export buttons
                this.enableExportButtons();

                console.log('✅ Restored classification result from previous session');
            }
        } catch (e) {
            console.error('Failed to restore current result:', e);
        }
    }

    /**
     * Get current results
     */
    getCurrentResults() {
        return this.currentResults;
    }

    /**
     * Get all results
     */
    getAllResults() {
        return this.allSessions;
    }
}

// Export for use in other modules
window.ResultsDisplayProfessional = ResultsDisplayProfessional;