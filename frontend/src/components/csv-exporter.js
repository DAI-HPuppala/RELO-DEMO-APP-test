/**
 * CSV Exporter Component
 * Handles exporting classification results to CSV format
 */
class CSVExporter {
    constructor() {
        this.results = [];
        this.sessionHistory = [];
    }

    /**
     * Add result to export queue
     */
    addResult(result) {
        const timestamp = new Date().toISOString();
        const exportRecord = {
            timestamp: timestamp,
            session_id: result.session_id || 'unknown',
            ...this.flattenObject(result)
        };
        this.results.push(exportRecord);
        this.sessionHistory.push(exportRecord);
    }

    /**
     * Flatten nested objects for CSV export
     */
    flattenObject(obj, prefix = '') {
        const flattened = {};
        
        for (const [key, value] of Object.entries(obj)) {
            const newKey = prefix ? `${prefix}_${key}` : key;
            
            if (value === null || value === undefined) {
                flattened[newKey] = '';
            } else if (typeof value === 'object' && !Array.isArray(value)) {
                Object.assign(flattened, this.flattenObject(value, newKey));
            } else if (Array.isArray(value)) {
                flattened[newKey] = value.join('; ');
            } else {
                flattened[newKey] = value;
            }
        }
        
        return flattened;
    }

    /**
     * Convert results to CSV format
     */
    convertToCSV(data) {
        if (!data || data.length === 0) {
            return '';
        }

        // Get all unique keys from all records
        const allKeys = new Set();
        data.forEach(record => {
            Object.keys(record).forEach(key => allKeys.add(key));
        });

        // Sort keys for consistent column order
        const headers = Array.from(allKeys).sort();
        
        // Create CSV header row
        const csvHeader = headers.map(h => this.escapeCSVValue(h)).join(',');
        
        // Create CSV data rows
        const csvRows = data.map(record => {
            return headers.map(header => {
                const value = record[header] || '';
                return this.escapeCSVValue(value);
            }).join(',');
        });

        return [csvHeader, ...csvRows].join('\n');
    }

    /**
     * Escape CSV values
     */
    escapeCSVValue(value) {
        if (value === null || value === undefined) {
            return '';
        }
        
        const stringValue = String(value);
        
        // Check if value needs escaping
        if (stringValue.includes(',') || stringValue.includes('"') || stringValue.includes('\n')) {
            // Escape quotes by doubling them
            const escaped = stringValue.replace(/"/g, '""');
            return `"${escaped}"`;
        }
        
        return stringValue;
    }

    /**
     * Export current session results to CSV
     */
    exportCurrentSession() {
        if (this.results.length === 0) {
            alert('No results to export');
            return;
        }

        const csv = this.convertToCSV(this.results);
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `relo_classification_${timestamp}.csv`;
        
        this.downloadFile(csv, filename, 'text/csv');
    }

    /**
     * Export all session history to CSV
     */
    exportAllSessions() {
        if (this.sessionHistory.length === 0) {
            alert('No historical data to export');
            return;
        }

        const csv = this.convertToCSV(this.sessionHistory);
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `relo_classification_history_${timestamp}.csv`;
        
        this.downloadFile(csv, filename, 'text/csv');
    }

    /**
     * Export results as JSON
     */
    exportJSON(data = null) {
        const exportData = data || this.results;
        
        if (!exportData || exportData.length === 0) {
            alert('No results to export');
            return;
        }

        const json = JSON.stringify(exportData, null, 2);
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `relo_classification_${timestamp}.json`;
        
        this.downloadFile(json, filename, 'application/json');
    }

    /**
     * Create and trigger file download
     */
    downloadFile(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        URL.revokeObjectURL(url);
    }

    /**
     * Clear current session results
     */
    clearCurrentSession() {
        this.results = [];
    }

    /**
     * Clear all history
     */
    clearAllHistory() {
        this.sessionHistory = [];
        this.results = [];
    }

    /**
     * Get formatted summary for display
     */
    getFormattedSummary(result) {
        const summary = [];
        
        // Key attributes to display
        const keyAttributes = [
            'item_type',
            'brand',
            'color',
            'size',
            'condition',
            'damage_detected',
            'return_valid',
            'confidence'
        ];

        keyAttributes.forEach(attr => {
            if (result[attr] !== undefined) {
                const label = attr.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                let value = result[attr];
                
                // Format boolean values
                if (typeof value === 'boolean') {
                    value = value ? 'Yes' : 'No';
                }
                
                // Format confidence as percentage
                if (attr === 'confidence' && typeof value === 'number') {
                    value = `${Math.round(value * 100)}%`;
                }
                
                summary.push({ label, value });
            }
        });

        return summary;
    }

    /**
     * Generate CSV report with statistics
     */
    generateReport() {
        if (this.sessionHistory.length === 0) {
            return null;
        }

        // Calculate statistics
        const stats = {
            total_items: this.sessionHistory.length,
            valid_returns: this.sessionHistory.filter(r => r.return_valid === true).length,
            invalid_returns: this.sessionHistory.filter(r => r.return_valid === false).length,
            damaged_items: this.sessionHistory.filter(r => r.damage_detected === true).length,
            avg_confidence: this.calculateAverage(this.sessionHistory.map(r => r.confidence || 0)),
            item_types: this.getUniqueValues(this.sessionHistory, 'item_type'),
            brands: this.getUniqueValues(this.sessionHistory, 'brand'),
            conditions: this.getUniqueValues(this.sessionHistory, 'condition')
        };

        return stats;
    }

    /**
     * Calculate average of array
     */
    calculateAverage(arr) {
        if (arr.length === 0) return 0;
        const sum = arr.reduce((a, b) => a + b, 0);
        return (sum / arr.length).toFixed(2);
    }

    /**
     * Get unique values for a field
     */
    getUniqueValues(data, field) {
        const values = new Set();
        data.forEach(record => {
            if (record[field]) {
                values.add(record[field]);
            }
        });
        return Array.from(values);
    }
}

// Export for use in other modules
window.CSVExporter = CSVExporter;