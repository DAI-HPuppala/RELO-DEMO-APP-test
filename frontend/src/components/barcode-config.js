/**
 * Barcode Configuration Component
 * Manages barcode detection settings and type selection
 */
class BarcodeConfig {
    constructor() {
        this.configPanel = document.getElementById('barcodeConfigPanel');
        this.toggleBtn = document.getElementById('barcodeToggleBtn');
        this.statusBadge = document.getElementById('barcodeStatusBadge');
        this.typeDropdown = document.getElementById('barcodeTypeDropdown');
        this.typeList = document.getElementById('barcodeTypeList');
        this.selectedTypesContainer = document.getElementById('selectedBarcodeTypes');
        this.applyBtn = document.getElementById('applyBarcodeConfig');
        this.cancelBtn = document.getElementById('cancelBarcodeConfig');

        // Configuration state
        this.config = {
            enabled: true,
            required: false,
            timeout_seconds: 0,
            auto_delay_ms: 1000,
            allowed_types: ['CODE128', 'QRCODE', 'EAN13'],
            supported_types: []
        };

        this.selectedTypes = new Set(this.config.allowed_types);
        this.isPanelOpen = false;

        this.setupEventListeners();
        this.loadConfig();
    }

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Toggle config panel
        if (this.toggleBtn) {
            this.toggleBtn.addEventListener('click', () => this.togglePanel());
        }

        // Dropdown toggle
        if (this.typeDropdown) {
            this.typeDropdown.addEventListener('click', (e) => {
                e.stopPropagation();
                this.typeList.classList.toggle('show');
            });
        }

        // Close dropdown when clicking outside
        document.addEventListener('click', (e) => {
            if (this.typeList && !this.typeDropdown.contains(e.target)) {
                this.typeList.classList.remove('show');
            }
        });

        // Apply and Cancel buttons
        if (this.applyBtn) {
            this.applyBtn.addEventListener('click', () => this.saveConfig());
        }

        if (this.cancelBtn) {
            this.cancelBtn.addEventListener('click', () => this.cancelConfig());
        }
    }

    /**
     * Toggle configuration panel
     */
    togglePanel() {
        this.isPanelOpen = !this.isPanelOpen;

        if (this.configPanel) {
            this.configPanel.classList.toggle('show', this.isPanelOpen);
        }

        if (this.toggleBtn) {
            const icon = this.toggleBtn.querySelector('.btn-icon');
            if (icon) {
                icon.textContent = this.isPanelOpen ? '⚙' : '⚙';
            }
        }
    }

    /**
     * Load configuration from backend
     */
    async loadConfig() {
        try {
            const response = await fetch('http://localhost:8000/api/barcode/config');
            const data = await response.json();

            this.config = {
                enabled: data.enabled,
                required: data.required,
                timeout_seconds: data.timeout_seconds,
                auto_delay_ms: data.auto_delay_ms,
                allowed_types: data.allowed_types || [],
                supported_types: data.supported_types || []
            };

            this.selectedTypes = new Set(this.config.allowed_types);

            this.updateUI();
            this.renderTypeCheckboxes();
            this.updateSelectedDisplay();

            console.log('Barcode config loaded:', this.config);
        } catch (error) {
            console.error('Failed to load barcode config:', error);
            this.showError('Failed to load configuration');
        }
    }

    /**
     * Render barcode type checkboxes
     */
    renderTypeCheckboxes() {
        if (!this.typeList) return;

        this.typeList.innerHTML = '';

        this.config.supported_types.forEach(type => {
            const item = document.createElement('div');
            item.className = 'dropdown-item';

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.id = `barcode-type-${type}`;
            checkbox.value = type;
            checkbox.checked = this.selectedTypes.has(type);

            checkbox.addEventListener('change', (e) => {
                if (e.target.checked) {
                    this.selectedTypes.add(type);
                } else {
                    this.selectedTypes.delete(type);
                }
                this.updateSelectedDisplay();
            });

            const label = document.createElement('label');
            label.htmlFor = `barcode-type-${type}`;
            label.textContent = type;

            item.appendChild(checkbox);
            item.appendChild(label);
            this.typeList.appendChild(item);
        });
    }

    /**
     * Update selected types display
     */
    updateSelectedDisplay() {
        if (!this.selectedTypesContainer) return;

        this.selectedTypesContainer.innerHTML = '';

        if (this.selectedTypes.size === 0) {
            const placeholder = document.createElement('span');
            placeholder.className = 'selected-placeholder';
            placeholder.textContent = 'No types selected';
            this.selectedTypesContainer.appendChild(placeholder);
            return;
        }

        this.selectedTypes.forEach(type => {
            const tag = document.createElement('span');
            tag.className = 'selected-type-tag';
            tag.textContent = type;

            const removeBtn = document.createElement('button');
            removeBtn.className = 'remove-type-btn';
            removeBtn.textContent = '×';
            removeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.selectedTypes.delete(type);
                this.updateSelectedDisplay();
                this.updateCheckboxState(type, false);
            });

            tag.appendChild(removeBtn);
            this.selectedTypesContainer.appendChild(tag);
        });
    }

    /**
     * Update checkbox state for a specific type
     */
    updateCheckboxState(type, checked) {
        const checkbox = document.getElementById(`barcode-type-${type}`);
        if (checkbox) {
            checkbox.checked = checked;
        }
    }

    /**
     * Save configuration to backend
     */
    async saveConfig() {
        try {
            const configUpdate = {
                allowed_types: Array.from(this.selectedTypes)
            };

            const response = await fetch('http://localhost:8000/api/barcode/config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(configUpdate)
            });

            const result = await response.json();

            if (result.success) {
                this.config.allowed_types = Array.from(this.selectedTypes);
                this.updateUI();
                this.togglePanel();
                this.showSuccess('Configuration saved successfully');
                console.log('Barcode config saved:', result);
            } else {
                this.showError('Failed to save configuration');
            }
        } catch (error) {
            console.error('Failed to save barcode config:', error);
            this.showError('Failed to save configuration');
        }
    }

    /**
     * Cancel configuration changes
     */
    cancelConfig() {
        // Revert to original config
        this.selectedTypes = new Set(this.config.allowed_types);
        this.renderTypeCheckboxes();
        this.updateSelectedDisplay();
        this.togglePanel();
    }

    /**
     * Update UI based on current config
     */
    updateUI() {
        // Update status badge
        if (this.statusBadge) {
            this.statusBadge.textContent = this.config.enabled ? 'Enabled' : 'Disabled';
            this.statusBadge.className = `status-badge ${this.config.enabled ? 'enabled' : 'disabled'}`;
        }

        // Update toggle button state
        if (this.toggleBtn) {
            this.toggleBtn.disabled = !this.config.enabled;
            this.toggleBtn.title = this.config.enabled
                ? 'Configure barcode detection'
                : 'Barcode detection is disabled';
        }
    }

    /**
     * Show success message
     */
    showSuccess(message) {
        // Create temporary success notification
        const notification = document.createElement('div');
        notification.className = 'barcode-notification success';
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
     * Show error message
     */
    showError(message) {
        // Create temporary error notification
        const notification = document.createElement('div');
        notification.className = 'barcode-notification error';
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
     * Get current configuration
     */
    getConfig() {
        return this.config;
    }

    /**
     * Check if barcode detection is enabled
     */
    isEnabled() {
        return this.config.enabled;
    }

    /**
     * Get selected barcode types
     */
    getSelectedTypes() {
        return Array.from(this.selectedTypes);
    }
}

// Export for use in other modules
window.BarcodeConfig = BarcodeConfig;
