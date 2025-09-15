/**
 * Control Panel Component V2
 * Provides pause/resume controls and session management
 */

class ControlPanelV2 extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        
        // State
        this.isRunning = false;
        this.isPaused = false;
        this.sessionId = null;
        this.hasCheckpoint = false;
        
        // Callbacks
        this.onStart = null;
        this.onPause = null;
        this.onResume = null;
        this.onRecover = null;
        this.onReset = null;
        
        this.render();
        this.attachEventListeners();
    }
    
    render() {
        this.shadowRoot.innerHTML = `
            <style>
                :host {
                    display: block;
                    padding: 20px;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                }
                
                .control-panel {
                    background: white;
                    border-radius: 12px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    padding: 20px;
                }
                
                .panel-header {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    margin-bottom: 20px;
                }
                
                .panel-title {
                    font-size: 18px;
                    font-weight: 600;
                    color: #333;
                }
                
                .session-info {
                    font-size: 12px;
                    color: #666;
                    padding: 4px 8px;
                    background: #f0f0f0;
                    border-radius: 4px;
                }
                
                .controls-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                    gap: 10px;
                    margin-bottom: 20px;
                }
                
                .control-btn {
                    padding: 12px 20px;
                    border: none;
                    border-radius: 8px;
                    font-size: 14px;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 8px;
                }
                
                .control-btn:disabled {
                    opacity: 0.5;
                    cursor: not-allowed;
                }
                
                .btn-start {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                }
                
                .btn-start:hover:not(:disabled) {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
                }
                
                .btn-pause {
                    background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%);
                    color: white;
                }
                
                .btn-pause:hover:not(:disabled) {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 12px rgba(255, 216, 155, 0.4);
                }
                
                .btn-resume {
                    background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
                    color: white;
                }
                
                .btn-resume:hover:not(:disabled) {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 12px rgba(132, 250, 176, 0.4);
                }
                
                .btn-reset {
                    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    color: white;
                }
                
                .btn-reset:hover:not(:disabled) {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 12px rgba(240, 147, 251, 0.4);
                }
                
                .btn-recover {
                    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                    color: white;
                }
                
                .btn-recover:hover:not(:disabled) {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 12px rgba(79, 172, 254, 0.4);
                }
                
                .status-bar {
                    display: flex;
                    align-items: center;
                    padding: 10px;
                    background: #f8f9fa;
                    border-radius: 8px;
                    margin-bottom: 15px;
                }
                
                .status-indicator {
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                    margin-right: 10px;
                    animation: pulse 2s infinite;
                }
                
                .status-idle {
                    background: #ccc;
                    animation: none;
                }
                
                .status-running {
                    background: #4caf50;
                }
                
                .status-paused {
                    background: #ff9800;
                }
                
                .status-error {
                    background: #f44336;
                }
                
                @keyframes pulse {
                    0% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7); }
                    50% { box-shadow: 0 0 0 10px rgba(76, 175, 80, 0); }
                    100% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); }
                }
                
                .status-text {
                    flex: 1;
                    font-size: 14px;
                    color: #666;
                }
                
                .checkpoint-indicator {
                    display: flex;
                    align-items: center;
                    gap: 5px;
                    font-size: 12px;
                    color: #666;
                }
                
                .checkpoint-icon {
                    color: #4caf50;
                }
                
                .advanced-controls {
                    margin-top: 15px;
                    padding-top: 15px;
                    border-top: 1px solid #e0e0e0;
                }
                
                .advanced-title {
                    font-size: 14px;
                    font-weight: 600;
                    color: #666;
                    margin-bottom: 10px;
                }
                
                .restart-option {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    margin-bottom: 10px;
                }
                
                .restart-checkbox {
                    width: 18px;
                    height: 18px;
                }
                
                .restart-label {
                    font-size: 14px;
                    color: #666;
                }
            </style>
            
            <div class="control-panel">
                <div class="panel-header">
                    <h2 class="panel-title">Flow Control</h2>
                    <div class="session-info" id="sessionInfo">No Session</div>
                </div>
                
                <div class="status-bar">
                    <div class="status-indicator status-idle" id="statusIndicator"></div>
                    <div class="status-text" id="statusText">Ready to start</div>
                    <div class="checkpoint-indicator" id="checkpointIndicator" style="display: none;">
                        <span class="checkpoint-icon">💾</span>
                        <span>Checkpoint saved</span>
                    </div>
                </div>
                
                <div class="controls-grid">
                    <button class="control-btn btn-start" id="startBtn">
                        <span>▶️</span>
                        <span>Start</span>
                    </button>
                    
                    <button class="control-btn btn-pause" id="pauseBtn" disabled>
                        <span>⏸️</span>
                        <span>Pause</span>
                    </button>
                    
                    <button class="control-btn btn-resume" id="resumeBtn" disabled>
                        <span>▶️</span>
                        <span>Resume</span>
                    </button>
                    
                    <button class="control-btn btn-reset" id="resetBtn">
                        <span>🔄</span>
                        <span>Reset</span>
                    </button>
                    
                    <button class="control-btn btn-recover" id="recoverBtn" disabled>
                        <span>📂</span>
                        <span>Recover</span>
                    </button>
                </div>
                
                <div class="advanced-controls">
                    <div class="advanced-title">Resume Options</div>
                    <div class="restart-option">
                        <input type="checkbox" id="restartAgent" class="restart-checkbox">
                        <label for="restartAgent" class="restart-label">
                            Restart current agent when resuming
                        </label>
                    </div>
                </div>
            </div>
        `;
    }
    
    attachEventListeners() {
        const startBtn = this.shadowRoot.getElementById('startBtn');
        const pauseBtn = this.shadowRoot.getElementById('pauseBtn');
        const resumeBtn = this.shadowRoot.getElementById('resumeBtn');
        const resetBtn = this.shadowRoot.getElementById('resetBtn');
        const recoverBtn = this.shadowRoot.getElementById('recoverBtn');
        
        startBtn.addEventListener('click', () => this.handleStart());
        pauseBtn.addEventListener('click', () => this.handlePause());
        resumeBtn.addEventListener('click', () => this.handleResume());
        resetBtn.addEventListener('click', () => this.handleReset());
        recoverBtn.addEventListener('click', () => this.handleRecover());
    }
    
    handleStart() {
        if (this.onStart) {
            this.onStart();
        }
        this.setRunningState();
    }
    
    handlePause() {
        if (this.onPause) {
            this.onPause();
        }
        this.setPausedState();
    }
    
    handleResume() {
        const restartAgent = this.shadowRoot.getElementById('restartAgent').checked;
        if (this.onResume) {
            this.onResume(restartAgent);
        }
        this.setRunningState();
    }
    
    handleReset() {
        if (this.onReset) {
            this.onReset();
        }
        this.setIdleState();
    }
    
    handleRecover() {
        if (this.onRecover) {
            this.onRecover();
        }
    }
    
    setIdleState() {
        this.isRunning = false;
        this.isPaused = false;
        
        const statusIndicator = this.shadowRoot.getElementById('statusIndicator');
        const statusText = this.shadowRoot.getElementById('statusText');
        const startBtn = this.shadowRoot.getElementById('startBtn');
        const pauseBtn = this.shadowRoot.getElementById('pauseBtn');
        const resumeBtn = this.shadowRoot.getElementById('resumeBtn');
        
        statusIndicator.className = 'status-indicator status-idle';
        statusText.textContent = 'Ready to start';
        
        startBtn.disabled = false;
        pauseBtn.disabled = true;
        resumeBtn.disabled = true;
    }
    
    setRunningState() {
        this.isRunning = true;
        this.isPaused = false;
        
        const statusIndicator = this.shadowRoot.getElementById('statusIndicator');
        const statusText = this.shadowRoot.getElementById('statusText');
        const startBtn = this.shadowRoot.getElementById('startBtn');
        const pauseBtn = this.shadowRoot.getElementById('pauseBtn');
        const resumeBtn = this.shadowRoot.getElementById('resumeBtn');
        
        statusIndicator.className = 'status-indicator status-running';
        statusText.textContent = 'Processing...';
        
        startBtn.disabled = true;
        pauseBtn.disabled = false;
        resumeBtn.disabled = true;
    }
    
    setPausedState() {
        this.isRunning = false;
        this.isPaused = true;
        
        const statusIndicator = this.shadowRoot.getElementById('statusIndicator');
        const statusText = this.shadowRoot.getElementById('statusText');
        const startBtn = this.shadowRoot.getElementById('startBtn');
        const pauseBtn = this.shadowRoot.getElementById('pauseBtn');
        const resumeBtn = this.shadowRoot.getElementById('resumeBtn');
        
        statusIndicator.className = 'status-indicator status-paused';
        statusText.textContent = 'Paused';
        
        startBtn.disabled = true;
        pauseBtn.disabled = true;
        resumeBtn.disabled = false;
    }
    
    setErrorState(message) {
        const statusIndicator = this.shadowRoot.getElementById('statusIndicator');
        const statusText = this.shadowRoot.getElementById('statusText');
        
        statusIndicator.className = 'status-indicator status-error';
        statusText.textContent = `Error: ${message}`;
    }
    
    updateSessionInfo(sessionId) {
        this.sessionId = sessionId;
        const sessionInfo = this.shadowRoot.getElementById('sessionInfo');
        if (sessionId) {
            sessionInfo.textContent = `Session: ${sessionId.slice(0, 8)}...`;
        } else {
            sessionInfo.textContent = 'No Session';
        }
    }
    
    showCheckpoint(show = true) {
        this.hasCheckpoint = show;
        const checkpointIndicator = this.shadowRoot.getElementById('checkpointIndicator');
        const recoverBtn = this.shadowRoot.getElementById('recoverBtn');
        
        checkpointIndicator.style.display = show ? 'flex' : 'none';
        recoverBtn.disabled = !show;
    }
    
    updateStatus(text) {
        const statusText = this.shadowRoot.getElementById('statusText');
        statusText.textContent = text;
    }
}

// Register custom element
customElements.define('control-panel-v2', ControlPanelV2);