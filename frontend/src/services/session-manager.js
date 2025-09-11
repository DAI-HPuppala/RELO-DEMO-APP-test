/**
 * Session Manager for handling classification session state
 */
class SessionManager {
    constructor() {
        this.currentSession = null;
        this.mode = 'automatic';
        this.agents = {
            'initial_classifier': {
                name: 'Initial Classifier',
                status: 'waiting',
                results: {}
            },
            'detail_extractor': {
                name: 'Detail Extractor',
                status: 'waiting',
                results: {}
            },
            'damage_detector': {
                name: 'Damage Detector',
                status: 'waiting',
                results: {}
            },
            'final_compiler': {
                name: 'Final Compiler',
                status: 'waiting',
                results: {}
            }
        };
        this.finalResults = null;
        this.startTime = null;
        this.processingTime = 0;
        this.framesAnalyzed = 0;
    }

    /**
     * Start a new session
     */
    startSession(sessionId, mode) {
        this.currentSession = sessionId;
        this.mode = mode;
        this.startTime = Date.now();
        this.resetAgents();
        this.finalResults = null;
        this.processingTime = 0;
        this.framesAnalyzed = 0;
        
        console.log(`Session started: ${sessionId} in ${mode} mode`);
    }

    /**
     * Stop the current session
     */
    stopSession() {
        if (this.startTime) {
            this.processingTime = (Date.now() - this.startTime) / 1000;
        }
        console.log(`Session stopped: ${this.currentSession}`);
        this.currentSession = null;
        this.startTime = null;
    }

    /**
     * Reset all agents to waiting state
     */
    resetAgents() {
        for (const agentId in this.agents) {
            this.agents[agentId].status = 'waiting';
            this.agents[agentId].results = {};
        }
    }

    /**
     * Update agent status
     */
    updateAgentStatus(agentId, status) {
        if (this.agents[agentId]) {
            this.agents[agentId].status = status;
        }
    }

    /**
     * Update agent results
     */
    updateAgentResults(agentId, results) {
        if (this.agents[agentId]) {
            this.agents[agentId].results = results;
            this.agents[agentId].status = 'completed';
        }
    }

    /**
     * Handle progressive update
     */
    handleProgressiveUpdate(data) {
        const { agent, attributes, frames_processed } = data;
        
        if (this.agents[agent]) {
            // Merge new attributes with existing ones
            this.agents[agent].results = {
                ...this.agents[agent].results,
                ...attributes
            };
        }
        
        if (frames_processed) {
            this.framesAnalyzed = Math.max(this.framesAnalyzed, frames_processed);
        }
    }

    /**
     * Set final results
     */
    setFinalResults(results) {
        this.finalResults = results.classification;
        this.processingTime = results.processing_time || this.processingTime;
        this.framesAnalyzed = results.total_frames || this.framesAnalyzed;
    }

    /**
     * Get current progress percentage
     */
    getProgress() {
        let completedAgents = 0;
        for (const agentId in this.agents) {
            if (this.agents[agentId].status === 'completed') {
                completedAgents++;
            }
        }
        return Math.round((completedAgents / 4) * 100);
    }

    /**
     * Get compiled results from all agents
     */
    getCompiledResults() {
        const compiled = {
            type: null,
            color_primary: null,
            color_secondary: null,
            pattern: null,
            brand: null,
            size: null,
            has_damage: false,
            damage_type: null
        };

        // Compile from initial classifier
        const initial = this.agents.initial_classifier.results;
        if (initial) {
            compiled.type = initial.type || compiled.type;
            compiled.color_primary = initial.color || initial.color_primary || compiled.color_primary;
            compiled.pattern = initial.pattern || compiled.pattern;
        }

        // Compile from detail extractor
        const detail = this.agents.detail_extractor.results;
        if (detail) {
            compiled.brand = detail.brand || compiled.brand;
            compiled.size = detail.size || compiled.size;
        }

        // Compile from damage detector
        const damage = this.agents.damage_detector.results;
        if (damage) {
            compiled.has_damage = damage.damage === 'Yes' || damage.has_damage || false;
            compiled.damage_type = damage.damage_type || compiled.damage_type;
        }

        return compiled;
    }

    /**
     * Export session data
     */
    exportSessionData() {
        return {
            session_id: this.currentSession,
            mode: this.mode,
            processing_time: this.processingTime,
            frames_analyzed: this.framesAnalyzed,
            agents: this.agents,
            final_results: this.finalResults || this.getCompiledResults(),
            timestamp: new Date().toISOString()
        };
    }
}

// Export for use in other modules
window.SessionManager = SessionManager;