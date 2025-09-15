/**
 * Shared agent configuration
 * Ensures consistency between frontend and backend
 */
const AGENT_CONFIG = {
    agents: [
        {
            id: 'initial_classifier',
            name: 'Initial Classifier',
            displayName: 'Initial Classification',
            timer: 4.0,
            description: 'Identifies basic garment attributes'
        },
        {
            id: 'detail_extractor',
            name: 'Detail Extractor',
            displayName: 'Detail Extraction',
            timer: 3.0,
            description: 'Extracts size and brand information'
        },
        {
            id: 'damage_detector',
            name: 'Damage Detector',
            displayName: 'Damage Detection',
            timer: 4.0,
            description: 'Detects damages and defects'
        },
        {
            id: 'final_compiler',
            name: 'Final Compiler',
            displayName: 'Final Compilation',
            timer: 2.0,
            description: 'Compiles final classification'
        }
    ],
    
    getAgentById: function(id) {
        return this.agents.find(agent => agent.id === id);
    },
    
    getAgentByName: function(name) {
        return this.agents.find(agent => 
            agent.name === name || 
            agent.displayName === name ||
            agent.id === name
        );
    },
    
    getTimers: function() {
        const timers = {};
        this.agents.forEach(agent => {
            timers[agent.id] = agent.timer;
        });
        return timers;
    },
    
    getSequence: function() {
        return this.agents.map(agent => agent.id);
    }
};

// Export for use
window.AGENT_CONFIG = AGENT_CONFIG;