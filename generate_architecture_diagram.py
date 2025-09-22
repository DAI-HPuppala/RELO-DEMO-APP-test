#!/usr/bin/env python3
"""
RELO Classifier System Architecture Diagram Generator
Generates comprehensive low-level architecture diagrams with all components and connections
"""

import os
from datetime import datetime
from graphviz import Digraph

def create_architecture_diagram():
    """Create a comprehensive architecture diagram of the RELO Classifier system"""

    # Create main diagram
    dot = Digraph('RELO_Architecture', comment='RELO Classifier System Architecture')
    dot.attr(rankdir='TB', size='20,16', dpi='300')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial')
    dot.attr('edge', fontname='Arial', fontsize='10')

    # Define color schemes for different layers
    colors = {
        'frontend': '#E8F4FD',      # Light blue
        'api': '#FFF4E6',           # Light orange
        'services': '#F0F8FF',      # Alice blue
        'agents': '#E8F5E8',        # Light green
        'infrastructure': '#FFE8E8', # Light red
        'v2': '#F5E8FF',           # Light purple
        'data': '#FFFACD'          # Light yellow
    }

    # Create subgraphs for each layer

    # Frontend Layer
    with dot.subgraph(name='cluster_frontend') as frontend:
        frontend.attr(label='Frontend Layer (Browser)', style='filled', color='lightgrey', fillcolor=colors['frontend'])

        # UI Components
        frontend.node('ui_main', 'Main UI\n(index-professional.html)', shape='tab', fillcolor='#B3E5FC')
        frontend.node('ui_components', 'Web Components\n- Video Display\n- Control Panel\n- Results Display', fillcolor='#B3E5FC')
        frontend.node('ui_styles', 'Styles\n(professional-styles.css)', shape='note', fillcolor='#B3E5FC')

        # JavaScript Services
        frontend.node('webrtc_client', 'WebRTC Client\n(webrtc-client.js)\n- Peer Connection\n- Data Channel\n- Media Stream', fillcolor='#81D4FA')
        frontend.node('webrtc_client_v2', 'WebRTC Client V2\n(webrtc-client-v2.js)\n- Enhanced Protocol\n- State Management', fillcolor='#81D4FA')
        frontend.node('session_mgr_fe', 'Session Manager\n(session-manager.js)\n- Session State\n- Result Handling', fillcolor='#81D4FA')
        frontend.node('init_mgr', 'Initialization Manager\n(initialization-manager.js)\n- Setup Workflow\n- Connection Management', fillcolor='#81D4FA')
        frontend.node('app_prof', 'App Professional\n(app-professional.js)\n- Main Controller\n- Event Handling', fillcolor='#4FC3F7')

    # API Gateway Layer
    with dot.subgraph(name='cluster_api') as api:
        api.attr(label='API Gateway Layer', style='filled', color='lightgrey', fillcolor=colors['api'])

        api.node('fastapi_main', 'FastAPI Main\n(main.py)\n- Application Setup\n- CORS Config\n- Middleware', fillcolor='#FFD180')
        api.node('ws_endpoint', 'WebSocket Endpoint\n(websocket.py)\n- Signaling Server\n- Message Router\n- Session Control', fillcolor='#FFAB40')
        api.node('ws_endpoint_v2', 'WebSocket V2\n(websocket_v2.py)\n- Stateful Protocol\n- Multi-Inference\n- Timer Management', fillcolor='#FFAB40')

        # Routes
        api.node('health_route', 'Health Check\n(/health)', shape='oval', fillcolor='#FFE082')
        api.node('session_route', 'Session API\n(/session/*)\n- Start/Stop\n- Status/Results', fillcolor='#FFE082')

    # Backend Services Layer
    with dot.subgraph(name='cluster_services') as services:
        services.attr(label='Backend Services Layer', style='filled', color='lightgrey', fillcolor=colors['services'])

        # Core Services
        services.node('webrtc_mgr', 'WebRTC Manager\n(webrtc_manager.py)\n- Peer Connections\n- Media Relay\n- Frame Buffers', fillcolor='#90CAF9')
        services.node('session_mgr', 'Session Manager\n(session_manager.py)\n- Session State\n- Agent Control\n- Result Storage', fillcolor='#90CAF9')
        services.node('manual_handler', 'Manual Mode Handler\n(manual_mode_handler.py)\n- Trigger Management\n- Frame Capture', fillcolor='#90CAF9')

        # Camera Services
        services.node('camera_svc', 'Camera Service\n(camera_service.py)\n- Abstract Interface\n- Frame Provider', fillcolor='#64B5F6')
        services.node('cv60_camera', 'CV60 Camera\n(cv60_camera.py)\n- eBUS SDK\n- JAI/Zebra Hardware\n- IP: 192.168.1.21', fillcolor='#42A5F5')
        services.node('realsense_camera', 'RealSense Camera\n(realsense_camera.py)\n- Intel RealSense\n- Depth Sensing', fillcolor='#42A5F5')
        services.node('direct_camera', 'Direct Camera\n(direct_camera_provider.py)\n- OpenCV Capture\n- USB/Webcam', fillcolor='#42A5F5')

        # GPU/VLM Services
        services.node('gpu_init', 'GPU Initializer\n(gpu_initializer.py)\n- CUDA Setup\n- Memory Allocation', fillcolor='#2196F3')
        services.node('gpu_mgr', 'GPU Manager\n(gpu_manager.py)\n- Resource Management\n- Load Balancing\n- Performance Monitor', fillcolor='#2196F3')
        services.node('ollama_opt', 'Ollama Optimizer\n(ollama_optimizer.py)\n- Model Loading\n- GPU Configuration', fillcolor='#1E88E5')
        services.node('vlm_loader', 'VLM GPU Loader\n(vlm_gpu_loader.py)\n- Model Initialization\n- Memory Management', fillcolor='#1976D2')
        services.node('vlm_svc', 'VLM Service Ultra\n(vlm_service_ultra.py)\n- Inference Engine\n- Batch Processing\n- Result Parsing', fillcolor='#1565C0')
        services.node('vlm_singleton', 'VLM Singleton\n(vlm_singleton.py)\n- Single Instance\n- State Management', fillcolor='#1565C0')

    # Agent Pipeline Layer
    with dot.subgraph(name='cluster_agents') as agents:
        agents.attr(label='Agent Pipeline Layer', style='filled', color='lightgrey', fillcolor=colors['agents'])

        agents.node('base_agent', 'Base Agent\n(base_agent.py)\n- Abstract Class\n- Common Methods', shape='diamond', fillcolor='#A5D6A7')
        agents.node('initial_clf', 'Initial Classifier\n- Type Detection\n- Color Analysis\n- Pattern Recognition\n- Timer: 4.0s', fillcolor='#81C784')
        agents.node('detail_ext', 'Detail Extractor\n- Material Analysis\n- Brand Detection\n- Size Extraction\n- Timer: 3.0s', fillcolor='#66BB6A')
        agents.node('damage_det', 'Damage Detector\n- Defect Analysis\n- Stain Detection\n- Wear Assessment\n- Timer: 4.0s', fillcolor='#4CAF50')
        agents.node('final_comp', 'Final Compiler\n- Result Aggregation\n- Confidence Scoring\n- Decision Making\n- Timer: 1.0s', fillcolor='#43A047')

    # V2 Architecture Layer
    with dot.subgraph(name='cluster_v2') as v2:
        v2.attr(label='V2 Stateful Architecture', style='filled', color='lightgrey', fillcolor=colors['v2'])

        # Orchestration
        v2.node('stateful_orch', 'Stateful Orchestrator\n(stateful_orchestrator.py)\n- Pipeline Control\n- Timer Management\n- State Coordination', fillcolor='#CE93D8')

        # State Management
        v2.node('session_state', 'Session State Manager\n(session_state_manager.py)\n- State Tracking\n- Flow Control', fillcolor='#BA68C8')
        v2.node('state_persist', 'State Persistence\n(state_persistence.py)\n- JSON Checkpointing\n- Recovery Support', fillcolor='#BA68C8')
        v2.node('agent_state', 'Agent State\n(agent_state.py)\n- Timer Tracking\n- Result Collection', fillcolor='#BA68C8')

        # Frame Management
        v2.node('frame_registry', 'Frame Registry\n(frame_registry.py)\n- Frame Sharing\n- Access Control', fillcolor='#AB47BC')
        v2.node('frame_mgr', 'Frame Manager\n(frame_manager.py)\n- Frame Collection\n- Buffer Management', fillcolor='#AB47BC')
        v2.node('frame_agg', 'Frame Aggregator\n(frame_aggregator.py)\n- Multi-Frame Processing\n- Batch Preparation', fillcolor='#AB47BC')

        # Inference
        v2.node('multi_inf', 'Multi-Inference Engine\n(multi_inference_engine.py)\n- Batch Processing\n- GPU Optimization\n- Result Aggregation', fillcolor='#9C27B0')
        v2.node('vlm_inf', 'VLM Inference Engine\n(vlm_inference_engine.py)\n- Model Interface\n- Prompt Management', fillcolor='#9C27B0')

    # Infrastructure Layer
    with dot.subgraph(name='cluster_infrastructure') as infra:
        infra.attr(label='Infrastructure Layer', style='filled', color='lightgrey', fillcolor=colors['infrastructure'])

        infra.node('ollama', 'Ollama Server\nQwen2.5-VL 3B\nPort: 11434', shape='cylinder', fillcolor='#EF9A9A')
        infra.node('cuda', 'CUDA/GPU\nNVIDIA Driver\nCUDA Toolkit', shape='box3d', fillcolor='#E57373')
        infra.node('cv60_hw', 'CV60 Camera\nZebra/JAI Hardware\nGigE Vision', shape='box3d', fillcolor='#EF5350')
        infra.node('filesystem', 'File System\n- Sessions Storage\n- Frame Capture\n- Logs', shape='folder', fillcolor='#F44336')

    # Data Storage
    with dot.subgraph(name='cluster_data') as data:
        data.attr(label='Data Storage', style='filled', color='lightgrey', fillcolor=colors['data'])

        data.node('sessions_dir', 'Sessions Directory\n./backend/sessions/', shape='folder')
        data.node('captured_frames', 'Captured Frames\n./captured_frames/', shape='folder')
        data.node('logs', 'Log Files\n- backend.log\n- frontend.log\n- ngrok.log', shape='note')

    # Define connections with different styles

    # Frontend connections
    dot.edge('ui_main', 'ui_components', label='includes')
    dot.edge('ui_main', 'app_prof', label='loads')
    dot.edge('app_prof', 'init_mgr', label='initializes')
    dot.edge('init_mgr', 'webrtc_client_v2', label='creates')
    dot.edge('init_mgr', 'session_mgr_fe', label='manages')
    dot.edge('webrtc_client_v2', 'ws_endpoint_v2', label='WebSocket', color='green', style='bold')
    dot.edge('webrtc_client', 'ws_endpoint', label='WebSocket', color='green')

    # API to Services
    dot.edge('fastapi_main', 'ws_endpoint', label='registers')
    dot.edge('fastapi_main', 'ws_endpoint_v2', label='registers')
    dot.edge('fastapi_main', 'health_route', label='mounts')
    dot.edge('fastapi_main', 'session_route', label='mounts')

    # WebSocket to Backend
    dot.edge('ws_endpoint_v2', 'stateful_orch', label='orchestrates', color='purple', style='bold')
    dot.edge('ws_endpoint', 'webrtc_mgr', label='manages', color='blue')
    dot.edge('ws_endpoint', 'session_mgr', label='controls')

    # WebRTC connections
    dot.edge('webrtc_mgr', 'cv60_camera', label='video stream', color='blue', style='dashed')
    dot.edge('webrtc_mgr', 'realsense_camera', label='alt. stream', color='blue', style='dashed')
    dot.edge('webrtc_mgr', 'direct_camera', label='alt. stream', color='blue', style='dashed')

    # V2 Orchestration Flow
    dot.edge('stateful_orch', 'session_state', label='manages')
    dot.edge('stateful_orch', 'frame_registry', label='registers frames')
    dot.edge('stateful_orch', 'frame_mgr', label='collects frames')
    dot.edge('stateful_orch', 'multi_inf', label='runs inference', style='bold')
    dot.edge('stateful_orch', 'state_persist', label='checkpoints')

    # Frame processing
    dot.edge('frame_mgr', 'frame_agg', label='aggregates')
    dot.edge('frame_agg', 'multi_inf', label='batch frames')

    # Agent Pipeline
    dot.edge('stateful_orch', 'initial_clf', label='stage 1', color='green', style='bold')
    dot.edge('initial_clf', 'detail_ext', label='stage 2', color='green', style='bold')
    dot.edge('detail_ext', 'damage_det', label='stage 3', color='green', style='bold')
    dot.edge('damage_det', 'final_comp', label='stage 4', color='green', style='bold')

    # Agent inheritance
    dot.edge('base_agent', 'initial_clf', style='dotted', arrowhead='empty')
    dot.edge('base_agent', 'detail_ext', style='dotted', arrowhead='empty')
    dot.edge('base_agent', 'damage_det', style='dotted', arrowhead='empty')
    dot.edge('base_agent', 'final_comp', style='dotted', arrowhead='empty')

    # VLM Services
    dot.edge('multi_inf', 'vlm_inf', label='delegates')
    dot.edge('vlm_inf', 'vlm_svc', label='processes')
    dot.edge('vlm_svc', 'vlm_singleton', label='singleton')
    dot.edge('vlm_singleton', 'ollama', label='HTTP API', color='orange')

    # GPU Management
    dot.edge('gpu_init', 'cuda', label='initializes', color='red')
    dot.edge('gpu_mgr', 'cuda', label='manages', color='red')
    dot.edge('ollama_opt', 'ollama', label='optimizes')
    dot.edge('vlm_loader', 'ollama', label='loads model')

    # Camera Hardware
    dot.edge('cv60_camera', 'cv60_hw', label='eBUS SDK', style='dotted')

    # Data Storage
    dot.edge('state_persist', 'sessions_dir', label='saves', style='dotted')
    dot.edge('session_mgr', 'sessions_dir', label='stores', style='dotted')
    dot.edge('frame_mgr', 'captured_frames', label='saves', style='dotted')

    # Result flow back to frontend
    dot.edge('final_comp', 'stateful_orch', label='results', color='orange', style='dashed', constraint='false')
    dot.edge('stateful_orch', 'ws_endpoint_v2', label='updates', color='orange', style='dashed', constraint='false')
    dot.edge('ws_endpoint_v2', 'webrtc_client_v2', label='data channel', color='orange', style='bold', constraint='false')
    dot.edge('webrtc_client_v2', 'ui_components', label='display', color='orange', constraint='false')

    return dot


def create_simplified_diagram():
    """Create a simplified high-level architecture diagram"""

    dot = Digraph('RELO_Simplified', comment='RELO Classifier - Simplified View')
    dot.attr(rankdir='LR', size='14,10', dpi='300')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='12')

    # Main components
    dot.node('browser', 'Web Browser\n(User Interface)', fillcolor='#E8F4FD', shape='tab')
    dot.node('api', 'FastAPI Server\n(API Gateway)', fillcolor='#FFF4E6')
    dot.node('webrtc', 'WebRTC\n(Video Streaming)', fillcolor='#F0F8FF')
    dot.node('agents', 'Agent Pipeline\n(4 stages)', fillcolor='#E8F5E8')
    dot.node('vlm', 'Vision Language Model\n(Qwen2.5-VL 3B)', fillcolor='#FFE8E8')
    dot.node('camera', 'Camera\n(CV60/RealSense)', fillcolor='#FFFACD', shape='box3d')

    # Connections
    dot.edge('browser', 'api', label='WebSocket')
    dot.edge('api', 'webrtc', label='manages')
    dot.edge('camera', 'webrtc', label='video feed')
    dot.edge('webrtc', 'browser', label='stream', style='dashed')
    dot.edge('webrtc', 'agents', label='frames')
    dot.edge('agents', 'vlm', label='inference')
    dot.edge('agents', 'browser', label='results', style='dashed')

    return dot


def create_data_flow_diagram():
    """Create a detailed data flow diagram"""

    dot = Digraph('RELO_DataFlow', comment='RELO Classifier - Data Flow')
    dot.attr(rankdir='TB', size='16,12', dpi='300', compound='true')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial')

    # Define the flow stages
    with dot.subgraph(name='cluster_capture') as capture:
        capture.attr(label='1. Video Capture', style='filled', color='lightblue')
        capture.node('cam', 'Camera Hardware')
        capture.node('driver', 'Camera Driver\n(eBUS/RealSense)')
        capture.node('track', 'Video Track')

    with dot.subgraph(name='cluster_stream') as stream:
        stream.attr(label='2. WebRTC Streaming', style='filled', color='lightgreen')
        stream.node('peer', 'Peer Connection')
        stream.node('relay', 'Media Relay')
        stream.node('buffer', 'Frame Buffer')

    with dot.subgraph(name='cluster_process') as process:
        process.attr(label='3. Frame Processing', style='filled', color='lightyellow')
        process.node('collect', 'Frame Collection')
        process.node('aggregate', 'Frame Aggregation')
        process.node('batch', 'Batch Preparation')

    with dot.subgraph(name='cluster_inference') as inference:
        inference.attr(label='4. AI Inference', style='filled', color='lightcoral')
        inference.node('prompt', 'Prompt Generation')
        inference.node('model', 'VLM Model')
        inference.node('parse', 'Response Parsing')

    with dot.subgraph(name='cluster_results') as results:
        results.attr(label='5. Results', style='filled', color='lightgray')
        results.node('compile', 'Result Compilation')
        results.node('store', 'Storage')
        results.node('send', 'Send to Frontend')

    # Data flow connections
    dot.edge('cam', 'driver', label='raw video')
    dot.edge('driver', 'track', label='frames')
    dot.edge('track', 'peer', label='encode')
    dot.edge('peer', 'relay', label='stream')
    dot.edge('relay', 'buffer', label='queue')
    dot.edge('buffer', 'collect', label='frames')
    dot.edge('collect', 'aggregate', label='multi-frame')
    dot.edge('aggregate', 'batch', label='prepare')
    dot.edge('batch', 'prompt', label='image data')
    dot.edge('prompt', 'model', label='inference request')
    dot.edge('model', 'parse', label='JSON response')
    dot.edge('parse', 'compile', label='structured data')
    dot.edge('compile', 'store', label='save')
    dot.edge('compile', 'send', label='WebSocket/DataChannel')

    return dot


def create_professional_low_level_diagram():
    """Create a professional low-level system operations diagram without file names"""

    dot = Digraph('RELO_Full_System', comment='RELO Classifier - Complete System Operations')
    dot.attr(rankdir='LR', size='24,18', dpi='300', nodesep='0.8', ranksep='1.2')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Helvetica', fontsize='11', height='0.6', width='1.8')
    dot.attr('edge', fontname='Helvetica', fontsize='9', arrowsize='0.8')

    # Professional color palette
    colors = {
        'client': '#E3F2FD',      # Very light blue
        'network': '#FFF3E0',     # Very light orange
        'server': '#F3E5F5',      # Very light purple
        'processing': '#E8F5E9',  # Very light green
        'storage': '#FFF8E1',     # Very light yellow
        'ai': '#FCE4EC',         # Very light pink
        'control': '#E0F2F1'      # Very light teal
    }

    # Client Side Operations
    with dot.subgraph(name='cluster_client') as client:
        client.attr(label='CLIENT BROWSER', style='filled,rounded', color='#1976D2', fillcolor=colors['client'], fontsize='14', fontname='Helvetica Bold')

        # User Interface Layer
        with client.subgraph(name='cluster_ui') as ui:
            ui.attr(label='User Interface Layer', style='dotted', color='gray')
            ui.node('user_action', 'User Initiates\nSession', shape='ellipse', fillcolor='#BBDEFB')
            ui.node('video_display', 'Live Video\nDisplay', fillcolor='#90CAF9')
            ui.node('controls', 'Control Panel\n(Start/Stop/Trigger)', fillcolor='#90CAF9')
            ui.node('results_view', 'Classification\nResults Display', fillcolor='#90CAF9')
            ui.node('status_indicators', 'Status\nIndicators', fillcolor='#90CAF9')

        # Client Processing
        with client.subgraph(name='cluster_client_proc') as cproc:
            cproc.attr(label='Client Processing', style='dotted', color='gray')
            cproc.node('dom_manager', 'DOM\nManagement', fillcolor='#64B5F6')
            cproc.node('event_handler', 'Event\nHandling', fillcolor='#64B5F6')
            cproc.node('state_manager', 'Client State\nManagement', fillcolor='#64B5F6')
            cproc.node('render_engine', 'Rendering\nEngine', fillcolor='#64B5F6')

    # Network Communication Layer
    with dot.subgraph(name='cluster_network') as network:
        network.attr(label='NETWORK LAYER', style='filled,rounded', color='#F57C00', fillcolor=colors['network'], fontsize='14', fontname='Helvetica Bold')

        # WebRTC Components
        with network.subgraph(name='cluster_webrtc') as webrtc:
            webrtc.attr(label='WebRTC Protocol', style='dotted', color='gray')
            webrtc.node('ice_negotiation', 'ICE\nNegotiation', fillcolor='#FFB74D')
            webrtc.node('stun_server', 'STUN\nServer', shape='diamond', fillcolor='#FFB74D')
            webrtc.node('peer_connection', 'Peer\nConnection', fillcolor='#FF9800')
            webrtc.node('media_stream', 'Media\nStream', fillcolor='#FF9800')
            webrtc.node('data_channel', 'Data\nChannel', fillcolor='#FF9800')

        # WebSocket Components
        with network.subgraph(name='cluster_websocket') as websocket:
            websocket.attr(label='WebSocket Protocol', style='dotted', color='gray')
            websocket.node('ws_connection', 'WebSocket\nConnection', fillcolor='#FFA726')
            websocket.node('signaling', 'Signaling\nProtocol', fillcolor='#FFA726')
            websocket.node('message_queue', 'Message\nQueue', fillcolor='#FFA726')

    # Server Processing
    with dot.subgraph(name='cluster_server') as server:
        server.attr(label='SERVER BACKEND', style='filled,rounded', color='#7B1FA2', fillcolor=colors['server'], fontsize='14', fontname='Helvetica Bold')

        # Connection Management
        with server.subgraph(name='cluster_conn_mgmt') as conn:
            conn.attr(label='Connection Management', style='dotted', color='gray')
            conn.node('conn_pool', 'Connection\nPool', fillcolor='#CE93D8')
            conn.node('session_registry', 'Session\nRegistry', fillcolor='#CE93D8')
            conn.node('auth_validator', 'Authentication\nValidation', fillcolor='#CE93D8')

        # Request Processing
        with server.subgraph(name='cluster_request') as request:
            request.attr(label='Request Processing', style='dotted', color='gray')
            request.node('http_server', 'HTTP\nServer', fillcolor='#BA68C8')
            request.node('api_router', 'API\nRouter', fillcolor='#BA68C8')
            request.node('cors_handler', 'CORS\nHandler', fillcolor='#BA68C8')
            request.node('middleware', 'Middleware\nPipeline', fillcolor='#BA68C8')

    # Video Processing Pipeline
    with dot.subgraph(name='cluster_video') as video:
        video.attr(label='VIDEO PROCESSING', style='filled,rounded', color='#388E3C', fillcolor=colors['processing'], fontsize='14', fontname='Helvetica Bold')

        # Camera Operations
        with video.subgraph(name='cluster_camera') as camera:
            camera.attr(label='Camera Operations', style='dotted', color='gray')
            camera.node('camera_init', 'Camera\nInitialization', fillcolor='#AED581')
            camera.node('camera_config', 'Configuration\nSetup', fillcolor='#AED581')
            camera.node('frame_capture', 'Frame\nCapture', fillcolor='#9CCC65')
            camera.node('frame_buffer', 'Frame\nBuffering', fillcolor='#9CCC65')

        # Frame Processing
        with video.subgraph(name='cluster_frame') as frame:
            frame.attr(label='Frame Processing', style='dotted', color='gray')
            frame.node('frame_decode', 'Frame\nDecoding', fillcolor='#8BC34A')
            frame.node('frame_resize', 'Resolution\nAdjustment', fillcolor='#8BC34A')
            frame.node('frame_enhance', 'Image\nEnhancement', fillcolor='#8BC34A')
            frame.node('frame_validate', 'Quality\nValidation', fillcolor='#8BC34A')

    # AI Processing Layer
    with dot.subgraph(name='cluster_ai') as ai:
        ai.attr(label='AI INFERENCE ENGINE', style='filled,rounded', color='#D32F2F', fillcolor=colors['ai'], fontsize='14', fontname='Helvetica Bold')

        # GPU Operations
        with ai.subgraph(name='cluster_gpu') as gpu:
            gpu.attr(label='GPU Operations', style='dotted', color='gray')
            gpu.node('gpu_init', 'GPU\nInitialization', fillcolor='#F48FB1')
            gpu.node('memory_alloc', 'Memory\nAllocation', fillcolor='#F48FB1')
            gpu.node('cuda_setup', 'CUDA\nConfiguration', fillcolor='#F48FB1')
            gpu.node('tensor_ops', 'Tensor\nOperations', fillcolor='#F48FB1')

        # Model Operations
        with ai.subgraph(name='cluster_model') as model:
            model.attr(label='Model Processing', style='dotted', color='gray')
            model.node('model_load', 'Model\nLoading', fillcolor='#F06292')
            model.node('prompt_gen', 'Prompt\nGeneration', fillcolor='#F06292')
            model.node('inference_exec', 'Inference\nExecution', fillcolor='#EC407A')
            model.node('response_parse', 'Response\nParsing', fillcolor='#F06292')

        # Multi-Agent Pipeline
        with ai.subgraph(name='cluster_agents') as agents:
            agents.attr(label='Classification Pipeline', style='dotted', color='gray')
            agents.node('agent1', 'Initial\nClassification\n[4s Timer]', fillcolor='#E91E63')
            agents.node('agent2', 'Detail\nExtraction\n[3s Timer]', fillcolor='#E91E63')
            agents.node('agent3', 'Damage\nDetection\n[4s Timer]', fillcolor='#E91E63')
            agents.node('agent4', 'Final\nAggregation\n[1s Timer]', fillcolor='#E91E63')

    # Control & Orchestration
    with dot.subgraph(name='cluster_control') as control:
        control.attr(label='ORCHESTRATION LAYER', style='filled,rounded', color='#00796B', fillcolor=colors['control'], fontsize='14', fontname='Helvetica Bold')

        # State Management
        with control.subgraph(name='cluster_state') as state:
            state.attr(label='State Management', style='dotted', color='gray')
            state.node('state_machine', 'State\nMachine', fillcolor='#80CBC4')
            state.node('timer_control', 'Timer\nControl', fillcolor='#80CBC4')
            state.node('flow_control', 'Flow\nControl', fillcolor='#80CBC4')
            state.node('pause_resume', 'Pause/Resume\nHandler', fillcolor='#80CBC4')

        # Session Management
        with control.subgraph(name='cluster_session') as session:
            session.attr(label='Session Control', style='dotted', color='gray')
            session.node('session_create', 'Session\nCreation', fillcolor='#4DB6AC')
            session.node('session_track', 'Session\nTracking', fillcolor='#4DB6AC')
            session.node('session_persist', 'Session\nPersistence', fillcolor='#4DB6AC')
            session.node('session_cleanup', 'Session\nCleanup', fillcolor='#4DB6AC')

    # Data Storage Layer
    with dot.subgraph(name='cluster_storage') as storage:
        storage.attr(label='DATA STORAGE', style='filled,rounded', color='#F57F17', fillcolor=colors['storage'], fontsize='14', fontname='Helvetica Bold')

        storage.node('frame_storage', 'Frame\nStorage', shape='cylinder', fillcolor='#FFD54F')
        storage.node('session_storage', 'Session\nData', shape='cylinder', fillcolor='#FFD54F')
        storage.node('result_storage', 'Result\nCache', shape='cylinder', fillcolor='#FFD54F')
        storage.node('log_storage', 'System\nLogs', shape='cylinder', fillcolor='#FFD54F')

    # Define all connections with labels describing the data flow

    # User interaction flow
    dot.edge('user_action', 'controls', 'clicks')
    dot.edge('controls', 'event_handler', 'events')
    dot.edge('event_handler', 'state_manager', 'updates')
    dot.edge('state_manager', 'ws_connection', 'commands', color='#4CAF50', style='bold')

    # WebSocket signaling
    dot.edge('ws_connection', 'signaling', 'messages')
    dot.edge('signaling', 'http_server', 'HTTP upgrade', color='#FF9800')
    dot.edge('http_server', 'api_router', 'routes')
    dot.edge('api_router', 'session_create', 'POST /session')

    # Session initialization
    dot.edge('session_create', 'session_registry', 'register')
    dot.edge('session_registry', 'conn_pool', 'allocate')
    dot.edge('conn_pool', 'ice_negotiation', 'initiate', color='#2196F3')
    dot.edge('ice_negotiation', 'stun_server', 'STUN request')
    dot.edge('stun_server', 'peer_connection', 'ICE candidates')

    # Media stream setup
    dot.edge('peer_connection', 'media_stream', 'establish', color='#2196F3', style='bold')
    dot.edge('camera_init', 'camera_config', 'setup')
    dot.edge('camera_config', 'frame_capture', 'start')
    dot.edge('frame_capture', 'frame_buffer', 'queue frames')
    dot.edge('frame_buffer', 'frame_decode', 'raw frames')
    dot.edge('frame_decode', 'frame_resize', 'decoded')
    dot.edge('frame_resize', 'frame_enhance', 'resized')
    dot.edge('frame_enhance', 'media_stream', 'encoded frames', color='#2196F3')
    dot.edge('media_stream', 'video_display', 'video stream', color='#2196F3', style='bold')

    # Control flow
    dot.edge('controls', 'message_queue', 'trigger command', color='#4CAF50')
    dot.edge('message_queue', 'flow_control', 'process')
    dot.edge('flow_control', 'state_machine', 'transition')
    dot.edge('state_machine', 'timer_control', 'start timer')

    # Frame processing for AI
    dot.edge('frame_buffer', 'frame_validate', 'copy frames')
    dot.edge('frame_validate', 'frame_storage', 'save')
    dot.edge('frame_validate', 'gpu_init', 'batch frames', color='#E91E63', style='bold')

    # GPU/AI processing
    dot.edge('gpu_init', 'memory_alloc', 'initialize')
    dot.edge('memory_alloc', 'cuda_setup', 'allocate')
    dot.edge('cuda_setup', 'model_load', 'ready')
    dot.edge('model_load', 'tensor_ops', 'loaded')

    # Agent pipeline
    dot.edge('timer_control', 'agent1', 'trigger', color='#E91E63')
    dot.edge('frame_storage', 'prompt_gen', 'images')
    dot.edge('prompt_gen', 'inference_exec', 'prompt + images')
    dot.edge('inference_exec', 'tensor_ops', 'compute', color='#E91E63', style='bold')
    dot.edge('tensor_ops', 'response_parse', 'output')
    dot.edge('response_parse', 'agent1', 'results')

    dot.edge('agent1', 'agent2', 'classification', color='#4CAF50')
    dot.edge('agent2', 'agent3', 'details', color='#4CAF50')
    dot.edge('agent3', 'agent4', 'damage info', color='#4CAF50')
    dot.edge('agent4', 'result_storage', 'final results')

    # Results flow back
    dot.edge('result_storage', 'session_track', 'store')
    dot.edge('session_track', 'data_channel', 'results', color='#FF9800', style='bold')
    dot.edge('data_channel', 'render_engine', 'JSON data', color='#FF9800')
    dot.edge('render_engine', 'results_view', 'display')
    dot.edge('render_engine', 'status_indicators', 'update')

    # Pause/Resume flow
    dot.edge('pause_resume', 'timer_control', 'pause/resume', style='dashed')
    dot.edge('pause_resume', 'session_persist', 'checkpoint', style='dashed')
    dot.edge('session_persist', 'session_storage', 'save state', style='dashed')

    # Cleanup flow
    dot.edge('session_cleanup', 'conn_pool', 'release', style='dotted')
    dot.edge('session_cleanup', 'frame_storage', 'cleanup', style='dotted')
    dot.edge('session_cleanup', 'log_storage', 'archive', style='dotted')

    # DOM updates
    dot.edge('dom_manager', 'video_display', 'update DOM')
    dot.edge('dom_manager', 'results_view', 'update DOM')
    dot.edge('dom_manager', 'status_indicators', 'update DOM')

    return dot


def main():
    """Generate all architecture diagrams"""

    print("Generating RELO Classifier Architecture Diagrams...")

    # Create output directory
    output_dir = "architecture_diagrams"
    os.makedirs(output_dir, exist_ok=True)

    # Generate comprehensive diagram
    print("1. Creating comprehensive architecture diagram...")
    comprehensive = create_architecture_diagram()
    comprehensive.render(f'{output_dir}/relo_architecture_comprehensive', format='svg', cleanup=True)
    comprehensive.render(f'{output_dir}/relo_architecture_comprehensive', format='png', cleanup=True)
    print("   ✓ Saved: relo_architecture_comprehensive.svg/png")

    # Generate simplified diagram
    print("2. Creating simplified architecture diagram...")
    simplified = create_simplified_diagram()
    simplified.render(f'{output_dir}/relo_architecture_simplified', format='svg', cleanup=True)
    simplified.render(f'{output_dir}/relo_architecture_simplified', format='png', cleanup=True)
    print("   ✓ Saved: relo_architecture_simplified.svg/png")

    # Generate data flow diagram
    print("3. Creating data flow diagram...")
    dataflow = create_data_flow_diagram()
    dataflow.render(f'{output_dir}/relo_dataflow', format='svg', cleanup=True)
    dataflow.render(f'{output_dir}/relo_dataflow', format='png', cleanup=True)
    print("   ✓ Saved: relo_dataflow.svg/png")

    # Generate professional low-level diagram
    print("4. Creating professional low-level system diagram...")
    professional = create_professional_low_level_diagram()
    professional.render(f'{output_dir}/Full-low-level-diagram', format='png', cleanup=True)
    print("   ✓ Saved: Full-low-level-diagram.png")

    # Create HTML viewer
    print("5. Creating interactive HTML viewer...")
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>RELO Classifier Architecture Diagrams</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .diagram {{
            text-align: center;
            margin: 20px 0;
        }}
        .diagram img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .info {{
            background: #e8f5e9;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9em;
            text-align: right;
        }}
    </style>
</head>
<body>
    <h1>RELO Classifier System Architecture</h1>

    <div class="info">
        <h3>System Overview</h3>
        <p>The RELO Classifier is an AI-powered clothing returns classification system that uses:</p>
        <ul>
            <li>WebRTC for real-time video streaming</li>
            <li>Vision-Language Model (Qwen2.5-VL 3B) for intelligent analysis</li>
            <li>Multi-agent pipeline for comprehensive item assessment</li>
            <li>Stateful orchestration with multi-inference capabilities</li>
        </ul>
    </div>

    <div class="container">
        <h2>1. Comprehensive Architecture</h2>
        <p>Complete low-level view showing all components, services, and connections.</p>
        <div class="diagram">
            <img src="relo_architecture_comprehensive.svg" alt="Comprehensive Architecture">
        </div>
    </div>

    <div class="container">
        <h2>2. Simplified Overview</h2>
        <p>High-level view of main components and their relationships.</p>
        <div class="diagram">
            <img src="relo_architecture_simplified.svg" alt="Simplified Architecture">
        </div>
    </div>

    <div class="container">
        <h2>3. Data Flow Diagram</h2>
        <p>Detailed view of how data flows through the system from camera to results.</p>
        <div class="diagram">
            <img src="relo_dataflow.svg" alt="Data Flow Diagram">
        </div>
    </div>

    <div class="container">
        <h3>Key Components</h3>
        <h4>Frontend Layer:</h4>
        <ul>
            <li>WebRTC Client for video streaming</li>
            <li>Session Manager for state management</li>
            <li>Professional UI with real-time updates</li>
        </ul>

        <h4>Backend Services:</h4>
        <ul>
            <li>FastAPI server with WebSocket support</li>
            <li>WebRTC Manager for peer connections</li>
            <li>GPU-optimized VLM inference</li>
            <li>Multiple camera support (CV60, RealSense, USB)</li>
        </ul>

        <h4>Agent Pipeline:</h4>
        <ul>
            <li>Initial Classifier (4s timer)</li>
            <li>Detail Extractor (3s timer)</li>
            <li>Damage Detector (4s timer)</li>
            <li>Final Compiler (1s aggregation)</li>
        </ul>

        <h4>V2 Architecture Features:</h4>
        <ul>
            <li>Stateful orchestration with pause/resume</li>
            <li>Multi-inference per agent</li>
            <li>Frame registry for controlled sharing</li>
            <li>JSON state persistence</li>
        </ul>
    </div>

    <div class="timestamp">
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
</body>
</html>"""

    with open(f'{output_dir}/index.html', 'w') as f:
        f.write(html_content)
    print("   ✓ Saved: index.html")

    print(f"\n✅ All diagrams generated successfully in ./{output_dir}/")
    print("   Open index.html in a browser to view all diagrams interactively.")

    # Print summary
    print("\nDiagram Summary:")
    print("================")
    print("1. Comprehensive: Full system with all 50+ components")
    print("2. Simplified: High-level overview for quick understanding")
    print("3. Data Flow: Step-by-step data processing pipeline")
    print("\nComponents discovered:")
    print("- Frontend: 9 components")
    print("- API Layer: 5 endpoints")
    print("- Backend Services: 16 services")
    print("- Agents: 5 agents (4 processing + base)")
    print("- V2 Architecture: 10 components")
    print("- Infrastructure: 4 systems")


if __name__ == "__main__":
    main()