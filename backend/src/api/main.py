"""Main FastAPI application."""
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables FIRST before any other imports
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from api.routes import session, health
from api.websocket import websocket_endpoint
from api.websocket_v2 import websocket_v2_endpoint
from services import SessionManager, WebRTCManager
from services.vlm_singleton import vlm_singleton, get_vlm_status
from services.ollama_optimizer import optimize_ollama_at_startup
from services.gpu_initializer import ensure_gpu_ready, gpu_initializer

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Force DEBUG level for troubleshooting
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler('../backend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress aiortc verbose logging
logging.getLogger('aiortc').setLevel(logging.WARNING)
logging.getLogger('aioice').setLevel(logging.WARNING)
logging.getLogger('aiortc.rtcpeerconnection').setLevel(logging.WARNING)
logging.getLogger('aiortc.rtcdatachannel').setLevel(logging.WARNING)

# Global service instances
session_manager = None
webrtc_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global session_manager, webrtc_manager
    
    # Startup
    logger.info("🚀 Starting Returns Classifier API with Unified VLM Service")
    
    try:
        # Initialize GPU FIRST - fix CUDA context issues
        logger.info("🎮 Initializing GPU and CUDA context...")
        gpu_ready = await ensure_gpu_ready()
        if gpu_ready:
            logger.info("✅ GPU initialized successfully - CUDA context ready")
            gpu_status = gpu_initializer.get_gpu_status()
            logger.info(f"   GPU: {gpu_status.get('gpu_name', 'Unknown')}")
            logger.info(f"   Memory: {gpu_status.get('free_memory_mb', 0)}MB free / {gpu_status.get('total_memory_mb', 0)}MB total")
        else:
            logger.warning("⚠️ GPU initialization failed - will use CPU fallback")
        
        logger.info("Initializing SessionManager...")
        session_manager = SessionManager()
        logger.info(f"SessionManager initialized: {session_manager}")
        
        logger.info("Initializing WebRTCManager...")
        webrtc_manager = WebRTCManager()
        logger.info(f"WebRTCManager initialized: {webrtc_manager}")

        # Start periodic cleanup task for memory management
        await webrtc_manager.start_periodic_cleanup()
        logger.info("✅ Started WebRTC periodic cleanup task")
        
        # NOTE: VLM initialization moved to frontend control
        # The VLM singleton will be initialized once during frontend initialization
        logger.info("📝 VLM will be initialized during frontend initialization sequence")
        logger.info("   - Single initialization point for entire application")
        logger.info("   - Progress tracking via frontend UI")
        logger.info("   - No redundant warmups")
        
        # Apply Ollama optimizations
        logger.info("⚡ Applying SOTA Ollama optimizations...")
        ollama_result = await optimize_ollama_at_startup()
        
        if ollama_result.get("configuration", {}).get("status") == "success":
            logger.info("✅ Ollama optimized successfully")
            if ollama_result.get("flash_attention"):
                logger.info("   - Flash Attention enabled")
            if ollama_result.get("benchmark"):
                avg_time = ollama_result["benchmark"].get("avg_time", 0)
                logger.info(f"   - Average inference time: {avg_time:.2f}s")
        else:
            logger.warning("⚠️ Ollama optimization incomplete")
        
        logger.info("All managers initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing managers: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Returns Classifier API")

    # Stop periodic cleanup task
    if webrtc_manager:
        await webrtc_manager.stop_periodic_cleanup()
        logger.info("Stopped WebRTC periodic cleanup task")
    
    # Clean up WebRTC connections
    if webrtc_manager:
        for session_id in list(webrtc_manager.peer_connections.keys()):
            await webrtc_manager.close_connection(session_id)


# Create FastAPI app
app = FastAPI(
    title="Returns Classifier API",
    description="Intelligent clothing returns classification system",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(session.router, prefix="/api/session", tags=["session"])
app.include_router(health.router, prefix="/api", tags=["health"])

# WebSocket endpoint
app.add_websocket_route("/ws/stream", websocket_endpoint)
# V2 WebSocket endpoint with stateful orchestration
app.add_websocket_route("/ws/v2/stream", websocket_v2_endpoint)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Returns Classifier API",
        "version": "1.0.0",
        "status": "running"
    }

# Camera status endpoint
@app.get("/api/camera/status")
async def camera_status():
    """Check camera availability and status."""
    from datetime import datetime
    try:
        import pyrealsense2 as rs
        
        # Check for RealSense devices
        ctx = rs.context()
        devices = ctx.devices
        
        if len(devices) > 0:
            device = devices[0]
            serial = device.get_info(rs.camera_info.serial_number)
            name = device.get_info(rs.camera_info.name)
            
            return {
                "available": True,
                "device_count": len(devices),
                "device_name": name,
                "device_serial": serial,
                "status": "ready",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "available": False,
                "device_count": 0,
                "device_name": None,
                "device_serial": None,
                "status": "no_device",
                "message": "No RealSense camera detected, will use test pattern",
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Camera status check failed: {e}")
        return {
            "available": False,
            "device_count": 0,
            "device_name": None,
            "device_serial": None,
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

# VLM Status endpoints
@app.get("/api/vlm/status")
async def vlm_status():
    """Get VLM service status and performance metrics."""
    status = vlm_singleton.get_status()
    return {
        **status,
        "timestamp": datetime.now().isoformat(),
        "system_specs": {
            "gpu": "NVIDIA RTX A1000 (8GB VRAM)",
            "cpu": "Intel i9-13900TE (32 cores)",
            "optimization": "Unified VLM with single initialization"
        }
    }

@app.get("/api/vlm/health")
async def vlm_health():
    """Check VLM service health and Ollama connectivity."""
    is_ready = vlm_singleton.is_ready()
    status = vlm_singleton.get_status()
    
    return {
        "healthy": is_ready,
        "ollama_connected": is_ready,
        "model_available": is_ready,
        "state": status.get("state"),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/vlm/warmup")
async def trigger_vlm_warmup():
    """Trigger VLM warmup (called from frontend initialization)."""
    try:
        # Progress callback for real-time updates
        progress_updates = []
        
        async def frontend_progress_callback(message: str, progress: int):
            progress_updates.append({"message": message, "progress": progress})
            logger.info(f"VLM Frontend [{progress:3d}%]: {message}")
        
        # Use singleton for single initialization
        warmup_result = await vlm_singleton.initialize_once(progress_callback=frontend_progress_callback)
        
        return {
            "success": True,
            "result": warmup_result,
            "progress_updates": progress_updates,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"VLM warmup failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"VLM warmup failed: {str(e)}"}
        )

@app.get("/api/vlm/quick-check")
async def vlm_quick_check():
    """Quick VLM readiness check without full warmup."""
    is_ready = vlm_singleton.is_ready()
    status = vlm_singleton.get_status()
    
    return {
        "ready": is_ready,
        "state": status.get("state"),
        "status": status,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/vlm/performance")
async def vlm_performance_metrics():
    """Get VLM performance metrics and optimization status."""
    status = vlm_singleton.get_status()
    
    if not vlm_singleton.is_ready():
        return JSONResponse(
            status_code=503,
            content={"detail": "VLM not initialized"}
        )
    
    return {
        "state": status.get("state"),
        "initialization_duration": status.get("duration"),
        "gpu_info": status.get("gpu_info"),
        "model_info": status.get("model_info"),
        "warmup_results": status.get("warmup_results"),
        "timestamp": datetime.now().isoformat(),
        "system_info": {
            "gpu": "NVIDIA RTX A1000 (8GB VRAM)",
            "cpu": "Intel i9-13900TE (32 cores)",
            "optimization": "Unified VLM with single initialization"
        }
    }

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions."""
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


def get_session_manager() -> SessionManager:
    """Get session manager instance."""
    return session_manager


def get_webrtc_manager() -> WebRTCManager:
    """Get WebRTC manager instance."""
    return webrtc_manager


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=True,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )