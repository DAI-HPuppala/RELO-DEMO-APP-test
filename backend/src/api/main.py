"""Main FastAPI application."""
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
import logging
from logging.handlers import TimedRotatingFileHandler
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
from services import SessionManager, WebRTCManager
from services.vlm_singleton import vlm_singleton, get_vlm_status
from services.ollama_optimizer import optimize_ollama_at_startup
from services.gpu_initializer import ensure_gpu_ready, gpu_initializer

# Configure logging with time-based rotation
# Production-ready hybrid approach:
# - New session directory on each app restart (tracks app lifecycle)
# - Time-based rotation within session (handles long-running apps)
# - NEVER delete logs (infinite retention with backupCount=0)

base_log_directory = Path(__file__).parent.parent.parent / os.getenv("LOG_DIRECTORY", "logs")
base_log_directory.mkdir(parents=True, exist_ok=True)

# Create session-specific directory with startup timestamp
session_start_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
session_log_directory = base_log_directory / f"session_{session_start_timestamp}"
session_log_directory.mkdir(parents=True, exist_ok=True)

# Get rotation settings from environment
rotation_interval_hours = int(os.getenv("LOG_ROTATION_INTERVAL", "5"))
# backupCount=0 means NEVER delete old logs (infinite retention)
backup_count = int(os.getenv("LOG_ROTATION_BACKUP_COUNT", "0"))
log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()

# Create log file path in session directory
log_file = session_log_directory / "relo_backend.log"

# Create time-based rotating file handler
# 'H' means rotate every N hours, where N is the interval
# backupCount=0 means keep ALL rotated logs forever (never delete)
file_handler = TimedRotatingFileHandler(
    filename=str(log_file),
    when='H',  # Rotate by hours
    interval=rotation_interval_hours,
    backupCount=backup_count,  # 0 = never delete
    encoding='utf-8',
    utc=False  # Use local time
)
file_handler.suffix = "%Y-%m-%d_%H-%M"  # Backup filename: relo_backend.log.2025-10-16_14-00

# Create console handler
console_handler = logging.StreamHandler()

# Set format for both handlers
log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(log_format)
console_handler.setFormatter(log_format)

# Configure root logger
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger(__name__)
logger.info(f"📝 Logging: {session_log_directory} (rotates every {rotation_interval_hours}h, never deleted)")

# Configure logging for all services - COMPREHENSIVE LOGGING
# Enable all important service logs
logging.getLogger('api.websocket').setLevel(logging.INFO)
logging.getLogger('services.webrtc_manager').setLevel(logging.INFO)
logging.getLogger('services.cv60_camera').setLevel(logging.INFO)
logging.getLogger('services.vlm_singleton').setLevel(logging.INFO)

# Enable aiortc logging for offline WebRTC troubleshooting
logging.getLogger('aiortc').setLevel(logging.INFO)  # INFO instead of DEBUG to avoid packet spam
logging.getLogger('aioice').setLevel(logging.INFO)  # INFO for ICE connection logs
logging.getLogger('aiortc.rtcpeerconnection').setLevel(logging.INFO)  # INFO for connection state
logging.getLogger('aiortc.rtcdatachannel').setLevel(logging.INFO)
logging.getLogger('aiortc.rtcrtpsender').setLevel(logging.WARNING)  # Suppress RTP packet logs

# Global service instances
session_manager = None
webrtc_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global session_manager, webrtc_manager
    
    # Startup
    logger.info("🚀 Starting Returns Classifier API")

    try:
        model_provider = os.getenv("MODEL_PROVIDER", "ollama").lower()
        logger.info(f"Model Provider: {model_provider}")

        # Initialize GPU for Ollama
        if model_provider == "ollama":
            gpu_ready = await ensure_gpu_ready()
            if gpu_ready:
                gpu_status = gpu_initializer.get_gpu_status()
                logger.info(f"GPU: {gpu_status.get('gpu_name', 'Unknown')} ({gpu_status.get('free_memory_mb', 0)}MB free)")
            else:
                logger.warning("GPU initialization failed - using CPU fallback")

        # Initialize managers
        session_manager = SessionManager()
        webrtc_manager = WebRTCManager()

        # Start cleanup tasks
        await webrtc_manager.start_periodic_cleanup()
        from api.websocket import start_cleanup_task
        start_cleanup_task()

        # Ollama optimizations
        if model_provider == "ollama":
            ollama_result = await optimize_ollama_at_startup()
            if ollama_result.get("configuration", {}).get("status") == "success":
                logger.info("Ollama optimizations applied")
            else:
                logger.warning("Ollama optimization incomplete")

        logger.info("✅ All services initialized")
    except Exception as e:
        logger.error(f"Error initializing managers: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Returns Classifier API")

    if webrtc_manager:
        await webrtc_manager.stop_periodic_cleanup()
        for session_id in list(webrtc_manager.peer_connections.keys()):
            await webrtc_manager.close_connection(session_id)

    logger.info("Shutdown complete")


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

# Camera Configuration and Control endpoints
@app.get("/api/camera/settings")
async def get_camera_settings():
    """Get current camera settings (from persistent storage)"""
    from services.camera_settings_manager import get_camera_settings_manager
    manager = get_camera_settings_manager()
    return manager.get_settings()

@app.post("/api/camera/settings")
async def update_camera_settings(settings: dict, persist: bool = True):
    """
    Update camera settings

    Args:
        settings: Settings dictionary to update
        persist: If True, save to disk permanently (default: True)
    """
    from services.camera_settings_manager import get_camera_settings_manager
    manager = get_camera_settings_manager()
    success = manager.save_settings(settings, make_persistent=persist)
    return {
        "success": success,
        "message": "Settings saved permanently to disk" if persist else "Settings updated (session only)",
        "settings": manager.get_settings()
    }

@app.post("/api/camera/measure-height")
async def measure_camera_height():
    """
    Measure camera height using stereo depth.
    Samples center ROI of depth map to find distance to ground/bench.
    """
    import numpy as np
    from services.oakd_camera import OakDCameraService
    from services.camera_settings_manager import get_camera_settings_manager

    try:
        camera_service = OakDCameraService()
        depth_frame = camera_service.capture_depth_frame()

        if depth_frame is None:
            logger.error("Camera height measurement failed: no depth frame")
            return {"success": False, "error": "Failed to capture depth frame"}

        h, w = depth_frame.shape
        center_roi = depth_frame[int(h*0.4):int(h*0.6), int(w*0.4):int(w*0.6)]
        valid_depths = center_roi[center_roi > 0]

        if len(valid_depths) == 0:
            logger.error("Camera height measurement failed: no valid depth data")
            return {"success": False, "error": "No valid depth measurements"}

        median_depth_mm = float(np.median(valid_depths))
        height_cm = median_depth_mm / 10.0

        fov_h_deg, fov_v_deg = 69, 55
        ground_width_cm = 2 * height_cm * np.tan(np.radians(fov_h_deg/2))
        ground_height_cm = 2 * height_cm * np.tan(np.radians(fov_v_deg/2))

        ground_coverage = {"width_cm": round(ground_width_cm, 2), "height_cm": round(ground_height_cm, 2)}

        manager = get_camera_settings_manager()
        manager.save_height_measurement(height_cm, ground_coverage, make_persistent=True)

        logger.info(f"Camera height measured: {height_cm:.1f}cm, coverage: {ground_width_cm:.1f}×{ground_height_cm:.1f}cm")

        return {
            "success": True,
            "camera_height_cm": round(height_cm, 2),
            "ground_coverage": ground_coverage,
            "sensor_resolution": {
                "width": 1796,
                "height": 2160
            },
            "depth_stats": {
                "median_mm": round(median_depth_mm, 2),
                "valid_pixels": int(len(valid_depths)),
                "roi_size": f"{center_roi.shape[0]}x{center_roi.shape[1]}"
            }
        }
    except Exception as e:
        logger.error(f"Error measuring camera height: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/api/camera/roi")
async def set_camera_roi(x_min: float, y_min: float, x_max: float, y_max: float, enabled: bool = True, persist: bool = True):
    """
    Set camera ROI (Region of Interest)

    Args:
        x_min, y_min, x_max, y_max: Normalized coordinates (0.0-1.0)
        enabled: Enable/disable ROI
        persist: Save permanently (default: True)
    """
    from services.camera_settings_manager import get_camera_settings_manager

    # Validate coordinates
    if not (0.0 <= x_min < x_max <= 1.0):
        return {"success": False, "error": "Invalid x coordinates"}
    if not (0.0 <= y_min < y_max <= 1.0):
        return {"success": False, "error": "Invalid y coordinates"}

    manager = get_camera_settings_manager()
    success = manager.save_roi(x_min, y_min, x_max, y_max, enabled=enabled, make_persistent=persist)

    return {
        "success": success,
        "message": "ROI saved permanently" if persist else "ROI updated (session only)",
        "roi": manager.get_roi_settings()
    }

@app.get("/api/camera/presets")
async def list_camera_presets():
    """List available camera configuration presets"""
    from services.camera_settings_manager import get_camera_settings_manager
    manager = get_camera_settings_manager()
    presets = manager.list_presets()
    return {"presets": presets}

@app.get("/api/camera/presets/{preset_name}")
async def load_camera_preset(preset_name: str):
    """Load a specific camera preset"""
    from services.camera_settings_manager import get_camera_settings_manager
    manager = get_camera_settings_manager()
    preset = manager.load_preset(preset_name)
    if preset:
        return {"success": True, "preset": preset}
    else:
        return {"success": False, "error": f"Preset '{preset_name}' not found"}

@app.post("/api/camera/presets/{preset_name}")
async def save_camera_preset(preset_name: str, description: str = ""):
    """Save current settings as a named preset"""
    from services.camera_settings_manager import get_camera_settings_manager
    manager = get_camera_settings_manager()
    success = manager.save_as_preset(preset_name, description)
    return {
        "success": success,
        "message": f"Preset '{preset_name}' saved" if success else "Failed to save preset"
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

@app.post("/api/session/force-reset")
async def force_reset_camera_and_memory():
    """
    Force camera reset and memory cleanup (called on page refresh).

    This endpoint:
    1. Forces video track recreation (even if < 5 min old)
    2. Clears frame buffers
    3. Cleans up stale monitoring_controls and orchestrators
    4. Preserves VLM model (stays loaded)

    Returns cleanup stats.
    """
    try:
        logger.info("🔄 Force reset requested - clearing camera and memory")

        cleanup_stats = {
            "video_track_reset": False,
            "frame_buffers_cleared": 0,
            "monitoring_controls_cleared": 0,
            "orchestrators_cleared": 0,
            "peer_connections_active": 0
        }

        # Import required modules
        from api import websocket

        # 1. Force reset video track (even if not stale)
        if webrtc_manager:
            # Check if video track exists
            if webrtc_manager._video_track:
                logger.info("Forcing video track reset")
                try:
                    if hasattr(webrtc_manager._video_track, 'stop'):
                        webrtc_manager._video_track.stop()
                    webrtc_manager._video_track = None
                    cleanup_stats["video_track_reset"] = True
                    logger.info("✓ Video track reset complete")
                except Exception as e:
                    logger.error(f"Error resetting video track: {e}")

            # 2. Clear all frame buffers
            buffer_count = len(webrtc_manager.frame_buffers)
            webrtc_manager.frame_buffers.clear()
            cleanup_stats["frame_buffers_cleared"] = buffer_count
            logger.info(f"✓ Cleared {buffer_count} frame buffers")

            # Count active connections
            cleanup_stats["peer_connections_active"] = len(webrtc_manager.peer_connections)

        # 3. Clear stale monitoring_controls (keep only active running sessions)
        stale_monitoring = []
        for session_id, control in list(websocket.monitoring_controls.items()):
            # Only keep if actively running
            if not control.get('running', False):
                stale_monitoring.append(session_id)

        for session_id in stale_monitoring:
            del websocket.monitoring_controls[session_id]

        cleanup_stats["monitoring_controls_cleared"] = len(stale_monitoring)
        logger.info(f"✓ Cleared {len(stale_monitoring)} stale monitoring_controls")

        # 4. Clear stale orchestrators (ones not in active connections)
        stale_orchestrators = []
        for session_id in list(websocket.orchestrators.keys()):
            if session_id not in websocket.active_connections:
                stale_orchestrators.append(session_id)

        for session_id in stale_orchestrators:
            try:
                await websocket.orchestrators[session_id].cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up orchestrator {session_id}: {e}")
            del websocket.orchestrators[session_id]

        cleanup_stats["orchestrators_cleared"] = len(stale_orchestrators)
        logger.info(f"✓ Cleared {len(stale_orchestrators)} stale orchestrators")

        # 5. VLM stays loaded (intentional - no reinit needed)
        logger.info("✓ VLM model preserved (no reinit)")

        logger.info(f"🎉 Force reset complete: {cleanup_stats}")

        return {
            "success": True,
            "message": "Camera and memory reset complete",
            "stats": cleanup_stats,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in force reset: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

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