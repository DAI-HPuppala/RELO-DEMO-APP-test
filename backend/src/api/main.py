"""Main FastAPI application."""
import os
import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables FIRST before any other imports
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from api.routes import session, health
from api.websocket import websocket_endpoint
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

        # Initialize barcode detection service
        logger.info("🔍 Initializing barcode detection service...")
        from services.barcode_detection_service import get_barcode_service
        barcode_service = get_barcode_service()
        barcode_init_success = await barcode_service.initialize()

        if barcode_init_success:
            logger.info("✅ Barcode detection service initialized successfully")
            stats = barcode_service.get_statistics()
            logger.info(f"   - Allowed types: {', '.join(stats['allowed_types'])}")
            logger.info(f"   - 2-stage detection enabled")
        else:
            logger.warning("⚠️ Barcode detection service failed to initialize")
            logger.warning("   - Barcode detection will be unavailable")
            logger.warning("   - Check pyzbar and libzbar0 installation")

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
        logger.info("📏 Starting camera height measurement using stereo depth...")

        # Create camera service and capture depth frame
        camera_service = OakDCameraService()
        depth_frame = camera_service.capture_depth_frame()

        if depth_frame is None:
            logger.error("Failed to capture depth frame")
            return {
                "success": False,
                "error": "Failed to capture depth frame - check camera connection"
            }

        # Sample center region of depth map (40%-60% of frame)
        # This assumes the bench/ground is roughly centered under camera
        h, w = depth_frame.shape
        center_roi = depth_frame[
            int(h*0.4):int(h*0.6),
            int(w*0.4):int(w*0.6)
        ]

        # Filter out invalid depth values (0 or NaN)
        valid_depths = center_roi[center_roi > 0]

        if len(valid_depths) == 0:
            logger.error("No valid depth measurements found in center ROI")
            return {
                "success": False,
                "error": "No valid depth measurements - ensure camera is positioned above surface"
            }

        # Calculate median depth (robust to outliers)
        median_depth_mm = float(np.median(valid_depths))
        height_cm = median_depth_mm / 10.0  # Convert mm to cm

        logger.info(f"✅ Measured camera height: {height_cm:.1f} cm ({median_depth_mm:.0f} mm)")

        # Calculate FOV coverage at this distance
        fov_h_deg = 69  # OAK-D Pro horizontal FOV
        fov_v_deg = 55  # OAK-D Pro vertical FOV

        ground_width_cm = 2 * height_cm * np.tan(np.radians(fov_h_deg/2))
        ground_height_cm = 2 * height_cm * np.tan(np.radians(fov_v_deg/2))

        ground_coverage = {
            "width_cm": round(ground_width_cm, 2),
            "height_cm": round(ground_height_cm, 2)
        }

        logger.info(f"📐 Ground coverage: {ground_width_cm:.1f}cm × {ground_height_cm:.1f}cm")

        # Save measurement to persistent settings
        manager = get_camera_settings_manager()
        manager.save_height_measurement(height_cm, ground_coverage, make_persistent=True)

        logger.info("💾 Height measurement saved to persistent settings")

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

# Barcode Detection endpoints
@app.get("/api/barcode/config")
async def get_barcode_config():
    """Get current barcode detection configuration."""
    from services.barcode_detection_service import get_barcode_service

    try:
        service = get_barcode_service()
        stats = service.get_statistics()

        return {
            "enabled": os.getenv('BARCODE_ENABLED', 'true').lower() == 'true',
            "required": os.getenv('BARCODE_REQUIRED', 'false').lower() == 'true',
            "timeout_seconds": int(os.getenv('BARCODE_DETECTION_TIMEOUT', '0')),
            "auto_delay_ms": int(os.getenv('BARCODE_AUTO_DELAY_MS', '1000')),
            "allowed_types": service.allowed_types,
            "supported_types": service.SUPPORTED_TYPES,
            "initialized": stats['initialized'],
            "save_frames": service.save_frames,
            "statistics": {
                "detection_count": stats['detection_count'],
                "avg_presence_time_ms": stats['avg_presence_time_ms'],
                "avg_decode_time_ms": stats['avg_decode_time_ms']
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting barcode config: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to get barcode config: {str(e)}"}
        )

@app.post("/api/barcode/config")
async def update_barcode_config(config: dict = Body(...)):
    """
    Update barcode detection configuration.

    Args:
        config: Configuration dictionary with optional keys:
            - allowed_types: List of barcode types to detect (e.g., ['CODE128', 'QRCODE'])
    """
    from services.barcode_detection_service import get_barcode_service

    try:
        service = get_barcode_service()

        # Update allowed types if provided
        if 'allowed_types' in config:
            allowed_types = config['allowed_types']
            if not isinstance(allowed_types, list):
                return JSONResponse(
                    status_code=400,
                    content={"detail": "allowed_types must be a list"}
                )

            service.set_allowed_types(allowed_types)
            logger.info(f"Updated barcode allowed types: {allowed_types}")

        return {
            "success": True,
            "message": "Barcode configuration updated",
            "config": {
                "allowed_types": service.allowed_types,
                "supported_types": service.SUPPORTED_TYPES
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error updating barcode config: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to update barcode config: {str(e)}"}
        )

@app.post("/api/barcode/manual-entry")
async def manual_barcode_entry(session_id: str, barcode_data: str):
    """
    Submit manually entered barcode data for current session.

    Args:
        session_id: Current session ID
        barcode_data: Manually entered barcode string
    """
    from api.websocket import orchestrators

    try:
        if not barcode_data or not barcode_data.strip():
            return JSONResponse(
                status_code=400,
                content={"detail": "Barcode data cannot be empty"}
            )

        # Get orchestrator for session
        if session_id not in orchestrators:
            return JSONResponse(
                status_code=404,
                content={"detail": f"Session {session_id} not found or not active"}
            )

        orchestrator = orchestrators[session_id]

        # Find barcode agent in current session state
        barcode_state = orchestrator.session_state.agent_states.get('barcode_detector')
        if not barcode_state:
            return JSONResponse(
                status_code=400,
                content={"detail": "Barcode detector not active in current session"}
            )

        # Set manual entry on barcode state
        barcode_state.barcode_data = barcode_data.strip()
        barcode_state.barcode_type = "MANUAL"
        barcode_state.barcode_manually_entered = True
        barcode_state.barcode_timestamp = time.time()

        logger.info(f"Manual barcode entry for session {session_id}: {barcode_data}")

        return {
            "success": True,
            "message": "Manual barcode entry recorded",
            "barcode_data": barcode_data.strip(),
            "barcode_type": "MANUAL",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error processing manual barcode entry: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to process manual entry: {str(e)}"}
        )

@app.post("/api/barcode/skip")
async def skip_barcode_detection(session_id: str):
    """
    Skip barcode detection for current session/cycle.

    Args:
        session_id: Current session ID
    """
    from api.websocket import orchestrators

    try:
        # Get orchestrator for session
        if session_id not in orchestrators:
            return JSONResponse(
                status_code=404,
                content={"detail": f"Session {session_id} not found or not active"}
            )

        orchestrator = orchestrators[session_id]

        # Find barcode agent in current session state
        barcode_state = orchestrator.session_state.agent_states.get('barcode_detector')
        if not barcode_state:
            return JSONResponse(
                status_code=400,
                content={"detail": "Barcode detector not active in current session"}
            )

        # Mark barcode as skipped
        from orchestration.models.agent_state import AgentStatus
        barcode_state.barcode_data = None
        barcode_state.barcode_type = None
        barcode_state.status = AgentStatus.COMPLETED

        logger.info(f"Barcode detection skipped for session {session_id}")

        return {
            "success": True,
            "message": "Barcode detection skipped",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error skipping barcode detection: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to skip barcode: {str(e)}"}
        )

@app.post("/api/barcode/retry-init")
async def retry_barcode_init():
    """
    Retry barcode detection service initialization.
    Useful if the service failed to initialize at startup.
    """
    from services.barcode_detection_service import get_barcode_service

    try:
        service = get_barcode_service()

        if service.is_initialized:
            return {
                "success": True,
                "message": "Barcode service already initialized",
                "already_initialized": True,
                "timestamp": datetime.now().isoformat()
            }

        # Retry initialization
        logger.info("Retrying barcode service initialization...")
        success = await service.reinitialize()

        if success:
            logger.info("✅ Barcode service initialized successfully")
            return {
                "success": True,
                "message": "Barcode service initialized successfully",
                "config": {
                    "allowed_types": service.allowed_types,
                    "supported_types": service.SUPPORTED_TYPES
                },
                "timestamp": datetime.now().isoformat()
            }
        else:
            logger.error("❌ Barcode service initialization failed")
            return JSONResponse(
                status_code=500,
                content={"detail": "Barcode service initialization failed. Check logs for details."}
            )
    except Exception as e:
        logger.error(f"Error retrying barcode initialization: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to retry initialization: {str(e)}"}
        )

@app.get("/api/barcode/statistics")
async def get_barcode_statistics():
    """Get barcode detection statistics and performance metrics."""
    from services.barcode_detection_service import get_barcode_service

    try:
        service = get_barcode_service()
        stats = service.get_statistics()

        return {
            **stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting barcode statistics: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to get statistics: {str(e)}"}
        )

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