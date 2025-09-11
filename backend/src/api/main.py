"""Main FastAPI application."""
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv

from api.routes import session, health
from api.websocket import websocket_endpoint
from api.websocket_v2 import websocket_v2_endpoint
from services import SessionManager, WebRTCManager

# Load environment variables
load_dotenv()

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
    logger.info("Starting Returns Classifier API")
    session_manager = SessionManager()
    webrtc_manager = WebRTCManager()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Returns Classifier API")
    # Clean up connections
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