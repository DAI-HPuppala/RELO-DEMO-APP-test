"""Health check API routes."""
from fastapi import APIRouter
from fastapi.responses import JSONResponse
import subprocess
import logging
import os
from services.oakd_camera import OakDCameraService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
async def health_check():
    """System health check."""
    health_status = {
        "status": "healthy",
        "gpu_available": check_gpu_available(),
        "model_loaded": check_model_loaded(),
        "camera_connected": check_camera_connected(),
        "version": "1.0.0"
    }
    
    return JSONResponse(
        content=health_status,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


@router.get("/camera/list")
async def list_cameras():
    """List available cameras."""
    cameras = []
    
    # Check for webcam
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cameras.append({
                "id": "0",
                "type": "webcam",
                "name": "Default Webcam",
                "available": True
            })
            cap.release()
    except:
        pass
    
    # Check for RealSense
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        if len(ctx.devices) > 0:
            for i, device in enumerate(ctx.devices):
                cameras.append({
                    "id": f"realsense_{i}",
                    "type": "realsense",
                    "name": f"Intel RealSense {device.get_info(rs.camera_info.name)}",
                    "available": True
                })
    except:
        pass
    
    # If no cameras found, add a placeholder
    if not cameras:
        cameras.append({
            "id": "none",
            "type": "webcam",
            "name": "No camera detected",
            "available": False
        })
    
    return {"cameras": cameras}


@router.get("/camera/resolution")
async def get_camera_resolution():
    """Get the configured camera resolution."""
    resolution = OakDCameraService.get_camera_resolution()
    return JSONResponse(
        content=resolution,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


@router.get("/camera/features")
async def get_camera_features():
    """Get enabled camera features."""
    tap_to_focus_enabled = os.getenv('TAP_TO_FOCUS_ENABLED', 'true').lower() == 'true'

    return JSONResponse(
        content={
            "tap_to_focus_enabled": tap_to_focus_enabled
        },
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


def check_gpu_available() -> bool:
    """Check if GPU is available."""
    try:
        # Check for NVIDIA GPU
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=2
        )
        return result.returncode == 0
    except:
        # Check for other GPU types or CPU-only mode
        return False


def check_model_loaded() -> bool:
    """Check if Ollama model is loaded."""
    try:
        # Check if Ollama is running and model is available
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=2
        )
        
        if result.returncode == 0:
            # Check if our specific model is in the list
            return "qwen2.5-vl" in result.stdout.lower()
        return False
    except:
        return False


def check_camera_connected() -> bool:
    """Check if a camera is connected."""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        is_connected = cap.isOpened()
        cap.release()
        return is_connected
    except:
        return False