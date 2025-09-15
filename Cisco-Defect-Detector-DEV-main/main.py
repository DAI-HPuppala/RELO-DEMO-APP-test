from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body, BackgroundTasks, Request
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoModelForCausalLM
from PIL import Image, ImageDraw
import io
from draw_utils import draw_box_with_label
import logging
import os
import sys
import time
from splunk_hec_sender import SplunkHECClient
import base64

# Configure detailed logging and override uvicorn default handlers
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

# Log Python paths and environment variables
logger.debug(f"Python paths: {sys.path}")
logger.debug(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
logger.debug(f"GENICAM_ROOT_V3_4: {os.environ.get('GENICAM_ROOT_V3_4', 'Not set')}")
logger.debug(f"GENICAM_GENTL64_PATH: {os.environ.get('GENICAM_GENTL64_PATH', 'Not set')}")

import torch
from typing import List

from simple_capture import capture_single_image, save_image, get_camera_statuses, reset_camera_stream
splunk_client = None
ebus_system = None

app = FastAPI()

@app.on_event("startup")
async def init_resources():
    global splunk_client, ebus_system
    # Splunk HEC setup
    url = os.getenv("SPLUNK_HEC_URL")
    token = os.getenv("SPLUNK_HEC_TOKEN")
    index = os.getenv("SPLUNK_HEC_INDEX", "main")
    logger.info(f"Splunk HEC URL={url!r}, token set={bool(token)}, index={index!r}")
    if url and token:
        splunk_client = SplunkHECClient(url, token, index=index)
    else:
        logger.warning("Splunk HEC not configured; skipping send initialization.")
    # eBUS system initialization
    try:
        import eBUS as ebus
        logger.info("Initializing eBUS system...")
        ebus_system = ebus.PvSystem()
        ebus_system.Find()
        logger.info(f"eBUS interfaces found: {ebus_system.GetInterfaceCount()}")
    except Exception as e:
        logger.error(f"eBUS init error: {e}", exc_info=True)
        ebus_system = None

@app.on_event("startup")
async def load_model():
    global model
    logging.info("Loading Moondream2 model with GPU acceleration...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-04-14",
        trust_remote_code=True,
        device_map="auto",
    )
    model.to(device)
    logging.info(f"Model loaded on {device}")

@app.get("/")
async def root():
    return {"status": "running"}

@app.get("/camera_health")
async def camera_health():
    """Return health status for all camera streams"""
    logger.info("camera_health endpoint called")
    # list all eBUS devices and merge with stream contexts
    all_devs = get_ebus_devices()
    ctx_stats = get_camera_statuses()
    ctx_map = {s['device']: s for s in ctx_stats}
    statuses = []
    for dev in all_devs:
        # normalize connection_id to Python string
        cid = get_ebus_value(dev.get('connection_id'))
        status = ctx_map.get(cid, {
            'device': cid,
            'stream_open': False,
            'buffers_queued': 0,
            'last_capture_time': None,
            'last_error': None
        })
        statuses.append(status)
    logger.info(f"camera_health statuses: {statuses}")
    return {"cameras": statuses}

@app.post("/camera_reset")
async def camera_reset(device: str = Query(..., description="Camera connection ID to reset")):
    """Reset a camera's stream context to recover from errors"""
    logger.info(f"Resetting camera stream for {device}")
    success = reset_camera_stream(device)
    if not success:
        raise HTTPException(status_code=404, detail=f"No active stream for device {device}")
    return {"device": device, "reset": True}

def run_detection(image: Image.Image, targets: List[str], annotated: bool):
    boxes = []
    for t in targets:
        res = model.detect(image, t)
        logging.info(f"Raw detection result for {t}: {res}")
        for obj in res.get("objects", []):
            if all(k in obj for k in ("x_min","y_min","x_max","y_max")):
                x0 = obj["x_min"] * image.width
                y0 = obj["y_min"] * image.height
                x1 = obj["x_max"] * image.width
                y1 = obj["y_max"] * image.height
            else:
                vals = obj.get("box") or obj.get("bbox")
                if not vals: continue
                x0,y0,x1,y1 = vals
            # clamp coordinates to image bounds
            x0c, x1c = max(0, x0), min(image.width, x1)
            y0c, y1c = max(0, y0), min(image.height, y1)
            w_px, h_px = max(0, x1c-x0c), max(0, y1c-y0c)
            area = w_px * h_px
            total = image.width * image.height
            rel = area / total
            # skip boxes spanning full width or height
            if w_px >= image.width or h_px >= image.height:
                logger.info(f"Filtering box spanning full dimension: w={w_px}px,h={h_px}px")
                continue
            # filter by relative area: skip <0.3% or >50%
            if rel < 0.003 or rel > 0.5:
                logger.info(f"Filtering box: {area} px ({rel:.2%}) outside [0.3%,50%]")
                continue
            boxes.append({"box":[x0,y0,x1,y1],"score":obj.get("score"),"target":t})
    if annotated:
        draw = ImageDraw.Draw(image)
        for b in boxes:
            draw_box_with_label(
                draw,
                b["box"],
                b["target"],
                b["score"],
                img_size=(image.width, image.height)
            )
        buf = io.BytesIO(); image.save(buf,format="JPEG"); buf.seek(0)
        return StreamingResponse(buf,media_type="image/jpeg")
    return JSONResponse({"boxes":boxes})

@app.post("/detect")
async def detect(file: UploadFile = File(...), target: List[str] = Query(["paint scratches"], description="List of detection targets"), annotated: bool = False):
    logging.info(f"Received /detect request: target={target}, annotated={annotated}, filename={file.filename}")
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")
    return run_detection(image, target, annotated)

@app.post("/capture")
async def capture_endpoint(background_tasks: BackgroundTasks, device: str = Query(None), filename: str = Query(None), directory: str = Query("images", description="Directory to save captures")):
    logger.info(f"Received /capture: device={device}, filename={filename}, directory={directory}")
    logger.debug("Calling capture_single_image() now")
    start = time.time()
    img_data = capture_single_image(device)
    if img_data is None:
        logger.warning(f"Initial capture failed for {device}, resetting stream and retry")
        reset_camera_stream(device)
        img_data = capture_single_image(device)
        if img_data is None:
            logger.error(f"Retry capture failed for {device}")
            raise HTTPException(500, "Failed to capture image after retry")
    duration = time.time() - start
    logger.info(f"capture_single_image took {duration:.3f}s")
    path = save_image(img_data,directory,filename)
    if not path:
        raise HTTPException(500,"Failed to save image")
    # schedule Splunk send in background
    splunk_queued = False
    if splunk_client:
        with open(path, "rb") as f:
            img_bytes = f.read()
        background_tasks.add_task(splunk_client.send_image, img_bytes, os.path.basename(path))
        splunk_queued = True
    return JSONResponse({
        "saved": path,
        "device": device,
        "filename_used": filename,
        "directory_used": directory,
        "splunk_queued": splunk_queued
    })

@app.post("/detect_capture")
async def detect_capture(
    background_tasks: BackgroundTasks,
    device: str = Query(None),
    target: List[str] = Query(["paint scratches"]),
    annotated: bool = False,
    filename: str = Query(None),
    directory: str = Query("images", description="Directory to save captures")
):
    logging.info(f"Received /detect_capture: device={device}, target={target}, annotated={annotated}")
    img_data = capture_single_image(device)
    if img_data is None:
        logger.warning(f"Initial detect_capture failed for {device}, resetting stream and retry")
        reset_camera_stream(device)
        img_data = capture_single_image(device)
        if img_data is None:
            logger.error(f"Retry detect_capture failed for {device}")
            raise HTTPException(500, "Failed to capture image after retry")
    # ensure output directory exists
    os.makedirs(directory, exist_ok=True)
    if not annotated:
        # save raw capture to static filename
        path = save_image(img_data, directory, filename)
        if not path:
            raise HTTPException(500, "Failed to save image")
    # convert to PIL
    if img_data.ndim==2:
        image = Image.fromarray(img_data,'L')
    else:
        image = Image.fromarray(img_data,'RGB')
    # Run detection first
    boxes = []
    for t in target:
        res = model.detect(image, t)
        logging.info(f"Detection result for {t}: {res}")
        for obj in res.get("objects", []):
            if all(k in obj for k in ("x_min", "y_min", "x_max", "y_max")):
                x0 = obj["x_min"] * image.width
                y0 = obj["y_min"] * image.height
                x1 = obj["x_max"] * image.width
                y1 = obj["y_max"] * image.height
            else:
                vals = obj.get("box") or obj.get("bbox")
                if not vals:
                    continue
                x0, y0, x1, y1 = vals
            # clamp coords
            x0c, x1c = max(0, x0), min(image.width, x1)
            y0c, y1c = max(0, y0), min(image.height, y1)
            w_px, h_px = max(0, x1c-x0c), max(0, y1c-y0c)
            area = w_px * h_px
            total = image.width * image.height
            rel = area / total
            # skip boxes spanning full width or height
            if w_px >= image.width or h_px >= image.height:
                logger.info(f"Filtering box spanning full dimension: w={w_px}px,h={h_px}px")
                continue
            # filter by relative area: skip <0.3% or >50%
            if rel < 0.003 or rel > 0.5:
                logger.info(f"Filtering box: {area} px ({rel:.2%}) outside [0.3%,50%]")
                continue
            boxes.append({"box":[x0, y0, x1, y1], "score": obj.get("score"), "target": t})
    if annotated:
        # Build detection boxes and annotate image
        draw = ImageDraw.Draw(image)
        for b in boxes:
            draw_box_with_label(
                draw,
                b["box"],
                b["target"],
                b["score"],
                img_size=(image.width, image.height)
            )
        
        # Annotate in memory and write only final JPEG to static filename
        buf = io.BytesIO(); image.save(buf, format="JPEG"); buf.seek(0)
        img_bytes = buf.getvalue()
        # determine display filename
        if filename:
            base, ext = os.path.splitext(filename)
            ext = ext or '.jpg'
            final_filename = f"{base}{ext}"
        else:
            final_filename = "image.jpg"
        final_path = os.path.join(directory, final_filename)
        image.save(final_path, format="JPEG")
        path = final_path
        
        # schedule annotated result send to Splunk after successful detection
        splunk_queued = False
        if splunk_client:
            b64 = base64.b64encode(img_bytes).decode('ascii')
            payload = {
                "device": device,
                "targets": target,
                "boxes": boxes,
                "image_b64": b64
            }
            logger.info(f"Splunk payload summary: {{'device': device, 'targets': target, 'boxes': boxes}}")
            background_tasks.add_task(splunk_client.send_event, payload)
            splunk_queued = True
        
        # Return detection results without image, include saved info and boxes
        response_payload = {
            "saved": path,
            "device": device,
            "filename_used": filename,
            "directory_used": directory,
            "splunk_queued": splunk_queued,
            "targets": target,
            "boxes": boxes
        }
        logger.info(f"Response payload: {response_payload}")
        return JSONResponse(response_payload)
    
    # For non-annotated, build and return JSON with saved info and detection boxes
    response_payload = {
        "saved": path,
        "device": device,
        "filename_used": filename,
        "directory_used": directory,
        "splunk_queued": False,
        "targets": target,
        "boxes": boxes
    }
    logger.info(f"Response payload: {response_payload}")
    return JSONResponse(response_payload)

def get_ebus_devices():
    """Get information about all available eBUS devices"""
    try:
        system = ebus_system
        system.Find()
        devices = []
        
        interface_count = system.GetInterfaceCount()
        logger.info(f"Found {interface_count} network interfaces")
        
        for i in range(interface_count):
            interface = system.GetInterface(i)
            device_count = interface.GetDeviceCount()
            logger.info(f"Interface {i} has {device_count} devices")
            
            for j in range(device_count):
                device_info = interface.GetDeviceInfo(j)
                try:
                    device = {
                        "interface_index": i,
                        "connection_id": device_info.GetConnectionID(),
                        "ip_address": device_info.GetIPAddress().GetAscii(),
                        "mac_address": device_info.GetMACAddress().GetAscii(),
                        "vendor": device_info.GetVendorName().GetAscii(),
                        "model": device_info.GetModelName().GetAscii(),
                        "version": device_info.GetVersion().GetAscii(),
                        "user_id": device_info.GetUserDefinedName().GetAscii()
                    }
                    devices.append(device)
                    logger.info(f"Found device: {device}")
                except Exception as e:
                    logger.error(f"Error getting device info: {e}")
                    continue
        
        return devices
    except Exception as e:
        logger.error(f"Error getting eBUS devices: {e}", exc_info=True)
        return []

def get_ebus_value(value):
    """Convert eBUS value to string safely"""
    try:
        if hasattr(value, 'GetAscii'):
            return value.GetAscii()
        return str(value)
    except Exception as e:
        logger.error(f"Error converting eBUS value: {e}")
        return str(value)

@app.get("/ebus_devices")
async def ebus_devices():
    """Get all available eBUS devices and their information"""
    try:
        system = ebus_system
        system.Find()
        devices = []
        
        interface_count = system.GetInterfaceCount()
        logger.info(f"Found {interface_count} network interfaces")
        
        for i in range(interface_count):
            interface = system.GetInterface(i)
            device_count = interface.GetDeviceCount()
            logger.info(f"Interface {i} has {device_count} devices")
            
            for j in range(device_count):
                device_info = interface.GetDeviceInfo(j)
                try:
                    device = {
                        "interface_index": i,
                        "connection_id": get_ebus_value(device_info.GetConnectionID()),
                        "ip_address": get_ebus_value(device_info.GetIPAddress()),
                        "mac_address": get_ebus_value(device_info.GetMACAddress()),
                        "vendor": get_ebus_value(device_info.GetVendorName()),
                        "model": get_ebus_value(device_info.GetModelName()),
                        "version": get_ebus_value(device_info.GetVersion()),
                        "user_id": get_ebus_value(device_info.GetUserDefinedName())
                    }
                    devices.append(device)
                    logger.info(f"Found device: {device}")
                except Exception as e:
                    logger.error(f"Error getting device info: {e}")
                    continue
        
        return JSONResponse({"devices": devices})
    except Exception as e:
        logger.error(f"Error getting eBUS devices: {e}", exc_info=True)
        return JSONResponse({"devices": [], "error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
