#!/usr/bin/env python3
"""
Lightweight integration of eBUS GigE capture for FastAPI.
Contains `capture_single_image` and `save_image` functions.
"""
# ensure Python knows where the eBUS bindings live
import os
import sys
import site
# prepend container site-packages paths to sys.path
for p in site.getsitepackages():
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# Add additional common paths for Debian packages
for path in [
    "/usr/lib/python3/dist-packages",
    "/usr/local/lib/python3.10/dist-packages",
    "/usr/lib/python3.10/dist-packages",
    "/usr/local/lib/python3.10/site-packages",
    "/opt/jai/ebus_sdk/Ubuntu-22.04-x86_64/lib"
]:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

# Add LD_LIBRARY_PATH for native libraries
os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + ':/opt/jai/ebus_sdk/Ubuntu-22.04-x86_64/lib'

# Debug path info
print("Python paths:", sys.path)
print("LD_LIBRARY_PATH:", os.environ.get('LD_LIBRARY_PATH', ''))

try:
    import eBUS as eb
    print("Successfully imported eBUS module")
except ImportError as e:
    print(f"ERROR importing eBUS: {e}")
    # Try to find any eBUS files in common locations
    for path in sys.path:
        if os.path.isdir(path):
            try:
                ebus_files = [f for f in os.listdir(path) if "ebus" in f.lower()]
                if ebus_files:
                    print(f"Found potential eBUS files in {path}: {ebus_files}")
            except Exception:
                pass

import time
from datetime import datetime

import numpy as np
import cv2
from PIL import Image

# ensure eBUS SDK path if installed in virtual env
venv_sp = os.path.join(sys.prefix, "Lib", "site-packages")
if os.path.isdir(venv_sp) and venv_sp not in sys.path:
    sys.path.insert(0, venv_sp)
import eBUS as eb

BUFFER_COUNT = 16

# persistent camera contexts per device
_contexts = {}

def capture_single_image(connection_id=None, _retry=False):
    """Capture a single image from the specified GigE camera, with per-device persistent streams"""
    global _contexts
    print("[DEBUG] Starting camera capture...")
    # discover camera if not specified
    if connection_id is None:
        print("[DEBUG] Searching for available cameras...")
        system = eb.PvSystem(); system.Find()
        for i in range(system.GetInterfaceCount()):
            iface = system.GetInterface(i)
            for j in range(iface.GetDeviceCount()):
                connection_id = iface.GetDeviceInfo(j).GetConnectionID()
                print(f"[DEBUG] Found device: {connection_id}")
                break
            if connection_id: break
        if connection_id is None:
            print("[DEBUG] No cameras found")
            return None
    # init or reuse context
    ctx = _contexts.get(connection_id)
    if not ctx:
        print(f"[DEBUG] Connecting to device {connection_id}")
        # attempt device connection
        try:
            result, device = eb.PvDevice.CreateAndConnect(connection_id)
        except Exception as e:
            print(f"[DEBUG] Connection error: {e}")
            if not _retry:
                return capture_single_image(connection_id, _retry=True)
            return None
        if device is None:
            print("[DEBUG] Failed to connect device")
            if not _retry:
                return capture_single_image(connection_id, _retry=True)
            return None
        print("[DEBUG] Opening stream...")
        # attempt stream open
        try:
            result, stream = eb.PvStream.CreateAndOpen(connection_id)
        except Exception as e:
            print(f"[DEBUG] Stream open error: {e}")
            try: device.Disconnect()
            except: pass
            if not _retry:
                return capture_single_image(connection_id, _retry=True)
            return None
        if stream is None:
            print("[DEBUG] Failed to open stream")
            device.Disconnect()
            if not _retry:
                return capture_single_image(connection_id, _retry=True)
            return None
        # configure packet size for GigE
        if isinstance(device, eb.PvDeviceGEV):
            system = eb.PvSystem(); system.Find()
            for k in range(system.GetInterfaceCount()):
                iface = system.GetInterface(k)
                try:
                    ip = str(iface.GetIPAddress(0))
                except:
                    continue
                if ip.startswith(('169.254','192.168')):
                    device.NegotiatePacketSize()
                    device.SetStreamDestination(ip, stream.GetLocalPort())
                    break
        # allocate buffers
        size = device.GetPayloadSize()
        buf_count = min(stream.GetQueuedBufferMaximum(), BUFFER_COUNT)
        buffers = []
        for _ in range(buf_count):
            buf = eb.PvBuffer(); buf.Alloc(size)
            buffers.append(buf); stream.QueueBuffer(buf)
        device.StreamEnable()
        device.GetParameters().Get("AcquisitionStart").Execute()
        ctx = {"device":device, "stream":stream, "buffers":buffers, "last_capture_time":None, "last_error":None}
        _contexts[connection_id] = ctx
    device = ctx["device"]; stream = ctx["stream"]
    # capture single frame
    device.GetParameters().Get("AcquisitionStart").Execute()
    try:
        result, buf, op = stream.RetrieveBuffer(500)
    except Exception as e:
        # retrieval error: reset context and retry once
        ctx['last_error'] = str(e)
        reset_camera_stream(connection_id)
        if not _retry:
            return capture_single_image(connection_id, _retry=True)
        return None
    if result.IsOK() and op.IsOK() and buf.GetPayloadType() == eb.PvPayloadTypeImage:
        img = buf.GetImage(); w,h = img.GetWidth(), img.GetHeight()
        raw = np.frombuffer(img.GetDataPointer(), dtype=np.uint8).copy()
        if img.GetPixelType() == eb.PvPixelMono8:
            frame = raw.reshape((h,w))
        else:
            temp = raw.reshape((h,w)); frame = cv2.cvtColor(temp, cv2.COLOR_BayerRG2BGR)
        stream.QueueBuffer(buf)
        device.GetParameters().Get("AcquisitionStop").Execute()
        # record successful capture
        ctx['last_capture_time'] = datetime.now().isoformat()
        ctx['last_error'] = None
        return frame
    # failed result or wrong payload: reset and retry once
    err = f"RetrieveBuffer failed: result={result}, op={op}"
    ctx['last_error'] = err
    reset_camera_stream(connection_id)
    if not _retry:
        return capture_single_image(connection_id, _retry=True)
    return None


def save_image(img_data, directory=None, filename=None):
    if img_data is None or not isinstance(img_data, np.ndarray):
        return None
    directory = directory or os.path.join(os.getcwd(), "images")
    os.makedirs(directory, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if filename:
        base, ext = os.path.splitext(filename)
        ext = ext or '.jpg'
        filename = f"{base}_{ts}{ext}"
    else:
        filename = f"image_{ts}.jpg"
    path = os.path.join(directory, filename)
    if img_data.dtype != np.uint8:
        img_data = (img_data / 256).astype(np.uint8)
    mode = 'L' if img_data.ndim == 2 else 'RGB'
    pil = Image.fromarray(img_data, mode)
    pil.save(path, 'JPEG', quality=95)
    return path

def get_camera_statuses():
    """Return health status for all camera contexts"""
    statuses = []
    for cam_id, ctx in _contexts.items():
        statuses.append({
            "device": cam_id,
            "stream_open": ctx["stream"] is not None,
            "buffers_queued": len(ctx["buffers"]),
            "last_capture_time": ctx.get("last_capture_time"),
            "last_error": ctx.get("last_error")
        })
    return statuses

def reset_camera_stream(connection_id: str) -> bool:
    """Reset and remove the camera context for a given camera."""
    ctx = _contexts.pop(connection_id, None)
    if not ctx:
        return False
    device = ctx["device"]
    stream = ctx["stream"]
    # Stop acquisition
    try:
        device.GetParameters().Get("AcquisitionStop").Execute()
    except Exception:
        pass
    # Close stream and disconnect device
    try:
        stream.Close()
    except Exception:
        pass
    try:
        device.Disconnect()
    except Exception:
        pass
    return True
