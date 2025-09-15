# Cisco Defect Detector API

A FastAPI-based service running as a systemd-managed Uvicorn server that captures images from GigE Vision cameras via eBUS, performs object detection using the Moondream2 model on GPU, annotates images with bounding boxes, and integrates with Splunk via HEC.

## Architecture

- **eBUS Capture**: Acquire frames from GigE Vision cameras using PvSystem
- **Detection**: Moondream2 model via Hugging Face Transformers on GPU
- **Annotation**: Draw bounding boxes and labels using Pillow
- **Splunk Integration**: Send images to HTTP Event Collector
- **Service**: FastAPI served by Uvicorn, managed by systemd

## Service Setup & Management

1. **Configure Environment**  
   Configure Splunk HEC (replace with your values):
   ```bash
   export SPLUNK_HEC_URL="http://192.168.20.7:8088"
   export SPLUNK_HEC_TOKEN="f440a5b9-d82b-4110-b60c-9e7fb7270b0b"
   export SPLUNK_HEC_INDEX="mfg_qa_vision"
   ```
   Your application will send events in JSON format. Each event payload is a JSON object with fields:
   ```json
   {
     "time": 1620000000.0,
     "event": { <detection or image_b64 data> },
     "index": "mfg_qa_vision",
     "source": "python_script",
     "sourcetype": "json"
   }
   ```
2. **Install systemd Unit**  
   ```bash
   sudo cp cisco-defect-detector.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable cisco-defect-detector.service
   ```
   **Example systemd unit [Service] section**  
   ```ini
   [Service]
   # eBUS SDK environment
   Environment=LD_LIBRARY_PATH=/opt/jai/ebus_sdk/Ubuntu-22.04-x86_64/lib/genicam/bin/Linux64_x64:/opt/jai/ebus_sdk/Ubuntu-22.04-x86_64/lib
   Environment=GENICAM_ROOT_V3_4=/opt/jai/ebus_sdk/Ubuntu-22.04-x86_64/lib/genicam
   Environment=GENICAM_GENTL64_PATH=/opt/jai/ebus_sdk/Ubuntu-22.04-x86_64/lib/genicam/bin/Linux64_x64
   Environment=PYTHONPATH=/opt/jai/ebus_sdk/Ubuntu-22.04-x86_64/lib
   # Splunk HEC configuration
   Environment=SPLUNK_HEC_URL=http://192.168.20.7:8088
   Environment=SPLUNK_HEC_TOKEN=f440a5b9-d82b-4110-b60c-9e7fb7270b0b
   Environment=SPLUNK_HEC_INDEX=mfg_qa_vision
   ```
3. **Start/Restart Service**  
   ```bash
   sudo systemctl start cisco-defect-detector.service
   sudo systemctl restart cisco-defect-detector.service
   ```
4. **Logs & Status**  
   ```bash
   sudo systemctl status cisco-defect-detector.service
   sudo journalctl -u cisco-defect-detector.service -f
   ```

## API Endpoints

- **GET /**  
  Health check → `{ "status": "running" }`

- **GET /camera_health**  
  Returns health status for all discovered cameras and streams.  
  Response JSON:
  ```json
  {
    "cameras": [
      {
        "device": "169.254.1.59",
        "stream_open": true,
        "buffers_queued": 16,
        "last_capture_time": "2025-05-30T15:12:34.123456",
        "last_error": null
      },
      {
        "device": "169.254.1.60",
        "stream_open": false,
        "buffers_queued": 0,
        "last_capture_time": null,
        "last_error": null
      }
    ]
  }
  ```

- **POST /capture**  
  Capture a raw image from a camera.  
  Query params:
  - `device` (string, required): camera IP  
  - `filename` (string, optional): filename to save (default: `{device}_{timestamp}.jpg`)  
  - `directory` (string, optional): directory to save captures (default: `images`)  
  Returns JSON:
  ```json
  {
    "saved": "/path/to/file.jpg",
    "device": "169.254.1.59",
    "filename_used": "169.254.1.59_1620000000.jpg",
    "directory_used": "images",
    "splunk_queued": false
  }
  ```

- **POST /detect_capture**  
  Capture a raw image, run detection, and optionally annotate.  
  Query params:
  - `device` (string, required): camera IP  
  - `target` (string, repeatable): label(s) to detect (default: `paint scratches`)  
  - `annotated` (boolean, optional, default false): return annotated JPEG if true  
  - `filename` (string, optional): filename to save raw capture (default: `{device}_{timestamp}.jpg`)  
  - `directory` (string, optional): directory to save captures (default: `images`)  
  Responses:
  - **annotated=false** → JSON:
    ```json
    {
      "saved": "/path/to/file.jpg",
      "device": "169.254.1.59",
      "filename_used": "169.254.1.59_1620000000.jpg",
      "directory_used": "images",
      "splunk_queued": true,
      "targets": ["cup"],
      "boxes": [
        { "box": [x0, y0, x1, y1], "score": 0.95, "target": "cup" }
      ]
    }
    ```
  - **annotated=true** → JPEG image with bounding boxes and header `X-Splunk-Sent: true|false`  
    The saved file on disk is also overwritten with the annotated image.  
  Logging: Splunk payload summary (device, targets, boxes) is logged; base64 image data is omitted to reduce verbosity.

- **POST /camera_reset**  
  Reset an active camera stream to recover from errors.  
  Query params:
  - `device` (string, required): camera connection ID to reset  
  Response JSON:
  ```json
  { "device": "169.254.1.59", "reset": true }
  ```

- **GET /ebus_devices**  
  Retrieve all connected eBUS (GigE Vision) devices.  
  Returns JSON with a `devices` array of objects:
  ```json
  {
    "devices": [
      {
        "interface_index": 1,
        "connection_id": "169.254.1.59",
        "ip_address": "169.254.1.59",
        "mac_address": "00:11:22:33:44:55",
        "vendor": "VendorName",
        "model": "ModelName",
        "version": "VersionString",
        "user_id": "UserDefinedName"
      }
    ]
  }
  ```

## Examples

JSON only:
```bash
curl -X POST "http://localhost:8080/detect_capture?device=169.254.1.59&target=cup"
```

Annotated image:
```bash
curl -X POST "http://localhost:8080/detect_capture?device=169.254.1.59&target=cup&annotated=true" --output annotated.jpg
```

Multi-target:
```bash
curl -X POST "http://localhost:8080/detect_capture?device=169.254.1.59&target=cup&target=measuring_tape&annotated=true" --output annotated.jpg
```

## Development Strategy
1. **Multi-camera ingest**: build a worker to capture images from 5 cameras→queue
2. **Batch requests**: batch images per camera and poll `/detect`
3. **Display dashboard**: frontend to pull annotated images for each camera
4. **Version tracking**: use Git for source and Docker tags for releases

## Version Control
```bash
git init
git add .
git commit -m "Initial commit: Cisco FastAPI defect detector"
