# config.py
from pathlib import Path
import os

# Base directory (where both config.py and main.py live)
BASE_DIR = Path(__file__).parent.parent.resolve()

# NTP server configuration
listenIp = "192.168.0.169"
listenPort = 123

# FTP server configuration
ADDRESS = "0.0.0.0"
FTP_PORT = int(os.getenv("FTP_PORT", "2121"))  # Default FTP port is 2121

# FTP root and archive
FTP_ROOT    = BASE_DIR / "tmp/ftp_root"
ARCHIVE_DIR = BASE_DIR / "tmp/archive"

# Certificates directory
DEVICE_CERT_PATH  = BASE_DIR / 'S3-bucket/creds/725f1775200120a81e2389e8aeb4a6a9a19f2ad987b498316926521104cbb423-certificate.pem.crt'
DEVICE_KEY_PATH   = BASE_DIR / 'S3-bucket/creds/725f1775200120a81e2389e8aeb4a6a9a19f2ad987b498316926521104cbb423-private.pem.key'

IOT_THING_NAME = 'TestThing'

# Output directory for parsed JSON files
OUTPUT_JSON_DIR = BASE_DIR / "parsed_json_files"

# S3 / credentials endpoint
CREDENTIALS_ENDPOINT = "https://chvtk0dqyd91a.credentials.iot.us-east-1.amazonaws.com/role-aliases/s3_thing_role_alias/credentials"
S3_BUCKET = "slap-test-bucket"
LOCATION = "PHX7"

# Directories to upload
BACKEND_DIR = BASE_DIR / "backend"
CAPTURED_FRAMES_DIR = BACKEND_DIR / "captured_frames"
EXPORTS_DIR = BACKEND_DIR / "exports"
LOGS_DIR = BACKEND_DIR / "logs"

# Archive configuration
ARCHIVE_ROOT = BASE_DIR / "archive"
ARCHIVE_RETENTION_DAYS = 2  # Keep archives for N days (1=one day, 2=two days, 7=one week, etc.)

# S3 upload configuration
# S3 structure: s3://bucket/YYYY-MM-DD/captured_frames/...
S3_USE_DATE_PREFIX = True  # Prefix uploads with today's date

# Upload settings
BATCH_SIZE = 100  # Number of files to upload per batch
MAX_WORKERS = 5  # Number of concurrent upload threads
MULTIPART_THRESHOLD = 100 * 1024 * 1024  # 100MB - use multipart for files larger than this
MULTIPART_CHUNKSIZE = 50 * 1024 * 1024  # 50MB chunk size for multipart uploads
MAX_SINGLE_FILE_SIZE = 5 * 1024 * 1024 * 1024  # 5GB - S3 limit for single file
MAX_RETRIES = 3  # Number of retry attempts per file
RETRY_DELAY = 2  # Seconds to wait between retries

# ==========================================
# SCHEDULE CONFIGURATION - CHANGE THESE!
# ==========================================
# After changing times, run: sudo ./update_schedule.sh

JOB_START_TIME = "17:43"      # When to run daily (24-hour format: "HH:MM")
JOB_RETRY_INTERVAL = 10        # Retry failed uploads every N minutes
JOB_DEADLINE_TIME = "17:46"    # Stop retrying after this time
JOB_MAX_RETRIES = 18           # Max retry attempts (auto-calculated: 180min / 10min = 18)

# File type filters
CAPTURED_FRAMES_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
EXPORTS_EXTENSIONS = {'.csv', '.json', '.xlsx', '.xls'}
LOGS_EXTENSIONS = {'.log'}

# Logging configuration
LOG_DIR = BASE_DIR / "S3-bucket" / "logs"
LOG_MAX_SIZE = 50 * 1024 * 1024  # 50MB - create new log file when exceeded
LOG_FILE_PREFIX = "s3_upload"

# Progress tracking
UPLOAD_STATE_FILE = BASE_DIR / "S3-bucket" / "upload_progress.json"

if __name__ == "__main__":
    # Ensure directories exist
    # FTP_ROOT.mkdir(parents=True, exist_ok=True)
    # ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    # CERT_DIR.mkdir(parents=True, exist_ok=True)
    # OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Configuration directories set up at:\n"
          f"FTP Root: {FTP_ROOT}\n"
          f"Archive: {ARCHIVE_DIR}\n"
          f"Certificates: {DEVICE_CERT_PATH}, {DEVICE_KEY_PATH}\n"
          f"Output JSON: {OUTPUT_JSON_DIR}")