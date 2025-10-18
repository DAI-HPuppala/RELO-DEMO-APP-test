"""
S3 Scheduled Uploader - Daily automated upload to S3 with archiving

Features:
- Scheduled execution (daily at 1am)
- Uploads captured_frames, exports, and logs
- Retry logic (1am-4am, every 10 minutes)
- Move uploaded files to archive
- N-day archive retention
- Comprehensive logging with auto-rotation
- Only moves successfully uploaded files
"""

import os
import sys
import time
import json
import threading
import requests
import boto3
import shutil
from boto3.s3.transfer import TransferConfig
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dateutil import parser as dateparser
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime, timedelta
import logging
from logging.handlers import RotatingFileHandler

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent))
import config

# ————— Credential caching —————
_cred_lock = threading.Lock()
_cached_creds = None
_expiry_time = 0

# ————— Progress tracking —————
_progress_lock = threading.Lock()

# ————— Logger setup —————
logger = None


def setup_logger():
    """Setup rotating file logger with auto-rotation"""
    global logger

    # Create logs directory
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger('S3Uploader')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Get current log file
    log_file = get_current_log_file()

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=config.LOG_MAX_SIZE,
        backupCount=10
    )
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_current_log_file() -> Path:
    """Get current log file path, create new one if current exceeds max size"""
    base_name = f"{config.LOG_FILE_PREFIX}_{datetime.now().strftime('%Y%m%d')}.log"
    log_file = config.LOG_DIR / base_name

    # Check if we need a new file due to size
    if log_file.exists() and log_file.stat().st_size > config.LOG_MAX_SIZE:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"{config.LOG_FILE_PREFIX}_{timestamp}.log"
        log_file = config.LOG_DIR / base_name

    return log_file


class UploadStats:
    """Thread-safe upload statistics tracker"""
    def __init__(self):
        self.lock = threading.Lock()
        self.total_files = 0
        self.uploaded = 0
        self.failed = 0
        self.skipped = 0
        self.total_bytes = 0
        self.uploaded_bytes = 0
        self.start_time = time.time()
        self.uploaded_files = set()  # Track successfully uploaded files
        self.failed_files = set()    # Track failed files

    def increment_uploaded(self, file_path: str, size: int):
        with self.lock:
            self.uploaded += 1
            self.uploaded_bytes += size
            self.uploaded_files.add(file_path)

    def increment_failed(self, file_path: str):
        with self.lock:
            self.failed += 1
            self.failed_files.add(file_path)

    def increment_skipped(self):
        with self.lock:
            self.skipped += 1

    def get_stats(self) -> Dict:
        with self.lock:
            elapsed = time.time() - self.start_time
            return {
                'total_files': self.total_files,
                'uploaded': self.uploaded,
                'failed': self.failed,
                'skipped': self.skipped,
                'total_bytes': self.total_bytes,
                'uploaded_bytes': self.uploaded_bytes,
                'elapsed_seconds': elapsed,
                'upload_speed_mbps': (self.uploaded_bytes / 1024 / 1024 / elapsed) if elapsed > 0 else 0,
                'progress_percent': (self.uploaded / self.total_files * 100) if self.total_files > 0 else 0,
                'uploaded_files': list(self.uploaded_files),
                'failed_files': list(self.failed_files)
            }


def get_temp_creds() -> Optional[dict]:
    """Fetch temporary AWS credentials from IoT credentials endpoint"""
    try:
        resp = requests.get(
            config.CREDENTIALS_ENDPOINT,
            headers={'x-amzn-iot-thingname': config.IOT_THING_NAME},
            cert=(str(config.DEVICE_CERT_PATH), str(config.DEVICE_KEY_PATH)),
            timeout=30
        )
        resp.raise_for_status()
        credentials = resp.json()
        return credentials
    except Exception as e:
        logger.error(f"Failed to fetch credentials: {e}")
        return None


def fetch_s3_client():
    """Fetch or refresh temporary credentials and return boto3 S3 client"""
    global _cached_creds, _expiry_time

    with _cred_lock:
        now = time.time()
        if _cached_creds is None or now + 120 > _expiry_time:
            creds_response = get_temp_creds()
            if not creds_response:
                logger.error("Failed to get credentials")
                return None

            creds = creds_response.get("credentials")
            if not creds:
                logger.error("Invalid credential response format")
                return None

            try:
                exp = dateparser.parse(creds["expiration"]).timestamp()
            except Exception as e:
                logger.error(f"Failed to parse credential expiration: {e}")
                return None

            _cached_creds = {
                "aws_access_key_id": creds["accessKeyId"],
                "aws_secret_access_key": creds["secretAccessKey"],
                "aws_session_token": creds["sessionToken"],
            }
            _expiry_time = exp
            logger.info(f"Credentials refreshed, valid until {datetime.fromtimestamp(exp)}")

        session = boto3.Session(**_cached_creds)
        transfer_config = TransferConfig(
            multipart_threshold=config.MULTIPART_THRESHOLD,
            multipart_chunksize=config.MULTIPART_CHUNKSIZE,
            max_concurrency=config.MAX_WORKERS,
            use_threads=True
        )

        client = session.client("s3")
        client._transfer_config = transfer_config
        return client


def collect_files_to_upload() -> Dict[str, List[Tuple[Path, str, int]]]:
    """
    Collect all files to upload from source directories

    Returns:
        Dict mapping source_type -> [(file_path, relative_path, file_size), ...]
    """
    files_by_type = {
        'captured_frames': [],
        'exports': [],
        'logs': []
    }

    # Collect captured_frames (images only)
    if config.CAPTURED_FRAMES_DIR.exists():
        logger.info(f"Scanning {config.CAPTURED_FRAMES_DIR}")
        for root, dirs, files in os.walk(config.CAPTURED_FRAMES_DIR):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext in config.CAPTURED_FRAMES_EXTENSIONS:
                    file_path = Path(root) / filename
                    try:
                        size = file_path.stat().st_size
                        relative = file_path.relative_to(config.CAPTURED_FRAMES_DIR)
                        files_by_type['captured_frames'].append((file_path, str(relative), size))
                    except Exception as e:
                        logger.error(f"Cannot access {file_path}: {e}")

    # Collect exports (.csv, .json, .xlsx)
    if config.EXPORTS_DIR.exists():
        logger.info(f"Scanning {config.EXPORTS_DIR}")
        for root, dirs, files in os.walk(config.EXPORTS_DIR):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext in config.EXPORTS_EXTENSIONS:
                    file_path = Path(root) / filename
                    try:
                        size = file_path.stat().st_size
                        relative = file_path.relative_to(config.EXPORTS_DIR)
                        files_by_type['exports'].append((file_path, str(relative), size))
                    except Exception as e:
                        logger.error(f"Cannot access {file_path}: {e}")

    # Collect logs (.log only)
    if config.LOGS_DIR.exists():
        logger.info(f"Scanning {config.LOGS_DIR}")
        for root, dirs, files in os.walk(config.LOGS_DIR):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext in config.LOGS_EXTENSIONS:
                    file_path = Path(root) / filename
                    try:
                        size = file_path.stat().st_size
                        relative = file_path.relative_to(config.LOGS_DIR)
                        files_by_type['logs'].append((file_path, str(relative), size))
                    except Exception as e:
                        logger.error(f"Cannot access {file_path}: {e}")

    total_files = sum(len(files) for files in files_by_type.values())
    logger.info(f"Found {total_files} files to upload:")
    logger.info(f"  - captured_frames: {len(files_by_type['captured_frames'])} files")
    logger.info(f"  - exports: {len(files_by_type['exports'])} files")
    logger.info(f"  - logs: {len(files_by_type['logs'])} files")

    return files_by_type


def upload_file_to_s3(
    s3_client,
    file_path: Path,
    s3_key: str,
    file_size: int,
    stats: UploadStats,
    retry_count: int = 0
) -> bool:
    """Upload a single file to S3 with retry logic"""
    try:
        # Determine content type
        ext = file_path.suffix.lower()
        content_type_map = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
            '.gif': 'image/gif', '.bmp': 'image/bmp', '.tiff': 'image/tiff',
            '.webp': 'image/webp', '.csv': 'text/csv', '.json': 'application/json',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.xls': 'application/vnd.ms-excel', '.log': 'text/plain'
        }
        content_type = content_type_map.get(ext, 'application/octet-stream')

        if file_size > config.MULTIPART_THRESHOLD:
            logger.info(f"Uploading (multipart) {s3_key} ({file_size / 1024 / 1024:.2f}MB)")
        else:
            logger.info(f"Uploading {s3_key} ({file_size / 1024:.2f}KB)")

        with open(file_path, 'rb') as f:
            s3_client.upload_fileobj(
                f,
                config.S3_BUCKET,
                s3_key,
                ExtraArgs={
                    'ServerSideEncryption': 'AES256',
                    'ContentType': content_type,
                    'Metadata': {
                        'original_path': str(file_path),
                        'upload_timestamp': datetime.now().isoformat()
                    }
                },
                Config=s3_client._transfer_config
            )

        logger.info(f"✅ Successfully uploaded: {s3_key}")
        stats.increment_uploaded(str(file_path), file_size)
        return True

    except Exception as e:
        logger.error(f"❌ Failed to upload {s3_key}: {e}")

        if retry_count < config.MAX_RETRIES:
            logger.info(f"Retrying {s3_key} (attempt {retry_count + 1}/{config.MAX_RETRIES})")
            time.sleep(config.RETRY_DELAY * (retry_count + 1))
            new_client = fetch_s3_client()
            if new_client:
                return upload_file_to_s3(new_client, file_path, s3_key, file_size, stats, retry_count + 1)

        stats.increment_failed(str(file_path))
        return False


def upload_all_files(files_by_type: Dict[str, List[Tuple]], stats: UploadStats) -> bool:
    """Upload all collected files to S3"""
    today = datetime.now().strftime('%Y-%m-%d')
    s3_client = fetch_s3_client()

    if not s3_client:
        logger.error("Failed to get S3 client")
        return False

    logger.info(f"\n{'='*60}")
    logger.info(f"Starting upload to S3 (Date prefix: {today})")
    logger.info(f"{'='*60}\n")

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        futures = []

        for source_type, files in files_by_type.items():
            for file_path, relative_path, file_size in files:
                # Build S3 key: YYYY-MM-DD/captured_frames/subfolder/file.jpg
                s3_key = f"{today}/{source_type}/{relative_path}"

                future = executor.submit(
                    upload_file_to_s3,
                    s3_client,
                    file_path,
                    s3_key,
                    file_size,
                    stats
                )
                futures.append(future)

        # Wait for all uploads
        for future in futures:
            try:
                future.result()
            except Exception as e:
                logger.error(f"Unexpected error: {e}")

    return stats.failed == 0


def move_to_archive(uploaded_files: Set[str]):
    """Move successfully uploaded files to archive with date-based structure"""
    today = datetime.now().strftime('%Y-%m-%d')
    archive_date_dir = config.ARCHIVE_ROOT / today
    archive_date_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"Moving uploaded files to archive: {archive_date_dir}")
    logger.info(f"{'='*60}\n")

    moved_count = 0

    for file_path_str in uploaded_files:
        file_path = Path(file_path_str)

        if not file_path.exists():
            logger.warning(f"File no longer exists: {file_path}")
            continue

        try:
            # Determine source type and build archive path
            if str(config.CAPTURED_FRAMES_DIR) in str(file_path):
                relative = file_path.relative_to(config.CAPTURED_FRAMES_DIR)
                archive_path = archive_date_dir / "captured_frames" / relative
            elif str(config.EXPORTS_DIR) in str(file_path):
                relative = file_path.relative_to(config.EXPORTS_DIR)
                archive_path = archive_date_dir / "exports" / relative
            elif str(config.LOGS_DIR) in str(file_path):
                relative = file_path.relative_to(config.LOGS_DIR)
                archive_path = archive_date_dir / "logs" / relative
            else:
                logger.warning(f"Unknown source for file: {file_path}")
                continue

            # Create parent directory
            archive_path.parent.mkdir(parents=True, exist_ok=True)

            # Move file
            shutil.move(str(file_path), str(archive_path))
            logger.info(f"Moved to archive: {file_path.name} -> {archive_path}")
            moved_count += 1

        except Exception as e:
            logger.error(f"Failed to move {file_path} to archive: {e}")

    logger.info(f"Moved {moved_count} files to archive")


def cleanup_old_archives():
    """Delete archives older than ARCHIVE_RETENTION_DAYS"""
    if not config.ARCHIVE_ROOT.exists():
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"Cleaning up archives older than {config.ARCHIVE_RETENTION_DAYS} days")
    logger.info(f"{'='*60}\n")

    cutoff_date = datetime.now() - timedelta(days=config.ARCHIVE_RETENTION_DAYS)
    deleted_count = 0

    for date_dir in config.ARCHIVE_ROOT.iterdir():
        if not date_dir.is_dir():
            continue

        try:
            # Parse date from directory name (YYYY-MM-DD)
            dir_date = datetime.strptime(date_dir.name, '%Y-%m-%d')

            if dir_date < cutoff_date:
                logger.info(f"Deleting old archive: {date_dir.name}")
                shutil.rmtree(date_dir)
                deleted_count += 1
        except ValueError:
            logger.warning(f"Skipping non-date directory: {date_dir.name}")
        except Exception as e:
            logger.error(f"Failed to delete {date_dir}: {e}")

    logger.info(f"Deleted {deleted_count} old archive directories")


def run_upload_job():
    """Main upload job with retry logic"""
    global logger
    logger = setup_logger()

    logger.info("="*60)
    logger.info("S3 SCHEDULED UPLOAD JOB STARTED")
    logger.info("="*60)
    logger.info(f"Start time: {datetime.now()}")
    logger.info(f"Retry until: {config.JOB_DEADLINE_TIME}")
    logger.info(f"Retry interval: {config.JOB_RETRY_INTERVAL} minutes")
    logger.info("")

    # Collect files
    files_by_type = collect_files_to_upload()
    total_files = sum(len(files) for files in files_by_type.values())

    if total_files == 0:
        logger.info("No files to upload. Job complete.")
        return True

    # Initialize stats
    stats = UploadStats()
    stats.total_files = total_files
    stats.total_bytes = sum(
        size for files in files_by_type.values()
        for _, _, size in files
    )

    logger.info(f"Total files: {stats.total_files}")
    logger.info(f"Total size: {stats.total_bytes / 1024 / 1024:.2f} MB")
    logger.info("")

    # Retry logic: 1am - 4am, every 10 minutes
    retry_count = 0
    deadline = datetime.strptime(config.JOB_DEADLINE_TIME, '%H:%M').time()

    while retry_count <= config.JOB_MAX_RETRIES:
        current_time = datetime.now().time()

        # Check if past deadline
        if current_time > deadline:
            logger.warning(f"Deadline ({config.JOB_DEADLINE_TIME}) exceeded. Stopping retries.")
            break

        if retry_count > 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"RETRY ATTEMPT {retry_count}/{config.JOB_MAX_RETRIES}")
            logger.info(f"{'='*60}\n")

        # Attempt upload
        success = upload_all_files(files_by_type, stats)

        # Get results
        final_stats = stats.get_stats()

        if success or final_stats['failed'] == 0:
            logger.info("\n" + "="*60)
            logger.info("UPLOAD SUCCESSFUL!")
            logger.info("="*60)
            break

        # If failed, prepare for retry with only failed files
        if retry_count < config.JOB_MAX_RETRIES:
            failed_files = set(final_stats['failed_files'])
            logger.warning(f"{len(failed_files)} files failed. Will retry in {config.JOB_RETRY_INTERVAL} minutes...")

            # Update files_by_type to only include failed files
            for source_type in files_by_type:
                files_by_type[source_type] = [
                    (fp, rp, size) for fp, rp, size in files_by_type[source_type]
                    if str(fp) in failed_files
                ]

            # Sleep before retry
            time.sleep(config.JOB_RETRY_INTERVAL * 60)
            retry_count += 1
        else:
            break

    # Final statistics
    final_stats = stats.get_stats()
    logger.info("\n" + "="*60)
    logger.info("UPLOAD JOB COMPLETE")
    logger.info("="*60)
    logger.info(f"Total files: {final_stats['total_files']}")
    logger.info(f"Successfully uploaded: {final_stats['uploaded']}")
    logger.info(f"Failed: {final_stats['failed']}")
    logger.info(f"Total data uploaded: {final_stats['uploaded_bytes'] / 1024 / 1024:.2f} MB")
    logger.info(f"Total time: {final_stats['elapsed_seconds'] / 60:.1f} minutes")
    logger.info(f"Retry attempts: {retry_count}")
    logger.info("="*60 + "\n")

    # Move uploaded files to archive
    if final_stats['uploaded'] > 0:
        move_to_archive(set(final_stats['uploaded_files']))

    # Cleanup old archives
    cleanup_old_archives()

    logger.info("\n" + "="*60)
    logger.info("JOB FINISHED")
    logger.info("="*60)
    logger.info(f"End time: {datetime.now()}")

    return final_stats['failed'] == 0


if __name__ == "__main__":
    try:
        success = run_upload_job()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        if logger:
            logger.info("\n\nJob interrupted by user")
        sys.exit(130)
    except Exception as e:
        if logger:
            logger.error(f"Unexpected error: {e}")
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)
