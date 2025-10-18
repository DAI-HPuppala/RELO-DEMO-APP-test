#!/usr/bin/env python3
"""
One-Time S3 Upload Script

Use this for:
- Manual uploads
- Testing
- One-off data transfers

This script uploads immediately without scheduling or retry logic.
For scheduled daily uploads, use s3_uploader_scheduled.py instead.
"""

import sys
from pathlib import Path

# Import the main uploader
sys.path.append(str(Path(__file__).parent))
from s3_uploader_scheduled import run_upload_job

if __name__ == "__main__":
    print("="*60)
    print("ONE-TIME S3 UPLOAD")
    print("="*60)
    print("This will upload all files NOW (no scheduling)")
    print("")

    try:
        success = run_upload_job()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nUpload interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
