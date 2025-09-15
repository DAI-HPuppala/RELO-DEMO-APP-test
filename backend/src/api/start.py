#!/usr/bin/env python3
"""
Startup script for the backend server with proper path configuration.
"""
import sys
import os
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

# Now import and run the main application
if __name__ == "__main__":
    import uvicorn
    
    # Use module string for reload to work properly
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload to avoid the warning
        log_level="info"
    )