#!/usr/bin/env python3
"""
Simple backend runner without complications.
"""
import sys
import os
from pathlib import Path

# Add src directory to Python path as the main package root
backend_dir = Path(__file__).parent
src_dir = backend_dir / "src"

# Make src the root for imports
sys.path.insert(0, str(src_dir))

if __name__ == "__main__":
    # Change working directory to backend to find config files
    os.chdir(str(backend_dir))
    
    import uvicorn
    from api.main import app
    
    print("Starting backend server...")
    print(f"Working directory: {os.getcwd()}")
    
    # Run the server directly
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )