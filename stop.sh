#!/bin/bash

# Intelligent Returns Classifier - Stop Script
# This script stops all running services

echo "========================================="
echo "Stopping Intelligent Returns Classifier"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Kill Python backend processes
echo -e "${YELLOW}Stopping backend server...${NC}"
pkill -f "python.*src/api/main.py" 2>/dev/null || true
pkill -f "python.*src/api/start.py" 2>/dev/null || true
pkill -f "python.*run_backend.py" 2>/dev/null || true
pkill -f "uvicorn" 2>/dev/null || true

# Kill frontend HTTP server
echo -e "${YELLOW}Stopping frontend server...${NC}"
pkill -f "python.*http.server.*8080" 2>/dev/null || true

# Kill any WebRTC related processes
pkill -f "aiortc" 2>/dev/null || true

# Optional: Stop Ollama (commented out by default as it might be used by other apps)
# echo -e "${YELLOW}Stopping Ollama...${NC}"
# pkill -f "ollama" 2>/dev/null || true

# Clean up any temporary files
echo -e "${YELLOW}Cleaning up temporary files...${NC}"
rm -rf /tmp/relo-classifier-* 2>/dev/null || true

# Check if processes are stopped
sleep 1
if pgrep -f "python.*src/api/main.py" > /dev/null; then
    echo -e "${RED}Warning: Backend may still be running${NC}"
else
    echo -e "${GREEN}✓ Backend stopped${NC}"
fi

if pgrep -f "python.*http.server.*8080" > /dev/null; then
    echo -e "${RED}Warning: Frontend server may still be running${NC}"
else
    echo -e "${GREEN}✓ Frontend stopped${NC}"
fi

echo ""
echo -e "${GREEN}All services stopped successfully!${NC}"
echo "========================================="