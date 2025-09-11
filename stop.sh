#!/bin/bash

# Intelligent Returns Classifier - Stop Script
# This script stops all running services and cleans up all data

echo "========================================="
echo "Stopping Intelligent Returns Classifier"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# CRITICAL: Kill ALL ngrok instances first (multiple methods)
echo -e "${YELLOW}Forcefully stopping ALL ngrok tunnels...${NC}"
# Method 1: Kill by name patterns
pkill -9 -f ngrok 2>/dev/null || true
killall -9 ngrok 2>/dev/null || true
# Method 2: Kill by port usage
lsof -ti:4040 | xargs -r kill -9 2>/dev/null || true
lsof -ti:4041 | xargs -r kill -9 2>/dev/null || true
# Method 3: Find and kill any remaining ngrok processes
for pid in $(ps aux | grep '[n]grok' | awk '{print $2}'); do
    kill -9 $pid 2>/dev/null || true
done

# Kill ALL Python backend processes (force kill duplicates)
echo -e "${YELLOW}Stopping all backend server instances...${NC}"
pkill -9 -f "python.*src/api/main.py" 2>/dev/null || true
pkill -9 -f "python.*src/api/start.py" 2>/dev/null || true
pkill -9 -f "python.*run_backend.py" 2>/dev/null || true
pkill -9 -f "uvicorn" 2>/dev/null || true
pkill -9 -f "python.*backend" 2>/dev/null || true
lsof -ti:8000 | xargs -r kill -9 2>/dev/null || true

# Kill ALL frontend HTTP server instances
echo -e "${YELLOW}Stopping all frontend server instances...${NC}"
pkill -9 -f "python.*http.server.*8080" 2>/dev/null || true
pkill -9 -f "python.*-m.*http.server" 2>/dev/null || true
lsof -ti:8080 | xargs -r kill -9 2>/dev/null || true

# Kill any WebRTC related processes
pkill -9 -f "aiortc" 2>/dev/null || true

# Optional: Stop Ollama (commented out by default as it might be used by other apps)
# echo -e "${YELLOW}Stopping Ollama...${NC}"
# pkill -f "ollama" 2>/dev/null || true

# Clean up logs
echo -e "${YELLOW}Cleaning up logs...${NC}"
rm -f backend.log 2>/dev/null || true
rm -f frontend.log 2>/dev/null || true
rm -f ngrok.log 2>/dev/null || true
rm -f backend/*.log 2>/dev/null || true
rm -f *.log 2>/dev/null || true

# Clean up captured frames
echo -e "${YELLOW}Cleaning up captured frames...${NC}"
rm -rf captured_frames/* 2>/dev/null || true
rm -rf backend/captured_frames/* 2>/dev/null || true
rm -rf /tmp/captured_frames/* 2>/dev/null || true

# Clean up session data and checkpoints
echo -e "${YELLOW}Cleaning up session data...${NC}"
rm -rf backend/data/sessions/* 2>/dev/null || true
rm -rf backend/data/checkpoints/* 2>/dev/null || true
rm -rf data/sessions/* 2>/dev/null || true
rm -rf data/checkpoints/* 2>/dev/null || true

# Clean up any temporary files
echo -e "${YELLOW}Cleaning up temporary files...${NC}"
rm -rf /tmp/relo-classifier-* 2>/dev/null || true
rm -rf /tmp/*.frame 2>/dev/null || true
rm -rf /tmp/*.jpg 2>/dev/null || true
rm -rf /tmp/*.png 2>/dev/null || true

# Clean up Python cache
echo -e "${YELLOW}Cleaning up Python cache...${NC}"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

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