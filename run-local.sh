#!/bin/bash

# Intelligent Returns Classifier - Local Run Script
# This script starts both backend and frontend services (local only)

set -e  # Exit on error

echo "========================================="
echo "Starting Intelligent Returns Classifier"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Kill any other existing instances
echo -e "${YELLOW}Killing any existing backend/frontend instances...${NC}"
pkill -9 -f "python.*src/api/main.py" 2>/dev/null || true
pkill -9 -f "python.*run_backend.py" 2>/dev/null || true
pkill -9 -f "python.*http.server.*8080" 2>/dev/null || true
pkill -9 -f "python.*-m.*http.server" 2>/dev/null || true
pkill -9 -f "uvicorn" 2>/dev/null || true
# Also kill any process using port 8080
lsof -ti:8080 | xargs -r kill -9 2>/dev/null || true
lsof -ti:8000 | xargs -r kill -9 2>/dev/null || true
sleep 1

# Check if Ollama is running
echo -e "${YELLOW}Checking Ollama status...${NC}"
if ! pgrep -x "ollama" > /dev/null; then
    echo -e "${YELLOW}Starting Ollama service...${NC}"
    ollama serve > /dev/null 2>&1 &
    sleep 2
fi

# Check if model is available
if ! ollama list | grep -q "qwen2.5vl:3b"; then
    echo -e "${YELLOW}Pulling Qwen2.5-VL model (this may take a while)...${NC}"
    ollama pull qwen2.5vl:3b
fi

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"
    
    # Kill backend if running
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    
    # Kill frontend server if running
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    
    echo -e "${GREEN}Services stopped successfully${NC}"
    exit 0
}

# Set trap for cleanup on Ctrl+C
trap cleanup INT TERM

# Activate virtual environment if exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source venv/bin/activate
fi

# Clear ALL logs, captured frames, and session data
echo -e "${YELLOW}Clearing all logs, captured frames, and session data...${NC}"

# Clear all log files
rm -f backend.log frontend.log 2>/dev/null || true
rm -f backend/*.log 2>/dev/null || true
rm -f *.log 2>/dev/null || true

# Clear all captured frames - including the specific debug directory
rm -rf captured_frames/* 2>/dev/null || true
rm -rf backend/captured_frames/* 2>/dev/null || true
rm -rf /tmp/captured_frames/* 2>/dev/null || true
# Clear the specific debug frames directory
rm -rf /home/denaliai/RELO-CLASSIFIER-DEV/RELO-DEMO-APP-test-RELO-DEV-APP-test/captured_frames/* 2>/dev/null || true
# Recreate the directory structure
mkdir -p /home/denaliai/RELO-CLASSIFIER-DEV/RELO-DEMO-APP-test-RELO-DEV-APP-test/captured_frames 2>/dev/null || true
mkdir -p captured_frames 2>/dev/null || true

# Clear session data and checkpoints
rm -rf backend/data/sessions/* 2>/dev/null || true
rm -rf backend/data/checkpoints/* 2>/dev/null || true
rm -rf data/sessions/* 2>/dev/null || true
rm -rf data/checkpoints/* 2>/dev/null || true

# Clear any temporary files
rm -rf /tmp/relo-classifier-* 2>/dev/null || true
rm -rf /tmp/*.frame /tmp/*.jpg /tmp/*.png 2>/dev/null || true

# Clear Python cache to ensure fresh start
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Create new log files
touch backend.log frontend.log

# Start Backend
echo -e "${GREEN}Starting Backend Server...${NC}"
cd backend
python run_backend.py > ../backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Wait for backend to be ready
echo -e "${YELLOW}Waiting for backend to start...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:8000/api/health > /dev/null; then
        echo -e "${GREEN}Backend is ready!${NC}"
        break
    fi
    sleep 1
done

# Start Frontend HTTP Server
echo -e "${GREEN}Starting Frontend Server...${NC}"
python -m http.server 8080 --directory frontend > /dev/null 2>&1 &
FRONTEND_PID=$!

# Wait for frontend to be fully ready
echo -e "${YELLOW}Waiting for frontend server to be ready...${NC}"
sleep 3
for i in {1..10}; do
    if curl -s http://localhost:8080 > /dev/null; then
        echo -e "${GREEN}Frontend server is ready!${NC}"
        break
    fi
    sleep 1
done

# Display access information
echo ""
echo "========================================="
echo -e "${GREEN}Application Started Successfully!${NC}"
echo "========================================="
echo ""
echo "Access the application at:"
echo -e "  Local:  ${GREEN}http://localhost:8080${NC}"
echo ""
echo "Backend API documentation:"
echo -e "  ${GREEN}http://localhost:8000/docs${NC}"
echo ""
echo "Camera Support:"
if lsusb | grep -q "Intel Corp" || ls /dev/video* 2>/dev/null | grep -q "video"; then
    echo -e "  ${GREEN}✓ Camera detected${NC}"
else
    echo -e "  ${YELLOW}⚠ No camera detected - please connect a camera${NC}"
fi
echo ""
echo "Logs:"
echo -e "  Backend: ${GREEN}backend.log${NC}"
echo -e "  Frontend: ${GREEN}frontend.log${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo "========================================="

# Keep script running without showing logs in terminal
wait $BACKEND_PID
