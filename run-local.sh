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

# Load environment variables to check MODEL_PROVIDER
if [ -f "backend/.env" ]; then
    set -a  # Auto-export all variables
    source <(grep -v '^#' backend/.env | grep -v '^[[:space:]]*$')
    set +a  # Disable auto-export
fi

MODEL_PROVIDER=${MODEL_PROVIDER:-ollama}

# Start appropriate model server based on provider
if [ "$MODEL_PROVIDER" = "huggingface" ]; then
    echo -e "${YELLOW}Using HuggingFace provider...${NC}"
    echo -e "${YELLOW}Checking HuggingFace model server status...${NC}"

    # Kill any existing HuggingFace server
    pkill -9 -f "hf_model_server.py" 2>/dev/null || true
    lsof -ti:11435 | xargs -r kill -9 2>/dev/null || true
    sleep 1

    # Start HuggingFace model server
    echo -e "${GREEN}Starting HuggingFace model server...${NC}"
    cd backend
    python3 src/services/hf_model_server.py > ../hf_server.log 2>&1 &
    HF_SERVER_PID=$!
    cd ..

    # Wait for HuggingFace server to be ready
    echo -e "${YELLOW}Waiting for HuggingFace server to initialize (may take a while for first run)...${NC}"
    for i in {1..60}; do
        if curl -s http://localhost:11435/health > /dev/null 2>&1; then
            echo -e "${GREEN}HuggingFace server is ready!${NC}"
            break
        fi
        if [ $i -eq 60 ]; then
            echo -e "${RED}HuggingFace server failed to start. Check hf_server.log${NC}"
            exit 1
        fi
        sleep 2
    done

else
    echo -e "${YELLOW}Using Ollama provider...${NC}"
    # Check if Ollama is running
    echo -e "${YELLOW}Checking Ollama status...${NC}"
    if ! pgrep -x "ollama" > /dev/null; then
        echo -e "${YELLOW}Starting Ollama service...${NC}"
        ollama serve > /dev/null 2>&1 &
        sleep 2
    fi

    # Check if model is available
    OLLAMA_MODEL=${OLLAMA_MODEL:-qwen2.5vl:3b}
    if ! ollama list | grep -q "$OLLAMA_MODEL"; then
        echo -e "${YELLOW}Pulling $OLLAMA_MODEL model (this may take a while)...${NC}"
        ollama pull $OLLAMA_MODEL
    fi
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

    # Kill HuggingFace server if running
    if [ ! -z "$HF_SERVER_PID" ]; then
        kill $HF_SERVER_PID 2>/dev/null || true
    fi
    pkill -9 -f "hf_model_server.py" 2>/dev/null || true

    echo -e "${GREEN}Services stopped successfully${NC}"
    exit 0
}

# Set trap for cleanup on Ctrl+C
trap cleanup INT TERM

# Activate virtual environment if exists (check backend/venv)
if [ -d "backend/venv" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source backend/venv/bin/activate
fi

# Clear captured frames and session data (LOGS ARE NEVER DELETED)
echo -e "${YELLOW}Clearing captured frames and session data...${NC}"

# NOTE: Log files are NEVER deleted - they use time-based rotation
# Logs are stored in: backend/logs/session_YYYY-MM-DD_HH-MM-SS/
# Old session-based logging (backend.log, frontend.log) is DEPRECATED

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

# Setup camera hardware
echo -e "${YELLOW}Setting up camera hardware...${NC}"

# Setup OAK-D camera network if needed
source ./setup_oakd_network.sh

# Setup CV60 camera if needed
source ./setup_cv60_camera.sh

# Start Backend (logs go to backend/logs/session_*/relo_backend.log via Python logging)
echo -e "${GREEN}Starting Backend Server...${NC}"
cd backend
python3 run_backend.py > /dev/null 2>&1 &
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
python3 -m http.server 8080 --directory frontend > /dev/null 2>&1 &
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
echo -e "  Backend: ${GREEN}backend/logs/session_<timestamp>/${NC}"
echo -e "  View latest: ${GREEN}ls -lt backend/logs/*/relo_backend.log | head -1${NC}"
echo -e "  Tail active: ${GREEN}tail -f backend/logs/session_*/relo_backend.log${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo "========================================="

# Keep script running without showing logs in terminal
wait $BACKEND_PID
