#!/bin/bash

# Intelligent Returns Classifier - Run Script with Ngrok Support
# This script starts both backend and frontend services with ngrok tunneling

set -e  # Exit on error

echo "========================================="
echo "Starting Intelligent Returns Classifier"
echo "With Ngrok Remote Access Support"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

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
    
    # Kill ngrok if running
    if [ ! -z "$NGROK_PID" ]; then
        kill $NGROK_PID 2>/dev/null || true
    fi
    
    # Kill ngrok frontend if running
    if [ ! -z "$NGROK_FRONTEND_PID" ]; then
        kill $NGROK_FRONTEND_PID 2>/dev/null || true
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

# Clear previous logs and captured frames
echo -e "${YELLOW}Clearing previous logs and captured frames...${NC}"
: > backend.log
if [ -d "backend/captured_frames" ]; then
    rm -rf backend/captured_frames/* 2>/dev/null || true
fi

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

# Start ngrok tunnel for BACKEND (port 8000) - this is key for WebSocket
echo -e "${GREEN}Starting ngrok tunnel for backend (WebSocket)...${NC}"
ngrok http 8000 --log-level=info --log=stdout > ngrok-backend.log 2>&1 &
NGROK_PID=$!

# Wait for ngrok to start and get the backend URL
sleep 3
NGROK_BACKEND_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | grep -o '"public_url":"[^"]*' | grep -o 'https://[^"]*' | head -1)

# Create a config file with the ngrok backend URL for the frontend to use
echo -e "${YELLOW}Creating ngrok configuration...${NC}"
cat > frontend/ngrok-config.js << EOF
// Auto-generated ngrok configuration
window.NGROK_BACKEND_URL = '${NGROK_BACKEND_URL}';
EOF

# Display access information
echo ""
echo "========================================="
echo -e "${GREEN}Application Started Successfully!${NC}"
echo "========================================="
echo ""
echo "Access the application at:"
echo -e "  Local Frontend:  ${GREEN}http://localhost:8080${NC}"
echo -e "  Local Backend:   ${GREEN}http://localhost:8000${NC}"
if [ ! -z "$NGROK_BACKEND_URL" ]; then
    echo ""
    echo -e "  ${GREEN}Ngrok Backend URL: ${NGROK_BACKEND_URL}${NC}"
    echo ""
    echo "For remote access:"
    echo "1. Open the frontend at http://localhost:8080"
    echo "2. The frontend will automatically use the ngrok backend URL"
else
    echo -e "  ${YELLOW}Note: ngrok tunnel not available${NC}"
fi
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
echo -e "  Ngrok: ${GREEN}ngrok-backend.log${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo "========================================="

# Keep script running without showing logs in terminal
wait $BACKEND_PID