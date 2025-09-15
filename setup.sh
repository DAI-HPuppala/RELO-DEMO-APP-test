#!/bin/bash

# Intelligent Returns Classifier - Setup Script
# This script installs all dependencies and prepares the environment

set -e  # Exit on error

echo "========================================="
echo "Setting up Intelligent Returns Classifier"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Check Python version
echo -e "${BLUE}Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
REQUIRED_VERSION="3.11"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}Error: Python 3.11+ is required (found $PYTHON_VERSION)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating Python virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip > /dev/null 2>&1

# Install backend dependencies
echo -e "${YELLOW}Installing backend dependencies...${NC}"
cd backend
pip install -r requirements.txt
cd ..
echo -e "${GREEN}✓ Backend dependencies installed${NC}"

# Check for Intel RealSense SDK (optional)
echo -e "${BLUE}Checking for Intel RealSense SDK...${NC}"
if command -v rs-enumerate-devices &> /dev/null; then
    echo -e "${GREEN}✓ Intel RealSense SDK detected${NC}"
else
    echo -e "${YELLOW}⚠ Intel RealSense SDK not found${NC}"
    echo "  To install (Ubuntu/Debian):"
    echo "    sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE"
    echo "    sudo add-apt-repository 'deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main'"
    echo "    sudo apt-get update"
    echo "    sudo apt-get install librealsense2-dkms librealsense2-utils"
    echo "  The application will use webcam as fallback"
fi

# Check for Ollama
echo -e "${BLUE}Checking for Ollama...${NC}"
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✓ Ollama is installed${NC}"
    
    # Check if Ollama is running
    if pgrep -x "ollama" > /dev/null; then
        echo -e "${GREEN}✓ Ollama service is running${NC}"
    else
        echo -e "${YELLOW}Starting Ollama service...${NC}"
        ollama serve > /dev/null 2>&1 &
        sleep 2
    fi
    
    # Check for required model
    echo -e "${YELLOW}Checking for Qwen2.5-VL model...${NC}"
    if ollama list 2>/dev/null | grep -q "qwen2.5vl:3b"; then
        echo -e "${GREEN}✓ Qwen2.5-VL model is available${NC}"
    else
        echo -e "${YELLOW}Pulling Qwen2.5-VL model (this will take a while)...${NC}"
        ollama pull qwen2.5vl:3b
        echo -e "${GREEN}✓ Model downloaded successfully${NC}"
    fi
else
    echo -e "${RED}✗ Ollama not installed${NC}"
    echo "  To install Ollama:"
    echo "    curl -fsSL https://ollama.ai/install.sh | sh"
    echo ""
    echo -e "${RED}Please install Ollama and run setup again${NC}"
    exit 1
fi

# Create necessary directories
echo -e "${YELLOW}Creating necessary directories...${NC}"
mkdir -p backend/exports 2>/dev/null || true
mkdir -p backend/logs 2>/dev/null || true
mkdir -p backend/temp 2>/dev/null || true
echo -e "${GREEN}✓ Directories created${NC}"

# Run system test
echo ""
echo -e "${BLUE}Running system test...${NC}"
python test_system.py

# Display completion message
echo ""
echo "========================================="
echo -e "${GREEN}Setup completed successfully!${NC}"
echo "========================================="
echo ""
echo "To start the application, run:"
echo -e "  ${GREEN}./run.sh${NC}"
echo ""
echo "To stop the application, run:"
echo -e "  ${GREEN}./stop.sh${NC}"
echo ""
echo "For development with virtual environment:"
echo -e "  ${GREEN}source venv/bin/activate${NC}"
echo ""
echo "========================================="