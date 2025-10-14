#!/bin/bash

# CV60 QR/Barcode Detector Run Script
# This script activates the virtual environment, clears logs, and runs the detector

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}CV60 QR/Barcode Detector${NC}"
echo -e "${GREEN}========================================${NC}"

# Get project root directory (one level up from bin/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Parse arguments
MODE="${1:-barcode}"  # Default to barcode mode
CAMERA_IP="${CV60_IP:-192.168.1.21}"

# Show usage if --help
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "Usage: $0 [mode] [camera_ip]"
    echo ""
    echo "Modes:"
    echo "  oakd       - OAK-D PoE camera @ 169.254.1.222 with 4K, autofocus (recommended)"
    echo "  oakd-auto  - OAK-D with AUTO-DETECTION (continuously scans every N frames)"
    echo "  oakd-auto-detection - OAK-D with LIGHTWEIGHT auto-detection (2-stage: fast check + async decode)"
    echo "  barcode    - Barcode-specific detector with optimized preprocessing (default)"
    echo "  cv60-auto  - CV60 with AUTO-DETECTION (continuously scans every N frames)"
    echo "  cv60-auto-detection - CV60 with LIGHTWEIGHT auto-detection (2-stage: fast check + async decode)"
    echo "  fast       - Manual trigger mode: 30fps display, press '+' to detect QR/barcode"
    echo "  auto       - Automatic detection every N frames at 30fps"
    echo "  enhanced   - Enhanced mode with all detection methods (slower)"
    echo "  test       - Test mode with fallback camera"
    echo ""
    echo "Examples:"
    echo "  $0 oakd               # Run with OAK-D camera (4K, autofocus)"
    echo "  $0 oakd-auto          # OAK-D with auto-detection"
    echo "  $0 oakd-auto-detection # OAK-D with lightweight auto-detection"
    echo "  $0                    # Run barcode scanner with default IP"
    echo "  $0 barcode 192.168.1.21  # Run barcode scanner with specific IP"
    echo "  $0 cv60-auto          # CV60 with auto-detection"
    echo "  $0 cv60-auto-detection # CV60 with lightweight auto-detection"
    echo "  $0 fast               # Run manual QR/barcode mode"
    echo "  $0 auto               # Run automatic detection mode"
    echo ""
    echo "Environment variables:"
    echo "  CV60_IP - Override default camera IP"
    exit 0
fi

# Override camera IP if provided as second argument
if [ ! -z "$2" ]; then
    CAMERA_IP="$2"
fi

# Clear logs
echo -e "${YELLOW}Clearing previous logs...${NC}"
mkdir -p logs
rm -f logs/*.log
echo -e "${GREEN}Logs cleared${NC}"

# Check if barcode-env exists
if [ ! -d "barcode-env" ]; then
    echo -e "${RED}Error: barcode-env virtual environment not found!${NC}"
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv barcode-env
    source barcode-env/bin/activate
    pip install --upgrade pip
    pip install "numpy==1.24.3" "opencv-python==4.8.1.78" "opencv-contrib-python==4.8.1.78" pyzbar qrcode[pil]
else
    # Activate virtual environment
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source barcode-env/bin/activate
fi

# Check if pyzbar works
echo -e "${YELLOW}Checking barcode detection libraries...${NC}"
python3 -c "from pyzbar import pyzbar; print('✓ pyzbar loaded successfully')" 2>/dev/null || {
    echo -e "${RED}Warning: pyzbar not working properly${NC}"
    echo -e "${YELLOW}Library zbar0 may not be installed${NC}"
}

python3 -c "import cv2; print('✓ OpenCV loaded successfully')" 2>/dev/null || {
    echo -e "${RED}Error: OpenCV not found${NC}"
    exit 1
}

# Check for OAK-D mode (including auto mode)
if [[ "$MODE" == "oakd" ]] || [[ "$MODE" == "OAKD" ]] || [[ "$MODE" == "oakd-auto" ]] || [[ "$MODE" == "OAKD-AUTO" ]]; then
    python3 -c "import depthai; print('✓ DepthAI loaded successfully')" 2>/dev/null || {
        echo -e "${RED}Error: DepthAI module not found${NC}"
        echo -e "${YELLOW}Install with: pip install depthai==2.24.0.0${NC}"
        exit 1
    }

    # Setup OAK-D network if needed
    OAKD_IP="169.254.1.222"
    echo -e "${YELLOW}Checking OAK-D camera network setup...${NC}"

    # Check if camera is reachable
    if ! ping -c 1 -W 1 "$OAKD_IP" > /dev/null 2>&1; then
        echo -e "${YELLOW}OAK-D camera not reachable, configuring network...${NC}"

        # Find which interface has the camera
        OAKD_IFACE=""
        for iface in $(ip link show | grep -E "^[0-9]+: enp" | cut -d: -f2 | tr -d ' '); do
            # Skip if interface is down
            if ! ip link show "$iface" | grep -q "state UP"; then
                continue
            fi

            # Try to ping from this interface
            timeout 2 ping -c 1 -W 1 -I "$iface" "$OAKD_IP" > /dev/null 2>&1

            # Check ARP for camera
            if arp -i "$iface" -a 2>/dev/null | grep -q "$OAKD_IP"; then
                OAKD_IFACE="$iface"
                echo -e "${GREEN}Found OAK-D camera on interface: $OAKD_IFACE${NC}"
                break
            fi
        done

        if [ -z "$OAKD_IFACE" ]; then
            echo -e "${YELLOW}⚠ Could not auto-detect OAK-D interface${NC}"
            echo -e "${YELLOW}Trying both enp2s0 and enp3s0...${NC}"
            # Try both common interfaces
            for iface in enp2s0 enp3s0; do
                if ip link show "$iface" 2>/dev/null | grep -q "state UP"; then
                    OAKD_IFACE="$iface"
                    break
                fi
            done
        fi

        if [ ! -z "$OAKD_IFACE" ]; then
            # Configure link-local IP on the interface
            if ! ip addr show "$OAKD_IFACE" 2>/dev/null | grep -q "inet.*169.254.1.1"; then
                echo -e "${YELLOW}Configuring $OAKD_IFACE with 169.254.1.1/16...${NC}"
                echo denaliai | sudo -S ip addr add 169.254.1.1/16 dev "$OAKD_IFACE" 2>&1 | grep -v "File exists" | grep -v "password for"
                sleep 2
            fi

            # Verify connectivity
            if ping -c 1 -W 1 "$OAKD_IP" > /dev/null 2>&1; then
                echo -e "${GREEN}✓ OAK-D camera is now reachable at $OAKD_IP${NC}"
            else
                echo -e "${YELLOW}⚠ Camera detected but not yet pingable${NC}"
                echo -e "${YELLOW}  Waiting for network to stabilize...${NC}"
                sleep 3
            fi
        else
            echo -e "${YELLOW}⚠ Could not configure network automatically${NC}"
        fi
    else
        echo -e "${GREEN}✓ OAK-D camera is reachable at $OAKD_IP${NC}"
    fi
fi

echo -e "${YELLOW}Camera IP: ${CAMERA_IP}${NC}"
echo -e "${YELLOW}Mode: ${MODE}${NC}"

# Ping camera to check connectivity (skip for OAK-D as we already checked)
if [[ "$MODE" != "oakd" ]] && [[ "$MODE" != "OAKD" ]] && [[ "$MODE" != "oakd-auto" ]] && [[ "$MODE" != "OAKD-AUTO" ]]; then
    echo -e "${YELLOW}Checking camera connectivity...${NC}"
    if ping -c 1 -W 1 "$CAMERA_IP" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Camera is reachable at ${CAMERA_IP}${NC}"
    else
        echo -e "${RED}⚠ Warning: Cannot reach camera at ${CAMERA_IP}${NC}"
        echo -e "${YELLOW}The detector will attempt to connect anyway...${NC}"
    fi
fi

# Run the appropriate detector based on mode
echo -e "${GREEN}========================================${NC}"
case "$MODE" in
    oakd|OAKD)
        echo -e "${GREEN}Starting OAK-D PoE Camera Detector (4K, 30 FPS)...${NC}"
        echo -e "${BLUE}Connecting to: 169.254.1.222 (PoE)${NC}"
        echo -e "${BLUE}Features: 4K resolution, autofocus, auto-exposure${NC}"
        echo -e "${BLUE}Detection: pyzbar + OpenCV QR detector${NC}"
        echo -e "${BLUE}Press 'q' or ESC to quit, 'd' to toggle debug, 'c' to clear history${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        python3 scripts/oakd/robust_qr_detector.py
        ;;
    barcode|BARCODE)
        echo -e "${GREEN}Starting Barcode Scanner (30 FPS)...${NC}"
        echo -e "${BLUE}Optimized for linear barcodes (Code128, Code39, EAN, etc.)${NC}"
        echo -e "${BLUE}Press '+' to scan for barcodes${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        python3 scripts/cv60/barcode_detector.py --ip "$CAMERA_IP"
        ;;
    cv60-auto|CV60-AUTO)
        echo -e "${GREEN}Starting CV60 AUTO Barcode Scanner (30 FPS)...${NC}"
        echo -e "${BLUE}Camera: CV60 at ${CAMERA_IP}${NC}"
        echo -e "${BLUE}Automatically scans every 5 frames${NC}"
        echo -e "${BLUE}Press '+/-' to adjust scan frequency, 'q' to quit${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        python3 scripts/cv60/auto_barcode_detector.py --camera cv60 --ip "$CAMERA_IP" --interval 5
        ;;
    oakd-auto|OAKD-AUTO)
        echo -e "${GREEN}Starting OAK-D AUTO Barcode Scanner (30 FPS)...${NC}"
        echo -e "${BLUE}Camera: OAK-D PoE at 169.254.1.222${NC}"
        echo -e "${BLUE}Automatically scans every 5 frames${NC}"
        echo -e "${BLUE}Press '+/-' to adjust scan frequency, 'q' to quit${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        python3 scripts/cv60/auto_barcode_detector.py --camera oakd --oak-ip 169.254.1.222 --interval 5
        ;;
    cv60-auto-detection|CV60-AUTO-DETECTION)
        echo -e "${GREEN}Starting CV60 LIGHTWEIGHT AUTO Detection (30 FPS)...${NC}"
        echo -e "${BLUE}Camera: CV60 at ${CAMERA_IP}${NC}"
        echo -e "${BLUE}Two-stage detection: Fast presence check + Async decode${NC}"
        echo -e "${BLUE}Stage 1: ~2-5ms gradient analysis (every frame)${NC}"
        echo -e "${BLUE}Stage 2: Heavy decode only when barcode detected (async)${NC}"
        echo -e "${BLUE}Press 'q' to quit, 'c' to clear history${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        python3 scripts/cv60/auto_lightweight_detector.py --camera cv60 --ip "$CAMERA_IP"
        ;;
    oakd-auto-detection|OAKD-AUTO-DETECTION)
        echo -e "${GREEN}Starting OAK-D LIGHTWEIGHT AUTO Detection (30 FPS)...${NC}"
        echo -e "${BLUE}Camera: OAK-D PoE at 169.254.1.222${NC}"
        echo -e "${BLUE}Two-stage detection: Fast presence check + Async decode${NC}"
        echo -e "${BLUE}Stage 1: ~2-5ms gradient analysis (every frame)${NC}"
        echo -e "${BLUE}Stage 2: Heavy decode only when barcode detected (async)${NC}"
        echo -e "${BLUE}Press 'q' to quit, 'c' to clear history${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        python3 scripts/cv60/auto_lightweight_detector.py --camera oakd --oak-ip 169.254.1.222
        ;;
    fast|FAST)
        echo -e "${GREEN}Starting Manual Trigger Detector (30 FPS)...${NC}"
        echo -e "${BLUE}Press '+' to capture frame and run ALL detection methods${NC}"
        echo -e "${BLUE}Each capture will show results from all available detectors${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        python3 scripts/cv60/manual_mode.py --ip "$CAMERA_IP"
        ;;
    auto|AUTO)
        echo -e "${GREEN}Starting Automatic Detection Mode (30 FPS)...${NC}"
        echo -e "${BLUE}Detection runs every 5 frames automatically${NC}"
        echo -e "${BLUE}Use '+' and '-' keys to adjust detection frequency${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        python3 scripts/cv60/fast_mode.py --ip "$CAMERA_IP" --interval 5
        ;;
    enhanced|ENHANCED)
        echo -e "${GREEN}Starting Enhanced QR/Barcode Detector...${NC}"
        echo -e "${BLUE}All detection methods enabled (slower)${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        python3 scripts/cv60/enhanced_mode.py --ip "$CAMERA_IP"
        ;;
    test|TEST)
        echo -e "${GREEN}Starting Test Mode with Fallback...${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        python3 deprecated/cv60_qr_with_fallback.py --ip "$CAMERA_IP" --fallback
        ;;
    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo "Use: oakd, oakd-auto, oakd-auto-detection, barcode, cv60-auto, cv60-auto-detection, fast, auto, enhanced, or test"
        exit 1
        ;;
esac

# Deactivate virtual environment
deactivate

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Detection session completed${NC}"
echo -e "${GREEN}========================================${NC}"