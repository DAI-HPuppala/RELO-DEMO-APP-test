#!/bin/bash

###############################################################################
# CV60 Camera Setup & Verification Script
# This script ensures the CV60 camera is accessible before starting the backend
###############################################################################

set -e

CV60_IP="192.168.1.21"
CV60_PORT=3956

echo "================================================================================"
echo "CV60 CAMERA SETUP & VERIFICATION"
echo "================================================================================"

# Function to check if camera is pingable
check_camera_network() {
    echo "🔍 Checking network connectivity to CV60 camera at $CV60_IP..."
    if timeout 2 ping -c 1 -W 1 "$CV60_IP" > /dev/null 2>&1; then
        echo "  ✅ Camera is pingable"
        return 0
    else
        echo "  ❌ Camera is not pingable"
        return 1
    fi
}

# Function to kill blocking processes (eBUSPlayerJAI, old backend instances)
kill_blocking_processes() {
    echo "🔍 Checking for processes blocking camera access..."

    # Kill eBUSPlayerJAI if running
    if pgrep -f "eBUSPlayer" > /dev/null 2>&1; then
        echo "  ⚠️  Found eBUSPlayerJAI running - killing..."
        pkill -9 -f "eBUSPlayer" || true
        sleep 1
        echo "  ✅ Killed eBUSPlayerJAI"
    fi

    # Check for stale connections to camera port
    BLOCKING_PIDS=$(lsof -t -i :$CV60_PORT 2>/dev/null || true)
    if [ -n "$BLOCKING_PIDS" ]; then
        echo "  ⚠️  Found processes holding camera port $CV60_PORT:"
        for pid in $BLOCKING_PIDS; do
            PROCESS_NAME=$(ps -p "$pid" -o comm= 2>/dev/null || echo "unknown")
            echo "     PID $pid: $PROCESS_NAME"

            # Only kill if it's NOT the current backend (avoid killing ourselves)
            if [[ "$PROCESS_NAME" == "eBUSPlayer"* ]] || [[ "$PROCESS_NAME" == "python"* ]]; then
                # Check if it's a stale python process (not the one we're about to start)
                if ps -p "$pid" -o cmd= | grep -q "run_backend.py"; then
                    echo "     ⏭️  Skipping backend process (will be replaced)"
                    pkill -9 -f "run_backend.py" || true
                    sleep 2
                fi
            fi
        done
        echo "  ✅ Cleaned up blocking processes"
    else
        echo "  ✅ No blocking processes found"
    fi
}

# Function to verify camera can be accessed (quick test)
verify_camera_access() {
    echo "🔍 Verifying camera hardware access..."

    # Try a quick connection test using Python and eBUS SDK
    if python3 - <<'EOF'
import sys
import os

# Setup eBUS SDK paths
for path in ["/opt/jai/ebus_sdk/Ubuntu-22.04-x86_64/lib"]:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + ':/opt/jai/ebus_sdk/Ubuntu-22.04-x86_64/lib'

try:
    import eBUS as eb

    # Quick connection test
    result, device = eb.PvDevice.CreateAndConnect("192.168.1.21")

    if not device:
        print(f"❌ Camera connection failed: {result}")
        sys.exit(1)

    # Open stream to verify full access
    result_stream, stream = eb.PvStream.CreateAndOpen("192.168.1.21")

    if not stream:
        print(f"❌ Stream open failed: {result_stream}")
        device.Disconnect()
        eb.PvDevice.Free(device)
        sys.exit(1)

    # Cleanup
    stream.Close()
    eb.PvStream.Free(stream)
    device.Disconnect()
    eb.PvDevice.Free(device)

    print("✅ Camera hardware verified - device and stream accessible")
    sys.exit(0)

except ImportError as e:
    print(f"❌ eBUS SDK not available: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Camera verification failed: {e}")
    sys.exit(1)
EOF
    then
        echo "  ✅ Camera verification passed"
        return 0
    else
        echo "  ❌ Camera verification failed"
        return 1
    fi
}

# Main execution
main() {
    # Check network connectivity
    if ! check_camera_network; then
        echo ""
        echo "❌ FATAL: CV60 camera is not reachable on network"
        echo "   Please check:"
        echo "   1. Camera is powered on"
        echo "   2. Network cable is connected"
        echo "   3. Camera IP is $CV60_IP"
        echo "   4. Host is on same network (192.168.1.x)"
        echo ""
        exit 1
    fi

    echo ""

    # Kill any blocking processes
    kill_blocking_processes

    echo ""

    # Verify camera access
    if ! verify_camera_access; then
        echo ""
        echo "⚠️  WARNING: Camera verification failed"
        echo "   The backend will attempt to connect, but may experience issues"
        echo ""
        # Don't exit - let backend try anyway
    fi

    echo ""
    echo "================================================================================"
    echo "✅ CV60 CAMERA SETUP COMPLETE"
    echo "================================================================================"
    echo ""
}

# Only run if sourced or executed directly
if [ "${BASH_SOURCE[0]}" == "${0}" ] || [ -z "${BASH_SOURCE[0]}" ]; then
    main
fi
