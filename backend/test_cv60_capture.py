#!/usr/bin/env python3
"""
Test script for CV60 camera capture using eBUS SDK.
Captures 3 frames and saves them to capture_frames directory.
"""
import os
import sys
import site
import time
import numpy as np
import cv2
from datetime import datetime

# Setup eBUS SDK paths
for p in site.getsitepackages():
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# Add additional common paths for Debian packages
for path in [
    "/usr/lib/python3/dist-packages",
    "/usr/local/lib/python3.10/dist-packages",
    "/usr/lib/python3.10/dist-packages",
    "/usr/local/lib/python3.10/site-packages",
    "/opt/jai/ebus_sdk/Ubuntu-22.04-x86_64/lib"
]:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

# Add LD_LIBRARY_PATH for native libraries
os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + ':/opt/jai/ebus_sdk/Ubuntu-22.04-x86_64/lib'

try:
    import eBUS as eb
    print("✅ Successfully imported eBUS module")
except ImportError as e:
    print(f"❌ ERROR importing eBUS: {e}")
    print("Please ensure eBUS SDK is installed")
    sys.exit(1)

# Configuration
CV60_IP = "192.168.1.21"
BUFFER_COUNT = 16
CAPTURE_COUNT = 3
OUTPUT_DIR = "capture_frames"


def test_cv60_capture():
    """Test CV60 camera capture with frame buffer."""
    print(f"\n📷 Testing CV60 Camera Capture")
    print(f"Camera IP: {CV60_IP}")
    print(f"Frames to capture: {CAPTURE_COUNT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 50)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize camera connection
    print(f"\n🔌 Connecting to CV60 at {CV60_IP}...")

    try:
        # Create and connect to device
        print(f"   Attempting connection...")
        result, device = eb.PvDevice.CreateAndConnect(CV60_IP)
        if not device:
            print(f"❌ Failed to connect to device at {CV60_IP}")
            print(f"   Result: {result}")
            if result.GetCode() == eb.PvResultCode.AlreadyConnected:
                print(f"   ⚠️  Camera appears to be already in use by another process")
                print(f"   Try stopping the backend service first")
            elif result.GetCode() == eb.PvResultCode.AccessDenied:
                print(f"   ⚠️  Access denied - check camera permissions")
            else:
                print(f"   Error code: {result.GetCode()}, Description: {result.GetDescription()}")
            return False

        print(f"✅ Connected to device")

        # Get device info
        device_info = device.GetDeviceInfo()
        if device_info:
            print(f"   Model: {device_info.GetModelName()}")
            print(f"   Vendor: {device_info.GetVendorName()}")
            print(f"   Serial: {device_info.GetSerialNumber()}")

        # Open stream
        print(f"\n📡 Opening stream...")
        result, stream = eb.PvStream.CreateAndOpen(CV60_IP)
        if not stream:
            print(f"❌ Failed to open stream")
            device.Disconnect()
            return False

        print(f"✅ Stream opened")

        # Configure packet size for GigE
        print(f"⚙️  Configuring GigE settings...")
        try:
            # Get device type
            device_gev = device.GetAsGEV()
            if device_gev:
                print(f"   Device is GigE Vision")
                # Negotiate packet size
                device_gev.NegotiatePacketSize()

                # Set stream destination
                system = eb.PvSystem()
                system.Find()

                for i in range(system.GetInterfaceCount()):
                    iface = system.GetInterface(i)
                    try:
                        # Get the first IP address
                        ip_count = iface.GetIPAddressCount()
                        if ip_count > 0:
                            ip = iface.GetIPAddress(0)
                            if ip.startswith(('192.168', '169.254', '10.')):
                                device_gev.SetStreamDestination(ip, stream.GetLocalPort())
                                print(f"✅ Configured for interface {ip}")
                                break
                    except Exception as e:
                        print(f"   Interface {i} error: {e}")
                        continue
        except Exception as e:
            print(f"   GigE configuration warning: {e}")
            print(f"   Continuing anyway...")

        # Allocate buffers
        print(f"\n💾 Allocating {BUFFER_COUNT} buffers...")
        size = device.GetPayloadSize()
        print(f"   Payload size: {size} bytes")

        buf_count = min(stream.GetQueuedBufferMaximum(), BUFFER_COUNT)
        buffers = []

        for i in range(buf_count):
            buf = eb.PvBuffer()
            buf.Alloc(size)
            buffers.append(buf)
            stream.QueueBuffer(buf)

        print(f"✅ Allocated {buf_count} buffers")

        # Start acquisition
        print(f"\n🎬 Starting acquisition...")
        device.StreamEnable()
        device.GetParameters().Get("AcquisitionStart").Execute()

        # Capture frames
        captured_frames = []
        print(f"\n📸 Capturing {CAPTURE_COUNT} frames...")

        for frame_num in range(CAPTURE_COUNT):
            print(f"\n   Frame {frame_num + 1}/{CAPTURE_COUNT}:")

            # Retrieve buffer
            result, buf, op = stream.RetrieveBuffer(5000)  # 5 second timeout

            if result.IsOK() and op and op.IsOK():
                if buf.GetPayloadType() == eb.PvPayloadTypeImage:
                    # Get image from buffer
                    img = buf.GetImage()
                    w, h = img.GetWidth(), img.GetHeight()
                    pixel_type = img.GetPixelType()

                    print(f"   - Resolution: {w}x{h}")
                    print(f"   - Pixel type: {pixel_type}")

                    # Convert to numpy array
                    raw = np.frombuffer(img.GetDataPointer(), dtype=np.uint8).copy()

                    # Convert based on pixel format
                    if pixel_type == eb.PvPixelMono8:
                        frame = cv2.cvtColor(raw.reshape((h, w)), cv2.COLOR_GRAY2BGR)
                    else:
                        # Assuming BayerRG8 format
                        temp = raw.reshape((h, w))
                        frame = cv2.cvtColor(temp, cv2.COLOR_BayerRG2BGR)

                    # Resize if needed
                    if frame.shape[:2] != (480, 640):
                        frame = cv2.resize(frame, (640, 480))

                    captured_frames.append(frame)
                    print(f"   ✅ Captured frame with shape {frame.shape}")
                else:
                    print(f"   ⚠️  Buffer is not an image")
            else:
                print(f"   ❌ Failed to retrieve buffer")
                print(f"      Result: {result}")
                if op:
                    print(f"      Operation: {op}")

            # Return buffer to stream
            if buf:
                stream.QueueBuffer(buf)

            # Small delay between captures
            if frame_num < CAPTURE_COUNT - 1:
                time.sleep(0.5)

        # Stop acquisition
        print(f"\n⏹️  Stopping acquisition...")
        device.GetParameters().Get("AcquisitionStop").Execute()
        device.StreamDisable()

        # Save captured frames
        if captured_frames:
            print(f"\n💾 Saving {len(captured_frames)} frames to {OUTPUT_DIR}/...")

            for i, frame in enumerate(captured_frames):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"cv60_capture_{i+1}_{timestamp}.jpg"
                filepath = os.path.join(OUTPUT_DIR, filename)

                cv2.imwrite(filepath, frame)
                print(f"   ✅ Saved: {filename}")
        else:
            print(f"\n⚠️  No frames captured")

        # Cleanup
        print(f"\n🧹 Cleaning up...")
        stream.Close()
        device.Disconnect()

        print(f"\n✅ Test completed successfully!")
        print(f"   Frames captured: {len(captured_frames)}")
        print(f"   Saved to: {os.path.abspath(OUTPUT_DIR)}")

        return len(captured_frames) > 0

    except Exception as e:
        print(f"\n❌ Error during capture: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ebus_discovery():
    """Test eBUS device discovery."""
    print(f"\n🔍 Discovering eBUS devices...")

    try:
        system = eb.PvSystem()
        system.Find()

        device_count = 0

        for i in range(system.GetInterfaceCount()):
            iface = system.GetInterface(i)
            print(f"\nInterface {i}: {iface.GetDisplayID()}")

            for j in range(iface.GetDeviceCount()):
                device_info = iface.GetDeviceInfo(j)
                device_count += 1

                print(f"  Device {j}:")
                print(f"    - Connection ID: {device_info.GetConnectionID()}")
                print(f"    - Display ID: {device_info.GetDisplayID()}")
                print(f"    - Model: {device_info.GetModelName()}")
                print(f"    - Vendor: {device_info.GetVendorName()}")
                print(f"    - Serial: {device_info.GetSerialNumber()}")
                print(f"    - IP: {device_info.GetIPAddress()}")

        print(f"\n📊 Total devices found: {device_count}")
        return device_count > 0

    except Exception as e:
        print(f"❌ Error during discovery: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("CV60 Camera Test Script")
    print("=" * 60)

    # First, try device discovery
    print("\n[Step 1] Device Discovery")
    devices_found = test_ebus_discovery()

    if not devices_found:
        print("\n⚠️  No devices found. Please check:")
        print("   1. CV60 camera is powered on")
        print("   2. Network cable is connected")
        print("   3. Camera IP is configured correctly")
        print("   4. Your network interface is in the same subnet")

    # Then try capture
    print("\n[Step 2] Frame Capture Test")
    success = test_cv60_capture()

    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Please check the output above.")

    print("\n" + "=" * 60)