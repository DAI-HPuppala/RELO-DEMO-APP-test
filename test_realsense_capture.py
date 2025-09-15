#!/usr/bin/env python3
"""Test RealSense camera capture and save frames."""
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
from datetime import datetime

def test_realsense_capture():
    """Test RealSense camera and capture frames."""
    print("=" * 60)
    print("RealSense Camera Test")
    print("=" * 60)
    
    # Create output directory
    output_dir = "/home/automation-dev/RELO-CLASSIFIER-specify-DEV/RELO-CLASSIFIER-DEV/backend/captured_frames"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Create context and check for devices
    ctx = rs.context()
    devices = ctx.devices
    
    print(f"\nFound {len(devices)} RealSense device(s)")
    
    if len(devices) == 0:
        print("ERROR: No RealSense devices found!")
        print("\nPossible issues:")
        print("1. Camera not connected properly")
        print("2. USB permissions issue")
        print("3. Driver not installed")
        return False
    
    # Print device info
    for i, device in enumerate(devices):
        print(f"\nDevice {i}:")
        try:
            serial = device.get_info(rs.camera_info.serial_number)
            name = device.get_info(rs.camera_info.name)
            firmware = device.get_info(rs.camera_info.firmware_version)
            print(f"  Name: {name}")
            print(f"  Serial: {serial}")
            print(f"  Firmware: {firmware}")
        except Exception as e:
            print(f"  Error getting device info: {e}")
    
    # Try to initialize pipeline
    print("\n" + "-" * 40)
    print("Initializing pipeline...")
    
    pipeline = rs.pipeline()
    config = rs.config()
    
    try:
        # Enable color stream
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start pipeline
        print("Starting pipeline...")
        profile = pipeline.start(config)
        
        # Get device and sensors
        device = profile.get_device()
        sensors = device.query_sensors()
        print(f"Found {len(sensors)} sensor(s)")
        
        for i, sensor in enumerate(sensors):
            sensor_name = sensor.get_info(rs.camera_info.name)
            print(f"  Sensor {i}: {sensor_name}")
            
            # Print supported options
            if sensor_name.lower().find('rgb') != -1 or sensor_name.lower().find('color') != -1:
                print("    Color sensor options:")
                options = [rs.option.exposure, rs.option.gain, rs.option.enable_auto_exposure]
                for opt in options:
                    if sensor.supports(opt):
                        value = sensor.get_option(opt)
                        opt_range = sensor.get_option_range(opt)
                        print(f"      {opt.name}: {value} (min:{opt_range.min}, max:{opt_range.max})")
        
        # Warm up camera
        print("\nWarming up camera (5 frames)...")
        for i in range(5):
            frames = pipeline.wait_for_frames()
            print(f"  Warmup frame {i+1}/5")
        
        # Capture and save frames
        print("\n" + "-" * 40)
        print("Capturing frames...")
        
        num_frames = 10
        captured_count = 0
        
        for i in range(num_frames):
            try:
                # Wait for frames
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    print(f"  Frame {i+1}: No color frame!")
                    continue
                
                # Convert to numpy array
                frame_data = np.asanyarray(color_frame.get_data())
                
                # Get frame info
                width = color_frame.get_width()
                height = color_frame.get_height()
                timestamp = color_frame.get_timestamp()
                frame_number = color_frame.get_frame_number()
                
                print(f"  Frame {i+1}: {width}x{height}, timestamp={timestamp}, frame_num={frame_number}")
                
                # Save frame
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"frame_{timestamp_str}_{i:03d}.jpg"
                filepath = os.path.join(output_dir, filename)
                
                success = cv2.imwrite(filepath, frame_data)
                if success:
                    print(f"    Saved: {filename}")
                    captured_count += 1
                else:
                    print(f"    ERROR: Failed to save {filename}")
                
                # Small delay between captures
                time.sleep(0.1)
                
            except Exception as e:
                print(f"  Frame {i+1}: Error - {e}")
        
        print(f"\nSuccessfully captured {captured_count}/{num_frames} frames")
        
        # Stop pipeline
        pipeline.stop()
        print("Pipeline stopped")
        return captured_count > 0
        
    except Exception as e:
        print(f"\nERROR: Failed to initialize camera: {e}")
        print("\nDetailed error information:")
        import traceback
        traceback.print_exc()
        
        if "No device connected" in str(e):
            print("\nThe camera is not being detected by the RealSense SDK.")
            print("Try:")
            print("1. Disconnect and reconnect the camera")
            print("2. Try a different USB port (preferably USB 3.0)")
            print("3. Check dmesg for USB errors: dmesg | tail -20")
        
        return False

if __name__ == "__main__":
    success = test_realsense_capture()
    print("\n" + "=" * 60)
    if success:
        print("TEST PASSED: Camera is working!")
    else:
        print("TEST FAILED: Camera issues detected")
    print("=" * 60)