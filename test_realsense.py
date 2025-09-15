#!/usr/bin/env python3
"""Test script to check for RealSense camera"""

import pyrealsense2 as rs
import numpy as np

def check_realsense_camera():
    """Check if RealSense camera is available"""
    print("Checking for RealSense cameras...")
    
    # Create a context object
    ctx = rs.context()
    
    # Query devices
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("❌ No RealSense devices found")
        return False
    
    print(f"✅ Found {len(devices)} RealSense device(s):")
    
    for i, device in enumerate(devices):
        print(f"\nDevice {i}:")
        print(f"  Name: {device.get_info(rs.camera_info.name)}")
        print(f"  Serial Number: {device.get_info(rs.camera_info.serial_number)}")
        print(f"  Firmware Version: {device.get_info(rs.camera_info.firmware_version)}")
        print(f"  USB Type: {device.get_info(rs.camera_info.usb_type_descriptor)}")
        
        # Check available sensors
        sensors = device.query_sensors()
        print(f"  Available sensors: {len(sensors)}")
        for j, sensor in enumerate(sensors):
            print(f"    Sensor {j}: {sensor.get_info(rs.camera_info.name)}")
    
    # Try to start a pipeline
    print("\nTrying to start RealSense pipeline...")
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Configure streams
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        # Start streaming
        pipeline.start(config)
        print("✅ Successfully started RealSense pipeline")
        
        # Get a few frames to verify it's working
        for i in range(5):
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame:
                print(f"  Frame {i+1}: Got color frame ({color_frame.get_width()}x{color_frame.get_height()})")
        
        # Stop streaming
        pipeline.stop()
        print("✅ RealSense camera is working properly")
        return True
        
    except Exception as e:
        print(f"❌ Failed to start pipeline: {e}")
        return False

if __name__ == "__main__":
    import sys
    success = check_realsense_camera()
    sys.exit(0 if success else 1)