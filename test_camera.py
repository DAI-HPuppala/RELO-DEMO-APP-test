#!/usr/bin/env python3
"""Test which camera indices work."""

import cv2

print("Testing camera indices...")
for i in range(10):
    try:
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"✓ Camera at index {i} works! Resolution: {width}x{height}")
                cap.release()
            else:
                print(f"✗ Camera at index {i} opened but can't read frames")
                cap.release()
        else:
            cap.release()
    except Exception as e:
        print(f"✗ Error with index {i}: {e}")

print("\nNow testing with direct device paths...")
for i in range(10):
    device = f"/dev/video{i}"
    try:
        cap = cv2.VideoCapture(device)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"✓ Camera at {device} works! Resolution: {width}x{height}")
                cap.release()
            else:
                print(f"✗ Camera at {device} opened but can't read frames")
                cap.release()
        else:
            cap.release()
    except Exception as e:
        print(f"✗ Error with {device}: {e}")