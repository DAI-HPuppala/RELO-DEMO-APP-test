"""Lightweight barcode detection module.

This module provides fast, CPU-efficient barcode presence detection
that can be used as a first-stage filter before heavy decoding.
"""

from .barcode_presence_detector import BarcodePresenceDetector, BarcodeRegion

__all__ = ['BarcodePresenceDetector', 'BarcodeRegion']
