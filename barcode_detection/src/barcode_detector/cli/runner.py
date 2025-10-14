"""CLI runner for barcode detection system."""

from enum import Enum
from typing import Optional
import sys
import os


class DetectorMode(Enum):
    """Detection modes."""
    OAKD = "oakd"
    BARCODE = "barcode"
    FAST = "fast"
    AUTO = "auto"
    ENHANCED = "enhanced"
    TEST = "test"


def run_detector(mode: DetectorMode, camera_ip: Optional[str] = None, **kwargs) -> int:
    """Run the appropriate detector based on mode.

    Args:
        mode: Detection mode to run
        camera_ip: Camera IP address (for CV60 modes)
        **kwargs: Additional arguments passed to the detector

    Returns:
        Exit code (0 for success, non-zero for failure)

    Example:
        ```python
        from barcode_detector.cli import run_detector, DetectorMode

        # Run OAK-D detector
        run_detector(DetectorMode.OAKD)

        # Run CV60 barcode scanner
        run_detector(DetectorMode.BARCODE, camera_ip="192.168.1.21")
        ```
    """
    try:
        if mode == DetectorMode.OAKD:
            # Import and run OAK-D detector
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
            import robust_qr_detector
            robust_qr_detector.main()
            return 0

        elif mode == DetectorMode.BARCODE:
            # Import and run barcode detector
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
            import cv60_barcode_detector
            # Override IP if provided
            if camera_ip:
                sys.argv = ['cv60_barcode_detector.py', '--ip', camera_ip]
            cv60_barcode_detector.main()
            return 0

        elif mode == DetectorMode.FAST:
            # Import and run manual detector
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
            import cv60_qr_manual
            if camera_ip:
                sys.argv = ['cv60_qr_manual.py', '--ip', camera_ip]
            cv60_qr_manual.main()
            return 0

        elif mode == DetectorMode.AUTO:
            # Import and run auto detector
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
            import cv60_qr_fast
            if camera_ip:
                sys.argv = ['cv60_qr_fast.py', '--ip', camera_ip, '--interval', '5']
            cv60_qr_fast.main()
            return 0

        elif mode == DetectorMode.ENHANCED:
            # Import and run enhanced detector
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
            import cv60_qr_enhanced
            if camera_ip:
                sys.argv = ['cv60_qr_enhanced.py', '--ip', camera_ip]
            cv60_qr_enhanced.main()
            return 0

        elif mode == DetectorMode.TEST:
            # Import and run test mode
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
            import cv60_qr_with_fallback
            if camera_ip:
                sys.argv = ['cv60_qr_with_fallback.py', '--ip', camera_ip, '--fallback']
            cv60_qr_with_fallback.main()
            return 0

        else:
            print(f"Unknown mode: {mode}")
            return 1

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 0
    except Exception as e:
        print(f"Error running detector: {e}")
        import traceback
        traceback.print_exc()
        return 1


__all__ = ['run_detector', 'DetectorMode']
