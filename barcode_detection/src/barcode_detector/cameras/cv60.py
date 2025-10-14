"""CV60 camera implementation using the existing CV60Camera from core."""

from typing import Optional, Tuple, Any
import numpy as np
from ..core.cv60_camera import create_cv60_camera as create_core_cv60_camera
from .base import BaseCamera


class CV60Camera(BaseCamera):
    """CV60 industrial camera wrapper.

    This is a thin wrapper around the core create_cv60_camera factory
    that provides a consistent interface with other cameras.

    Example:
        ```python
        from barcode_detector.cameras import CV60Camera

        camera = CV60Camera(ip_address="192.168.1.21")
        if camera.connect():
            ret, frame = camera.read()
            # Process frame
            camera.disconnect()
        ```
    """

    def __init__(self, ip_address: str = "192.168.1.21"):
        """Initialize CV60 camera.

        Args:
            ip_address: IP address of the CV60 camera
        """
        super().__init__()
        self.ip_address = ip_address
        self._camera = None

    def connect(self) -> bool:
        """Connect to the CV60 camera.

        Returns:
            bool: True if connection successful, False otherwise
        """
        self._camera = create_core_cv60_camera(ip_address=self.ip_address)
        if self._camera and self._camera.isOpened():
            self.is_opened = True
            # Get camera properties
            import cv2
            self.width = int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self._camera.get(cv2.CAP_PROP_FPS)) or 30
            return True
        return False

    def disconnect(self) -> None:
        """Disconnect from the camera."""
        if self._camera:
            self._camera.release()
            self._camera = None
        self.is_opened = False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        if not self._camera:
            return (False, None)
        return self._camera.read()

    def get_property(self, prop: str) -> Any:
        """Get camera property.

        Args:
            prop: Property name

        Returns:
            Property value
        """
        if not self._camera:
            return None
        # CV60 uses OpenCV interface
        import cv2
        prop_map = {
            'width': cv2.CAP_PROP_FRAME_WIDTH,
            'height': cv2.CAP_PROP_FRAME_HEIGHT,
            'fps': cv2.CAP_PROP_FPS,
        }
        if prop in prop_map:
            return self._camera.get(prop_map[prop])
        return None

    def set_property(self, prop: str, value: Any) -> bool:
        """Set camera property.

        Args:
            prop: Property name
            value: Property value

        Returns:
            bool: True if successful, False otherwise
        """
        if not self._camera:
            return False
        # CV60 uses OpenCV interface
        import cv2
        prop_map = {
            'width': cv2.CAP_PROP_FRAME_WIDTH,
            'height': cv2.CAP_PROP_FRAME_HEIGHT,
            'fps': cv2.CAP_PROP_FPS,
        }
        if prop in prop_map:
            return self._camera.set(prop_map[prop], value)
        return False


def create_cv60_camera(ip_address: str = "192.168.1.21") -> Optional[CV60Camera]:
    """Factory function to create and connect a CV60 camera.

    Args:
        ip_address: IP address of the CV60 camera

    Returns:
        CV60Camera instance if successful, None otherwise

    Example:
        ```python
        from barcode_detector.cameras import create_cv60_camera

        camera = create_cv60_camera()
        if camera:
            # Use camera
            camera.disconnect()
        ```
    """
    camera = CV60Camera(ip_address=ip_address)
    if camera.connect():
        return camera
    return None


__all__ = ['CV60Camera', 'create_cv60_camera']
