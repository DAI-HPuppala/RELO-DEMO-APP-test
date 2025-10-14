"""OAK-D camera implementation using the existing OAKDCamera from core."""

from typing import Optional, Tuple, Any
import numpy as np
from ..core.oak_d_camera import OAKDCamera as CoreOAKDCamera, DEPTHAI_AVAILABLE
from .base import BaseCamera


class OAKDCamera(BaseCamera):
    """OAK-D PoE camera wrapper with 4K resolution and autofocus support.

    This is a thin wrapper around the core OAKDCamera implementation
    that provides a consistent interface with other cameras.

    Example:
        ```python
        from barcode_detector.cameras import OAKDCamera

        camera = OAKD Camera(resolution_4k=True, fps=30, ip_address="169.254.1.222")
        if camera.connect():
            ret, frame = camera.read()
            # Process frame
            camera.disconnect()
        ```
    """

    def __init__(self, resolution_4k: bool = True, fps: int = 30,
                 ip_address: Optional[str] = "169.254.1.222"):
        """Initialize OAK-D camera.

        Args:
            resolution_4k: Use 4K resolution (3840x2160). If False, uses 1080p
            fps: Target FPS (default: 30)
            ip_address: IP address for PoE devices (default: "169.254.1.222")
        """
        super().__init__()
        self._camera = CoreOAKDCamera(
            resolution_4k=resolution_4k,
            fps=fps,
            ip_address=ip_address
        )
        self.fps = fps
        self.width = 3840 if resolution_4k else 1920
        self.height = 2160 if resolution_4k else 1080

    def connect(self) -> bool:
        """Connect to the OAK-D camera.

        Returns:
            bool: True if connection successful, False otherwise
        """
        success = self._camera.initialize()
        self.is_opened = success
        return success

    def disconnect(self) -> None:
        """Disconnect from the camera."""
        self._camera.release()
        self.is_opened = False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        return self._camera.read()

    def get_property(self, prop: str) -> Any:
        """Get camera property.

        Args:
            prop: Property name (e.g., 'lens_position')

        Returns:
            Property value
        """
        if prop == 'lens_position':
            return self._camera.get_lens_position()
        return None

    def set_property(self, prop: str, value: Any) -> bool:
        """Set camera property.

        Args:
            prop: Property name
            value: Property value

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if prop == 'manual_focus':
                self._camera.set_manual_focus(int(value))
                return True
            elif prop == 'auto_focus':
                self._camera.set_auto_focus(str(value))
                return True
            elif prop == 'exposure':
                exposure_time, iso = value
                self._camera.set_exposure(int(exposure_time), int(iso))
                return True
            elif prop == 'auto_exposure':
                self._camera.set_auto_exposure(bool(value))
                return True
        except Exception:
            pass
        return False

    # Convenience methods
    def set_manual_focus(self, focus_value: int) -> None:
        """Set manual focus position.

        Args:
            focus_value: Focus value 0-255 (0=far, 255=near)
        """
        self._camera.set_manual_focus(focus_value)

    def set_auto_focus(self, mode: str = "CONTINUOUS_VIDEO") -> None:
        """Set autofocus mode.

        Args:
            mode: Autofocus mode - "CONTINUOUS_VIDEO", "CONTINUOUS_PICTURE", "AUTO", or "OFF"
        """
        self._camera.set_auto_focus(mode)

    def get_lens_position(self) -> Optional[int]:
        """Get the current lens position (focus value).

        Returns:
            Lens position (0-255) or None if not available
        """
        return self._camera.get_lens_position()


def create_oakd_camera(resolution_4k: bool = True, fps: int = 30,
                       ip_address: Optional[str] = "169.254.1.222") -> Optional[OAKDCamera]:
    """Factory function to create and connect an OAK-D camera.

    Args:
        resolution_4k: Use 4K resolution
        fps: Target FPS
        ip_address: IP address for PoE devices

    Returns:
        OAKDCamera instance if successful, None otherwise

    Example:
        ```python
        from barcode_detector.cameras import create_oakd_camera

        camera = create_oakd_camera()
        if camera:
            # Use camera
            camera.disconnect()
        ```
    """
    if not DEPTHAI_AVAILABLE:
        return None

    camera = OAKDCamera(resolution_4k=resolution_4k, fps=fps, ip_address=ip_address)
    if camera.connect():
        return camera
    return None


__all__ = ['OAKDCamera', 'create_oakd_camera', 'DEPTHAI_AVAILABLE']
