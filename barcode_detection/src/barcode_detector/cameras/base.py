"""Base camera interface."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any
import numpy as np


class BaseCamera(ABC):
    """Abstract base class for camera implementations."""

    def __init__(self):
        self.is_opened = False
        self.width = 0
        self.height = 0
        self.fps = 0

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the camera.

        Returns:
            bool: True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the camera."""
        pass

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        pass

    @abstractmethod
    def get_property(self, prop: str) -> Any:
        """Get camera property.

        Args:
            prop: Property name

        Returns:
            Property value
        """
        pass

    @abstractmethod
    def set_property(self, prop: str, value: Any) -> bool:
        """Set camera property.

        Args:
            prop: Property name
            value: Property value

        Returns:
            bool: True if successful, False otherwise
        """
        pass

    def isOpened(self) -> bool:
        """Check if camera is opened.

        Returns:
            bool: True if opened, False otherwise
        """
        return self.is_opened

    def get(self, prop: int) -> float:
        """Get camera property (OpenCV-style interface).

        Args:
            prop: Property ID (cv2.CAP_PROP_*)

        Returns:
            float: Property value
        """
        import cv2
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        elif prop == cv2.CAP_PROP_FPS:
            return float(self.fps)
        return 0.0

    def release(self) -> None:
        """Release camera resources."""
        self.disconnect()
