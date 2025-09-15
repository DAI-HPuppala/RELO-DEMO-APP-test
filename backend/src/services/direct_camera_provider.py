"""Direct camera frame provider for agents - bypasses WebRTC pipeline."""
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class DirectCameraProvider:
    """Provides frames directly from camera to agents, bypassing WebRTC."""

    @staticmethod
    async def get_frame_provider(session_id: str):
        """Create a frame provider that pulls directly from camera.

        This bypasses WebRTC processing for lower latency and
        ensures agents see raw camera frames.
        """

        async def provider() -> Optional[np.ndarray]:
            """Direct camera frame provider."""
            try:
                # Import here to avoid circular dependency
                from services.cv60_camera import CV60VideoTrack

                # Access the camera's frame queue directly
                # This is more efficient than going through WebRTC
                with CV60VideoTrack._contexts_lock:
                    ctx = CV60VideoTrack._contexts.get("192.168.1.21")
                    if not ctx:
                        logger.warning("No camera context available")
                        return None

                    # Get the latest captured frame from camera
                    # This bypasses WebRTC encoding/decoding
                    device = ctx.get("device")
                    stream = ctx.get("stream")

                    if not device or not stream:
                        logger.warning("Camera device or stream not available")
                        return None

                    # Retrieve buffer directly from eBUS stream
                    import eBUS as eb
                    result, buf, op = stream.RetrieveBuffer(100)  # 100ms timeout

                    if result.IsOK() and op and op.IsOK() and buf.GetPayloadType() == eb.PvPayloadTypeImage:
                        img = buf.GetImage()
                        w, h = img.GetWidth(), img.GetHeight()
                        raw = np.frombuffer(img.GetDataPointer(), dtype=np.uint8).copy()

                        # Get pixel type
                        pixel_type = img.GetPixelType()

                        # Process based on format (same as CV60VideoTrack)
                        if len(raw) == h * w * 3:
                            # Already RGB
                            frame_data = raw.reshape((h, w, 3))
                        elif len(raw) == h * w:
                            # Bayer or Mono - convert to BGR
                            import cv2
                            temp = raw.reshape((h, w))

                            if pixel_type == eb.PvPixelMono8:
                                frame_data = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)
                            else:
                                # Try BayerBG first (based on previous findings)
                                try:
                                    frame_data = cv2.cvtColor(temp, cv2.COLOR_BayerBG2BGR)
                                except:
                                    try:
                                        frame_data = cv2.cvtColor(temp, cv2.COLOR_BayerRG2BGR)
                                    except:
                                        # Fallback to grayscale
                                        frame_data = np.stack([temp, temp, temp], axis=2)

                            # Resize if needed
                            if frame_data.shape[:2] != (480, 640):
                                frame_data = cv2.resize(frame_data, (640, 480))
                        else:
                            logger.warning(f"Unknown format: {len(raw)} bytes")
                            stream.QueueBuffer(buf)
                            return None

                        # Return buffer to stream
                        stream.QueueBuffer(buf)

                        logger.debug(f"Direct camera frame captured: {frame_data.shape}")
                        return frame_data

                    # Return buffer if capture failed
                    if buf:
                        stream.QueueBuffer(buf)

                    return None

            except Exception as e:
                logger.error(f"Direct camera provider error: {e}")
                return None

        return provider


def get_direct_camera_provider(session_id: str):
    """Get a direct camera frame provider for agents.

    Benefits:
    - Lower latency (no WebRTC encoding/decoding)
    - Raw frames direct from camera
    - No color space conversions from WebRTC
    - Reduced CPU usage (bypasses video codec)

    Drawbacks:
    - Agents and frontend may see slightly different frames
    - No WebRTC buffering/smoothing
    - Direct hardware access may conflict with stream
    """
    import asyncio
    provider_instance = DirectCameraProvider()
    return asyncio.create_task(provider_instance.get_frame_provider(session_id))