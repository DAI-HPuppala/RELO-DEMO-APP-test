"""Camera Settings Manager - Persistent storage for OAK-D camera configurations"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Path to persistent settings file
SETTINGS_DIR = Path(__file__).parent.parent.parent.parent / "roi_finder" / "configs"
PERSISTENT_SETTINGS_FILE = SETTINGS_DIR / "persistent_camera_settings.json"
DEFAULT_SETTINGS_FILE = SETTINGS_DIR / "default_camera_settings.json"


class CameraSettingsManager:
    """Manages persistent camera settings for OAK-D Pro"""

    def __init__(self):
        """Initialize settings manager"""
        self.current_settings = self._load_persistent_settings()
        logger.info("CameraSettingsManager initialized")

    def _load_persistent_settings(self) -> Dict[str, Any]:
        """Load persistent settings from file, or use defaults if not found"""
        try:
            if PERSISTENT_SETTINGS_FILE.exists():
                with open(PERSISTENT_SETTINGS_FILE, 'r') as f:
                    settings = json.load(f)
                logger.info(f" Loaded persistent camera settings from {PERSISTENT_SETTINGS_FILE}")
                return settings
            elif DEFAULT_SETTINGS_FILE.exists():
                with open(DEFAULT_SETTINGS_FILE, 'r') as f:
                    settings = json.load(f)
                logger.info(f" Loaded default camera settings from {DEFAULT_SETTINGS_FILE}")
                return settings
            else:
                logger.warning("No settings file found, using factory defaults")
                return self._get_factory_defaults()
        except Exception as e:
            logger.error(f"Error loading settings: {e}, using factory defaults")
            return self._get_factory_defaults()

    def _get_factory_defaults(self) -> Dict[str, Any]:
        """Get factory default settings"""
        return {
            "name": "Factory Defaults",
            "description": "OAK-D Pro factory settings",
            "created": datetime.now().isoformat(),
            "camera_height_cm": None,
            "ground_coverage": {"width_cm": None, "height_cm": None},
            "roi": {"enabled": False, "x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0},
            "focus": {"mode": "continuous_video", "manual_position": 128, "range_min": 0, "range_max": 255},
            "exposure": {"mode": "auto", "auto_limit_us": 33000, "compensation": 0, "lock": False, "manual_time_us": 16000, "manual_iso": 800},
            "white_balance": {"mode": "auto", "lock": False, "manual_kelvin": 5500},
            "image_quality": {"sharpness": 1, "luma_denoise": 1, "chroma_denoise": 4, "saturation": 0, "contrast": 0, "brightness": 0},
            "advanced": {"anti_banding": "60hz", "effect_mode": "off", "scene_mode": "unsupported"}
        }

    def save_settings(self, settings: Dict[str, Any], make_persistent: bool = True) -> bool:
        """
        Save settings to memory and optionally to persistent storage

        Args:
            settings: Settings dictionary to save
            make_persistent: If True, save to disk permanently

        Returns:
            True if successful, False otherwise
        """
        try:
            # Update current settings
            self.current_settings.update(settings)
            self.current_settings["updated"] = datetime.now().isoformat()

            if make_persistent:
                # Ensure directory exists
                SETTINGS_DIR.mkdir(parents=True, exist_ok=True)

                # Save to persistent file
                with open(PERSISTENT_SETTINGS_FILE, 'w') as f:
                    json.dump(self.current_settings, f, indent=2)

                logger.info(f" Camera settings saved permanently to {PERSISTENT_SETTINGS_FILE}")
                logger.info(f" Settings will persist across reboots")
            else:
                logger.info("📝 Camera settings updated (session only, not persistent)")

            return True
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            return False

    def get_settings(self) -> Dict[str, Any]:
        """Get current settings"""
        return self.current_settings.copy()

    def get_roi_settings(self) -> Dict[str, Any]:
        """Get ROI-specific settings"""
        return self.current_settings.get("roi", {})

    def get_focus_settings(self) -> Dict[str, Any]:
        """Get focus-specific settings"""
        return self.current_settings.get("focus", {})

    def get_exposure_settings(self) -> Dict[str, Any]:
        """Get exposure-specific settings"""
        return self.current_settings.get("exposure", {})

    def get_white_balance_settings(self) -> Dict[str, Any]:
        """Get white balance-specific settings"""
        return self.current_settings.get("white_balance", {})

    def get_image_quality_settings(self) -> Dict[str, Any]:
        """Get image quality-specific settings"""
        return self.current_settings.get("image_quality", {})

    def save_height_measurement(self, height_cm: float, ground_coverage: Dict[str, float], make_persistent: bool = True) -> bool:
        """Save measured camera height"""
        settings_update = {
            "camera_height_cm": height_cm,
            "ground_coverage": ground_coverage,
            "height_measured_at": datetime.now().isoformat()
        }
        return self.save_settings(settings_update, make_persistent=make_persistent)

    def save_roi(self, x_min: float, y_min: float, x_max: float, y_max: float, enabled: bool = True, make_persistent: bool = True) -> bool:
        """Save ROI configuration"""
        settings_update = {
            "roi": {
                "enabled": enabled,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max
            }
        }
        return self.save_settings(settings_update, make_persistent=make_persistent)

    def load_preset(self, preset_name: str) -> Optional[Dict[str, Any]]:
        """Load a named preset configuration"""
        try:
            preset_file = SETTINGS_DIR / f"{preset_name}.json"
            if preset_file.exists():
                with open(preset_file, 'r') as f:
                    preset = json.load(f)
                logger.info(f" Loaded preset: {preset_name}")
                return preset
            else:
                logger.warning(f"Preset not found: {preset_name}")
                return None
        except Exception as e:
            logger.error(f"Error loading preset {preset_name}: {e}")
            return None

    def save_as_preset(self, preset_name: str, description: str = "") -> bool:
        """Save current settings as a named preset"""
        try:
            preset = self.current_settings.copy()
            preset["name"] = preset_name
            preset["description"] = description
            preset["created"] = datetime.now().isoformat()

            preset_file = SETTINGS_DIR / f"{preset_name}.json"
            with open(preset_file, 'w') as f:
                json.dump(preset, f, indent=2)

            logger.info(f" Saved preset: {preset_name} to {preset_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving preset {preset_name}: {e}")
            return False

    def list_presets(self) -> list:
        """List available presets"""
        try:
            presets = []
            for file in SETTINGS_DIR.glob("*.json"):
                if file.stem not in ["default_camera_settings", "persistent_camera_settings"]:
                    presets.append(file.stem)
            return presets
        except Exception as e:
            logger.error(f"Error listing presets: {e}")
            return []


# Global singleton instance
_settings_manager = None

def get_camera_settings_manager() -> CameraSettingsManager:
    """Get or create the global settings manager instance"""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = CameraSettingsManager()
    return _settings_manager
