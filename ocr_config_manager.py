"""
OCR Configuration loader and manager for the enhanced OCR system.
"""

import json
import os
from typing import Dict, Any


class OCRConfig:
    """Configuration manager for OCR settings."""
    
    def __init__(self, config_file: str = "ocr_config.json"):
        """
        Initialize OCR configuration.
        
        Args:
            config_file: Path to the configuration file
        """
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading OCR config: {e}")
                return self.get_default_config()
        else:
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "ocr_settings": {
                "enhanced_ocr_enabled": True,
                "default_psm": 3,
                "default_language": "eng",
                "save_debug": True,
                "preprocessing": {
                    "enhance_contrast": True,
                    "contrast_factor": 1.5,
                    "enhance_brightness": False,
                    "brightness_factor": 1.2,
                    "enhance_sharpness": False,
                    "sharpness_factor": 1.3,
                    "grayscale": False,
                    "binary_threshold": False,
                    "threshold_value": 127,
                    "gaussian_blur": False,
                    "blur_radius": 1,
                    "median_filter": False,
                    "median_size": 3,
                    "morphological_ops": False,
                    "morph_kernel_size": 3,
                    "erosion": False,
                    "dilation": False,
                    "opening": False,
                    "closing": False
                },
                "benchmark_settings": {
                    "auto_benchmark": False,
                    "benchmark_interval_hours": 24,
                    "psm_modes_to_test": [3, 6, 7, 8, 11],
                    "languages_to_test": ["eng", "deu"],
                    "save_benchmark_results": True
                }
            },
            "directories": {
                "ocr_debug": "ocr_debug",
                "ocr_benchmarks": "ocr_benchmarks",
                "screenshots": "screenshots"
            },
            "advanced_settings": {
                "confidence_threshold": 30.0,
                "min_text_length": 5,
                "max_processing_time": 10.0
            }
        }
    
    def save_config(self):
        """Save current configuration to file."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving OCR config: {e}")
    
    def get_ocr_settings(self) -> Dict[str, Any]:
        """Get OCR settings."""
        return self.config.get("ocr_settings", {})
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self.config.get("ocr_settings", {}).get("preprocessing", {})
    
    def get_directories(self) -> Dict[str, str]:
        """Get directory settings."""
        return self.config.get("directories", {})
    
    def get_advanced_settings(self) -> Dict[str, Any]:
        """Get advanced settings."""
        return self.config.get("advanced_settings", {})
    
    def is_enhanced_ocr_enabled(self) -> bool:
        """Check if enhanced OCR is enabled."""
        return self.config.get("ocr_settings", {}).get("enhanced_ocr_enabled", True)
    
    def get_default_psm(self) -> int:
        """Get default PSM mode."""
        return self.config.get("ocr_settings", {}).get("default_psm", 3)
    
    def get_default_language(self) -> str:
        """Get default language."""
        return self.config.get("ocr_settings", {}).get("default_language", "eng")
    
    def should_save_debug(self) -> bool:
        """Check if debug information should be saved."""
        return self.config.get("ocr_settings", {}).get("save_debug", True)
    
    def update_setting(self, path: str, value: Any):
        """
        Update a specific setting using dot notation.
        
        Args:
            path: Dot-separated path to the setting (e.g., "ocr_settings.default_psm")
            value: New value for the setting
        """
        keys = path.split('.')
        current = self.config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
        self.save_config()


# Global config instance
ocr_config = OCRConfig()


def get_ocr_config() -> OCRConfig:
    """Get the global OCR configuration instance."""
    return ocr_config


if __name__ == "__main__":
    # Test configuration
    config = OCRConfig()
    print("OCR Configuration loaded:")
    print(f"Enhanced OCR enabled: {config.is_enhanced_ocr_enabled()}")
    print(f"Default PSM: {config.get_default_psm()}")
    print(f"Default language: {config.get_default_language()}")
    print(f"Save debug: {config.should_save_debug()}")
    print(f"Preprocessing config: {config.get_preprocessing_config()}")