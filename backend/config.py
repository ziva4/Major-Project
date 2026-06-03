import json
import os

class Config:
    """Configuration management for Violence Detection System"""

    DEFAULTS = {
        "server": {
            "host": "0.0.0.0",
            "port": 5000,
            "debug": False
        },
        "video": {
            "width": 640,
            "height": 480,
            "fps": 30,
            "camera_index": 0
        },
        "detection": {
            "confidence_threshold": 0.6,
            "smoothing_frames": 5,
            "alert_history_size": 100
        },
        "model": {
            "model_path": "models/violence_detector.h5",
            "scaler_path": "models/scaler.pkl",
            "training_samples": 1000,
            "training_epochs": 20,
            "batch_size": 32
        },
        "features": {
            "color_histogram_bins": 8,
            "feature_extraction_size": 64,
            "total_features": 128
        }
    }

    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.config = self.load_or_create()

    def load_or_create(self):
        """Load config from file or create with defaults"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                print(f"✓ Config loaded from {self.config_file}")
                return config
            except Exception as e:
                print(f"✗ Error loading config: {e}")
                print("Using default configuration")

        # Create and save default config
        self.save()
        return self.DEFAULTS

    def save(self):
        """Save current config to file"""
        os.makedirs(os.path.dirname(self.config_file) if os.path.dirname(self.config_file) else ".", exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"✓ Config saved to {self.config_file}")

    def get(self, *keys, default=None):
        """Get config value with dot notation: get('server', 'port')"""
        current = self.config
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
            else:
                return default
        return current if current is not None else default

    def set(self, *keys_and_value):
        """Set config value with dot notation: set('server', 'port', 5001)"""
        keys = keys_and_value[:-1]
        value = keys_and_value[-1]

        current = self.config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value
        self.save()

    def reset_to_defaults(self):
        """Reset all config to defaults"""
        self.config = self.DEFAULTS.copy()
        self.save()
        print("✓ Config reset to defaults")

    def __repr__(self):
        return json.dumps(self.config, indent=2)

if __name__ == "__main__":
    # Test config functionality
    config = Config()

    print("Current Configuration:")
    print(config)
    print(f"\nServer Port: {config.get('server', 'port')}")
    print(f"Confidence Threshold: {config.get('detection', 'confidence_threshold')}")
    print(f"Video Resolution: {config.get('video', 'width')}x{config.get('video', 'height')}")
