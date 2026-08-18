"""
config.py — Delta-V configuration loader.

Loads environment variables from .env (if present) and exposes a typed
get_config() function with sensible defaults for all keys.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def get_config() -> dict:
    """Return the full application configuration as a dict.

    All keys have safe defaults so the app runs without any .env file.
    watsonx.ai keys default to None — callers must check before using Granite.
    """
    return {
        # TLE data source: "celestrak" | "spacetrack"
        "data_source": os.getenv("DATA_SOURCE", "celestrak").lower(),

        # Space-Track.org credentials (only used when data_source == "spacetrack")
        "spacetrack_user": os.getenv("SPACETRACK_USER", ""),
        "spacetrack_pass": os.getenv("SPACETRACK_PASS", ""),

        # IBM watsonx.ai (None means Granite is unavailable; use fallback text)
        "watsonx_api_key": os.getenv("WATSONX_API_KEY") or None,
        "watsonx_project_id": os.getenv("WATSONX_PROJECT_ID") or None,
        "watsonx_url": os.getenv(
            "WATSONX_URL", "https://us-south.ml.cloud.ibm.com"
        ),

        # File paths (relative to project root)
        "tle_cache_path": "data/cache/tle_cache.txt",
        "tle_cache_meta_path": "data/cache/tle_cache_meta.json",
        "tle_fallback_path": "data/tle_snapshot_fallback.csv",
        "model_path": "data/models/characterize_model.pkl",
        "validation_dir": "data/validation",

        # Risk map tuning defaults (overridable from UI)
        "risk_weight_density": 0.5,
        "risk_weight_detections": 0.5,
        "recency_decay_half_life_days": 7.0,

        # Prioritization defaults
        "reference_orbit_km": 400.0,       # chase vehicle starting altitude
        "plane_change_deg": 5.0,           # worst-case plane change assumption
    }


def granite_available(config: dict) -> bool:
    """Return True if watsonx.ai credentials are configured."""
    return bool(config.get("watsonx_api_key") and config.get("watsonx_project_id"))
