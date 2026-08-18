"""
pipeline/map/risk_map.py — Grid-Based Risk-Density Map

THIS IS NOT A BAYESIAN FILTER OR PARTICLE FILTER.

It is a grid-based accumulator that combines:
  1. Tracked-object density from TLE propagation (factual baseline)
  2. Characterized detections from Stage 1 (recency- and confidence-weighted)

The result is a composite risk density per altitude band, updated with each
new batch of detections. It provides a live, evolving regional risk picture —
not individual object tracking.

Algorithm
---------
For each altitude band b:

    detection_weight(b) = sum over detections in b of:
        confidence_i * exp(-age_days_i / decay_half_life)

    composite_risk_density(b) = w1 * density_norm(b) + w2 * detection_weight_norm(b)

where:
    density_norm       = density_per_km / max(density_per_km) over all bands
    detection_weight_norm = detection_weight / max(detection_weight) if > 0 else 0
    w1, w2             = configurable weights (default 0.5 / 0.5)
    decay_half_life    = configurable (default 7 days)

State
-----
The map holds an internal list of detection records. Calling update() appends
new detections; calling compute() re-evaluates the full composite score.
State can be reset with clear().

Detection record schema
-----------------------
Each detection is a dict with:
    altitude_band_km : str   — e.g. "400-600"  (must match density DataFrame)
    confidence       : float — [0, 1]
    timestamp        : float — Unix timestamp (seconds since epoch)
    size_class       : str   — "small" | "medium" | "large" (optional, for severity)
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from pipeline.map.density import ALTITUDE_BANDS

DECAY_HALF_LIFE_DAYS_DEFAULT = 7.0
W1_DEFAULT = 0.5   # weight for TLE density
W2_DEFAULT = 0.5   # weight for detection accumulator

_BAND_LABELS = [f"{lo}-{hi}" for lo, hi in ALTITUDE_BANDS]


class RiskDensityMap:
    """Grid-based risk-density accumulator across LEO altitude bands.

    This is a grid-based accumulator, not a Bayesian filter or particle filter.
    It does not track individual objects. It provides a live, evolving regional
    risk picture updated with each new detection batch.

    Usage
    -----
    map = RiskDensityMap()
    map.update(detections)          # add a batch of detection dicts
    df = map.compute(density_df)    # compute composite scores
    """

    def __init__(
        self,
        w1: float = W1_DEFAULT,
        w2: float = W2_DEFAULT,
        decay_half_life_days: float = DECAY_HALF_LIFE_DAYS_DEFAULT,
    ):
        self.w1 = w1
        self.w2 = w2
        self.decay_half_life_days = decay_half_life_days
        self._detections: List[Dict[str, Any]] = []

    # ── Mutation ──────────────────────────────────────────────────────────────

    def update(self, detections: List[Dict[str, Any]]) -> None:
        """Append new detections to the internal state.

        Parameters
        ----------
        detections : list of dict
            Each dict must have:
                altitude_band_km : str   — band label e.g. "400-600"
                confidence       : float — [0, 1]
                timestamp        : float — Unix timestamp (optional; defaults to now)
        """
        now = time.time()
        for d in detections:
            record = {
                "altitude_band_km": str(d["altitude_band_km"]),
                "confidence":       float(d.get("confidence", 1.0)),
                "timestamp":        float(d.get("timestamp", now)),
                "size_class":       str(d.get("size_class", "medium")),
            }
            self._detections.append(record)

    def clear(self) -> None:
        """Remove all stored detections."""
        self._detections = []

    # ── Computation ───────────────────────────────────────────────────────────

    def compute(self, density_df: pd.DataFrame) -> pd.DataFrame:
        """Compute composite risk density for all altitude bands.

        Parameters
        ----------
        density_df : pd.DataFrame
            Output of pipeline.map.density.compute_density().
            Must contain: altitude_band_km, density_per_km.

        Returns
        -------
        pd.DataFrame with columns:
            altitude_band_km, band_label, band_lo_km, band_hi_km,
            tracked_object_count, density_per_km,
            detection_count, detection_confidence_weighted,
            composite_risk_density, last_updated
        """
        now_ts = time.time()
        now_str = datetime.now(timezone.utc).isoformat()

        # ── Step 1: compute recency-weighted detection sum per band ───────────
        band_detection_weights = {label: 0.0 for label in _BAND_LABELS}
        band_detection_counts  = {label: 0   for label in _BAND_LABELS}

        for d in self._detections:
            band = d["altitude_band_km"]
            if band not in band_detection_weights:
                continue
            age_days = (now_ts - d["timestamp"]) / 86400.0
            recency = math.exp(-age_days / self.decay_half_life_days)
            band_detection_weights[band] += d["confidence"] * recency
            band_detection_counts[band]  += 1

        # ── Step 2: normalise both density and detection weight ───────────────
        density_values = density_df.set_index("altitude_band_km")["density_per_km"]
        max_density = max(density_values.max(), 1e-12)

        det_values = np.array([band_detection_weights[b] for b in _BAND_LABELS])
        max_det    = max(det_values.max(), 1e-12)

        # ── Step 3: compute composite score ──────────────────────────────────
        rows = []
        for row_d in density_df.itertuples(index=False):
            band = row_d.altitude_band_km
            density_norm = row_d.density_per_km / max_density
            det_weight   = band_detection_weights.get(band, 0.0)
            det_norm     = det_weight / max_det
            composite    = self.w1 * density_norm + self.w2 * det_norm

            rows.append({
                "altitude_band_km":              band,
                "band_label":                    row_d.band_label,
                "band_lo_km":                    row_d.band_lo_km,
                "band_hi_km":                    row_d.band_hi_km,
                "tracked_object_count":          row_d.object_count,
                "density_per_km":                row_d.density_per_km,
                "detection_count":               band_detection_counts.get(band, 0),
                "detection_confidence_weighted": det_weight,
                "composite_risk_density":        composite,
                "last_updated":                  now_str,
            })

        return pd.DataFrame(rows)

    @property
    def detection_count(self) -> int:
        """Total number of detections stored."""
        return len(self._detections)


# ── Missing import ────────────────────────────────────────────────────────────
import math  # noqa: E402 — placed after class to keep top-of-file clean
