"""
pipeline/characterize/features.py — Feature Extraction for ML Classifier

Extracts an ~18-dimensional feature vector from a (light_curve, InversionResult)
pair. The ML model in model.py receives these features — NOT the raw light curve.

This keeps the classifier interpretable and physically grounded: every feature
either comes directly from the deterministic inversion or is a well-defined
statistical descriptor of the brightness time-series.

Feature inventory
-----------------
From InversionResult (physically derived):
  [0]  rotation_rate_hz           — dominant apparent frequency
  [1]  rotation_rate_uncertainty_hz
  [2]  amplitude                  — peak-to-trough / mean
  [3]  size_estimate_m            — midpoint estimate from amplitude heuristic
  [4]  snr_estimate               — estimated signal quality
  [5]  fourier_coeff_0            — DC / mean brightness (normalised)
  [6]  fourier_coeff_1            — fundamental amplitude
  [7]  fourier_coeff_2
  [8]  fourier_coeff_3            — first 4 AC coefficients
  [9]  shape_hint_encoded         — sphere=0, flat_plate=1, cylinder=2, tumbling=3, unknown=4
  [10] log_rotation_rate_hz       — log10(rotation_rate_hz + 1e-3) for log-scale separation

Statistical (from raw light curve):
  [11] mean
  [12] std
  [13] skewness
  [14] kurtosis
  [15] peak_count                 — number of local maxima
  [16] coefficient_of_variation   — std / mean
  [17] pct10_90_range             — 90th percentile − 10th percentile

Total: 18 features (indices 0–17)
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.signal import find_peaks

from pipeline.characterize.inversion import InversionResult

# Encoding for shape_hint categorical feature
_SHAPE_HINT_ENCODING = {
    "sphere":     0,
    "flat_plate": 1,
    "cylinder":   2,
    "tumbling":   3,
    "unknown":    4,
}

FEATURE_NAMES = [
    "rotation_rate_hz",
    "rotation_rate_uncertainty_hz",
    "amplitude",
    "size_estimate_m",
    "snr_estimate",
    "fourier_coeff_0",
    "fourier_coeff_1",
    "fourier_coeff_2",
    "fourier_coeff_3",
    "shape_hint_encoded",
    "log_rotation_rate_hz",
    "mean",
    "std",
    "skewness",
    "kurtosis",
    "peak_count",
    "coefficient_of_variation",
    "pct10_90_range",
]

N_FEATURES = len(FEATURE_NAMES)  # 17


def extract_features(lc: np.ndarray, inv: InversionResult) -> np.ndarray:
    """Extract a fixed-length feature vector from a light curve + inversion result.

    Parameters
    ----------
    lc : np.ndarray, shape (N,)
        Normalised brightness time-series (same array passed to invert()).
    inv : InversionResult
        Output of inversion.invert() for this light curve.

    Returns
    -------
    np.ndarray, shape (N_FEATURES,), dtype float64
        Feature vector. No NaN or Inf values.

    Raises
    ------
    ValueError
        If lc is empty or contains no finite values.
    """
    lc = np.asarray(lc, dtype=np.float64)
    if len(lc) == 0:
        raise ValueError("light curve is empty")
    if not np.any(np.isfinite(lc)):
        raise ValueError("light curve contains no finite values")

    # ── From InversionResult ──────────────────────────────────────────────────
    # Fourier coefficients — take first 4 AC coefficients (indices 1–4)
    # inv.fourier_coefficients[0] is DC; [1],[2],[3] are the first 3 AC terms
    coeffs = inv.fourier_coefficients
    fc0 = float(coeffs[0]) if len(coeffs) > 0 else 0.0
    fc1 = float(coeffs[1]) if len(coeffs) > 1 else 0.0
    fc2 = float(coeffs[2]) if len(coeffs) > 2 else 0.0
    fc3 = float(coeffs[3]) if len(coeffs) > 3 else 0.0

    shape_enc = float(_SHAPE_HINT_ENCODING.get(inv.shape_hint, 4))

    # ── Statistical features from raw light curve ─────────────────────────────
    lc_mean = float(np.mean(lc))
    lc_std  = float(np.std(lc))

    # Skewness and kurtosis — scipy handles edge cases (constant array → 0)
    lc_skew = float(stats.skew(lc))
    lc_kurt = float(stats.kurtosis(lc))

    # Peak count: number of local maxima above a prominence threshold
    prominence = max(0.02, lc_std * 0.1)
    peaks, _ = find_peaks(lc, prominence=prominence)
    peak_count = float(len(peaks))

    # Coefficient of variation (guard against zero mean)
    cv = float(lc_std / lc_mean) if lc_mean > 1e-9 else 0.0

    # 10th–90th percentile range
    pct10, pct90 = float(np.percentile(lc, 10)), float(np.percentile(lc, 90))
    pct_range = pct90 - pct10

    # Log-scale rotation rate — spreads out the low end where large/medium overlap
    # (small=2-5 Hz, medium=0.4-2 Hz, large=0.02-0.4 Hz on the linear scale)
    log_rot = float(np.log10(inv.rotation_rate_hz + 1e-3))

    # ── Assemble feature vector ───────────────────────────────────────────────
    features = np.array([
        inv.rotation_rate_hz,
        inv.rotation_rate_uncertainty_hz,
        inv.amplitude,
        inv.size_estimate_m,
        inv.snr_estimate,
        fc0, fc1, fc2, fc3,
        shape_enc,
        log_rot,
        lc_mean,
        lc_std,
        lc_skew,
        lc_kurt,
        peak_count,
        cv,
        pct_range,
    ], dtype=np.float64)

    # Safety: replace any NaN / Inf with 0.0 (defensive — should not occur)
    features = np.where(np.isfinite(features), features, 0.0)

    assert features.shape == (N_FEATURES,), (
        f"Feature vector has wrong shape: {features.shape}, expected ({N_FEATURES},)"
    )
    return features
