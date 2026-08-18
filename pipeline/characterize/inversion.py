"""
pipeline/characterize/inversion.py — Light-Curve Inversion (Primary Characterization Math)

This is the PRIMARY characterization method. Given a raw brightness time-series,
it estimates physical properties using deterministic math:

    1. Fourier decomposition  → rotation rate + harmonic structure
    2. Amplitude estimation   → size class heuristic
    3. Coefficient analysis   → shape hint

Every output traces to a documented formula. No model weights involved.
The ML classifier in model.py is SECONDARY — it refines these estimates,
it does not replace them.

Derivations
-----------

Rotation rate
    The light curve is periodic at the rotation frequency f (or 2f for symmetric
    shapes like flat plates where |cos| rectifies the signal). We compute the
    real FFT, skip the DC bin, and take the bin with maximum power:

        rotation_rate_hz = argmax(|FFT[1:N/2]|) * (1 / (N * cadence_s))

    Uncertainty is half the FFT bin width:

        rotation_rate_uncertainty_hz = 0.5 / (N * cadence_s)

Amplitude
    Peak-to-trough ratio of the mean-normalised curve:

        amplitude = (max(lc) - min(lc)) / mean(lc)    [dimensionless]

    Used as a heuristic size proxy (larger debris tumbles more violently):
        amplitude < 0.1   → small  (~1–3 cm)
        0.1 ≤ amp < 0.4   → medium (~3–7 cm)
        amp ≥ 0.4         → large  (~7–10 cm)

    This is a heuristic — not a ground-truth size measurement.
    Documented as such in every output.

Shape hint (from Fourier coefficient structure)
    We examine the power spectrum of the first N_COEFF Fourier coefficients:
        - Sphere:      very low AC power (almost all energy in DC bin)
        - Flat plate:  single dominant harmonic, low higher harmonics
        - Tumbling:    two comparable fundamental frequencies, high spread
        - Cylinder:    harmonic content between flat plate and tumbling

SNR estimate
    Signal power = variance of curve after mean subtraction
    Noise floor  = mean power of the top 20% of FFT frequencies (high-freq noise)
    SNR          = signal_power / noise_floor  (dimensionless ratio)

Reference: Kaasalainen & Torppa (2001). Icarus 153(1), 24–36.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────────
N_FOURIER_COEFFS = 8     # number of Fourier coefficients to retain
NOISE_FLOOR_FRAC = 0.20  # fraction of high-frequency bins used for SNR noise floor

# Amplitude thresholds for size heuristic (documented — not ground truth)
AMP_SMALL_MAX  = 0.10
AMP_MEDIUM_MAX = 0.40

# Size midpoint estimates (metres) used as continuous size_estimate_m
_SIZE_MIDPOINT_M = {"small": 0.02, "medium": 0.05, "large": 0.085}

# Shape classification thresholds — tuned against generator output
#
# dom_frac   = dominant_AC_bin_power / total_AC_power
#              (how concentrated the signal is in one frequency)
# second_ratio = second_strongest_AC_peak / dominant_AC_peak
#              (how prominent the second frequency is relative to the first)
#
# sphere    : dom_frac < 0.04   → no frequency stands out (noise only)
# flat_plate: dom_frac > 0.10   AND second_ratio < 0.35  → one dominant freq
# tumbling  : dom_frac > 0.08   AND second_ratio > 0.35  → two competing freqs
# cylinder  : everything else (intermediate)
_SPHERE_DOM_FRAC_MAX     = 0.04   # dominant AC bin fraction below → sphere
_FLAT_PLATE_DOM_FRAC_MIN = 0.10   # dominant AC bin fraction above → could be flat/tumb
_TUMBLING_SECOND_RATIO   = 0.35   # second/first peak ratio above → tumbling


@dataclass
class InversionResult:
    """Physical estimates derived from a light curve by deterministic math.

    All fields are computed from formulas documented in inversion.py.
    The ML classifier in model.py receives these as features.

    Attributes
    ----------
    rotation_rate_hz : float
        Dominant apparent frequency from FFT peak (Hz).
        For |cos| shapes (flat plate) this is 2× the physical rotation rate.
    rotation_rate_uncertainty_hz : float
        Half-width of one FFT frequency bin — the fundamental resolution limit.
    amplitude : float
        (max - min) / mean of the normalised light curve.  Dimensionless.
    size_class : str
        Heuristic size class: "small" | "medium" | "large".
        Derived from amplitude thresholds — documented as a heuristic.
    size_estimate_m : float
        Midpoint size estimate in metres corresponding to size_class.
        Treat as an order-of-magnitude estimate only.
    shape_hint : str
        Structural shape hint: "sphere" | "flat_plate" | "cylinder" | "tumbling" | "unknown".
        Derived from Fourier coefficient power distribution.
    fourier_coefficients : np.ndarray, shape (N_FOURIER_COEFFS,)
        First N real Fourier coefficients (normalised amplitudes).
        DC component (index 0) is the mean brightness.
    snr_estimate : float
        Estimated signal-to-noise ratio.  >10 is reliable; <5 treat with caution.
    n_samples : int
        Length of the input light curve.
    cadence_s : float
        Seconds between samples (used to convert FFT bins to Hz).
    """
    rotation_rate_hz: float           # dominant apparent FFT frequency (= 2× physical for |cos| shapes)
    rotation_rate_uncertainty_hz: float
    amplitude: float
    size_class: str
    size_estimate_m: float
    shape_hint: str
    fourier_coefficients: np.ndarray
    snr_estimate: float
    n_samples: int
    cadence_s: float


def invert(light_curve: np.ndarray, cadence_s: float = 0.1) -> InversionResult:
    """Estimate physical properties of a debris object from its light curve.

    Parameters
    ----------
    light_curve : np.ndarray, shape (N,)
        Brightness time-series, any scale (need not be in [0, 1]).
        Must contain at least 8 samples.
    cadence_s : float
        Time between samples in seconds.  Default 0.1 s.

    Returns
    -------
    InversionResult
        All physical estimates with documented derivations.

    Raises
    ------
    ValueError
        If light_curve has fewer than 8 samples.
    """
    lc = np.asarray(light_curve, dtype=np.float64)
    n = len(lc)
    if n < 8:
        raise ValueError(f"Light curve too short: {n} samples (minimum 8)")

    # ── 1. FFT ────────────────────────────────────────────────────────────────
    fft_full   = np.fft.rfft(lc)
    fft_power  = np.abs(fft_full)
    # Frequency axis (Hz)
    freqs      = np.fft.rfftfreq(n, d=cadence_s)

    # ── 2. Rotation rate from dominant non-DC frequency ───────────────────────
    # Skip bin 0 (DC / mean brightness) — start from bin 1
    ac_power   = fft_power[1:]
    ac_freqs   = freqs[1:]
    dom_idx    = int(np.argmax(ac_power))          # index into ac_power
    rotation_rate_hz = float(ac_freqs[dom_idx])

    # Frequency resolution = 1 / total_duration
    freq_resolution = 1.0 / (n * cadence_s)
    rotation_rate_uncertainty_hz = 0.5 * freq_resolution

    # ── 3. Fourier coefficients (first N_FOURIER_COEFFS real amplitudes) ──────
    # Normalised so that coefficient[0] = mean brightness
    n_coeffs = min(N_FOURIER_COEFFS, len(fft_full))
    fourier_coefficients = (np.abs(fft_full[:n_coeffs]) / n).astype(np.float64)

    # ── 4. Amplitude (peak-to-trough ratio) ───────────────────────────────────
    lc_mean = float(np.mean(lc))
    if lc_mean == 0.0:
        amplitude = 0.0
    else:
        amplitude = float((lc.max() - lc.min()) / lc_mean)

    # ── 5. Size class heuristic ───────────────────────────────────────────────
    # Documented as heuristic — amplitude is a proxy for size, not exact.
    if amplitude < AMP_SMALL_MAX:
        size_class = "small"
    elif amplitude < AMP_MEDIUM_MAX:
        size_class = "medium"
    else:
        size_class = "large"
    size_estimate_m = _SIZE_MIDPOINT_M[size_class]

    # ── 6. Shape hint from Fourier power distribution ─────────────────────────
    # Work entirely in AC bins (skip DC / bin 0).
    total_ac = float(np.sum(ac_power) + 1e-12)

    # Fraction of AC power in the dominant bin
    dom_frac = float(ac_power[dom_idx]) / total_ac

    # Second-strongest AC peak (zero out ±2 bins around dominant, take max)
    ac_no_dom = ac_power.copy()
    lo = max(0, dom_idx - 2)
    hi = min(len(ac_no_dom), dom_idx + 3)
    ac_no_dom[lo:hi] = 0.0
    second_peak = float(ac_no_dom.max())
    second_ratio = second_peak / (float(ac_power[dom_idx]) + 1e-12)

    if dom_frac < _SPHERE_DOM_FRAC_MAX:
        # No frequency stands out — noise-dominated → sphere
        shape_hint = "sphere"
    elif dom_frac >= _FLAT_PLATE_DOM_FRAC_MIN and second_ratio < _TUMBLING_SECOND_RATIO:
        # One clear dominant frequency, weak second peak → flat_plate
        shape_hint = "flat_plate"
    elif dom_frac >= _FLAT_PLATE_DOM_FRAC_MIN and second_ratio >= _TUMBLING_SECOND_RATIO:
        # Strong dominant AND strong second peak → two-axis tumble
        shape_hint = "tumbling"
    else:
        # Intermediate: dominant exists but weaker; second peak present → cylinder
        shape_hint = "cylinder"

    # ── 7. SNR estimate ───────────────────────────────────────────────────────
    # Signal power: variance of the AC component
    signal_power = float(np.var(lc - lc_mean))
    # Noise floor: mean power in top NOISE_FLOOR_FRAC of frequency bins
    n_noise_bins = max(1, int(NOISE_FLOOR_FRAC * len(ac_power)))
    sorted_power = np.sort(ac_power)
    noise_floor  = float(np.mean(sorted_power[-n_noise_bins:]) ** 2) / (n ** 2) + 1e-12
    snr_estimate = float(signal_power / noise_floor)
    # Cap at a sensible display range
    snr_estimate = min(snr_estimate, 1e6)

    return InversionResult(
        rotation_rate_hz=rotation_rate_hz,
        rotation_rate_uncertainty_hz=rotation_rate_uncertainty_hz,
        amplitude=amplitude,
        size_class=size_class,
        size_estimate_m=size_estimate_m,
        shape_hint=shape_hint,
        fourier_coefficients=fourier_coefficients,
        snr_estimate=snr_estimate,
        n_samples=n,
        cadence_s=cadence_s,
    )
