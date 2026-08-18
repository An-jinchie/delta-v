"""
pipeline/characterize/generator.py — Synthetic Light-Curve Generator

Generates physics-based brightness time-series for space debris objects.
This is the sole source of training data for the ML classifier (Sub-Task 4)
and provides a known-ground-truth baseline for validating inversion math
against controlled signals (Sub-Task 3) before testing on real noisy data.

Physical model
--------------
Brightness at time t:

    B(t) = albedo * A_proj(t) * (L + S(t))

where:
    A_proj(t)  = projected area as a function of rotation phase (shape-dependent)
    L          = Lambertian reflection coefficient (0.7, fixed)
    S(t)       = specular highlight term (narrow Gaussian spike at peak phase)

Shapes
------
    sphere      : constant projected area → flat curve + noise only
    flat_plate  : A_proj = |cos(2π * f * t)| (one dominant frequency)
    cylinder    : A_proj = 0.5 * |cos(2π * f * t)| + 0.5  (frequency + DC offset)
    tumbling    : sum of two sinusoids at different frequencies (two-axis tumble)

Noise
-----
    Gaussian noise: B_noisy = B + N(0, σ),  σ = max(B) / SNR
    Output normalised to [0, 1].

Reference: Kaasalainen & Torppa (2001). Icarus, 153(1), 24–36.
           For hackathon scope the sinusoidal approximation is honest and sufficient.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
LAMBERTIAN_COEFF = 0.7      # fraction of reflected light that is Lambertian
SPECULAR_COEFF   = 0.3      # fraction that is specular (at-peak highlight)
SPECULAR_WIDTH   = 0.05     # width of specular Gaussian as fraction of period
N_SAMPLES_DEFAULT = 256     # time-series length (number of brightness samples)
CADENCE_S_DEFAULT = 0.1     # seconds between samples → 25.6 s total arc

# Size-class parameter ranges  (rotation_hz, albedo)
# Rotation rate ranges are intentionally non-overlapping to give the classifier
# a learnable discriminant. Small debris spins fast; large debris spins slowly.
#
# Nyquist constraint: cadence=0.1s → Nyquist=5Hz. For |cos| shapes the apparent
# FFT frequency is 2×f_phys. So f_phys must stay below 2.5 Hz to avoid aliasing.
# Ranges chosen so apparent frequencies (2×f) are separated across classes:
#   small:  f_phys 1.0–2.4 Hz  → apparent 2.0–4.8 Hz
#   medium: f_phys 0.3–1.0 Hz  → apparent 0.6–2.0 Hz
#   large:  f_phys 0.02–0.3 Hz → apparent 0.04–0.6 Hz
#
# Reference: Schildknecht et al. (2005) — LEO debris rotation rates 0.05–10 Hz.
_SIZE_PARAMS = {
    "small":  {"rotation_rate_range": (1.0, 2.4),   "albedo_range": (0.05, 0.15)},
    "medium": {"rotation_rate_range": (0.3, 1.0),   "albedo_range": (0.08, 0.25)},
    "large":  {"rotation_rate_range": (0.02, 0.3),  "albedo_range": (0.10, 0.35)},
}

_SHAPES = ("sphere", "flat_plate", "cylinder", "tumbling")
_SIZES  = ("small", "medium", "large")


@dataclass
class LightCurveParams:
    """All parameters needed to generate one synthetic light curve."""
    size_class: str          # "small" | "medium" | "large"
    shape: str               # "sphere" | "flat_plate" | "cylinder" | "tumbling"
    rotation_rate_hz: float  # primary rotation frequency in Hz
    phase_offset: float      # initial phase in radians
    albedo: float            # surface reflectivity in [0, 1]
    snr: float               # signal-to-noise ratio (higher = cleaner curve)
    n_samples: int = N_SAMPLES_DEFAULT
    cadence_s: float = CADENCE_S_DEFAULT
    seed: Optional[int] = None


class LightCurveGenerator:
    """Generates synthetic brightness time-series for debris objects.

    Usage
    -----
    >>> gen = LightCurveGenerator()
    >>> params = LightCurveParams(
    ...     size_class="medium", shape="flat_plate",
    ...     rotation_rate_hz=0.5, phase_offset=0.0, albedo=0.15, snr=20.0, seed=42)
    >>> curve = gen.generate(params)
    >>> assert curve.shape == (256,)
    >>> assert 0.0 <= curve.min() and curve.max() <= 1.0
    """

    def generate(self, params: LightCurveParams) -> np.ndarray:
        """Generate a normalised brightness time-series.

        Parameters
        ----------
        params : LightCurveParams

        Returns
        -------
        np.ndarray, shape (n_samples,), values in [0, 1]
        """
        rng = np.random.default_rng(params.seed)
        t = np.arange(params.n_samples) * params.cadence_s

        # ── Projected area A_proj(t) ─────────────────────────────────────────
        f = params.rotation_rate_hz
        phi = params.phase_offset

        if params.shape == "sphere":
            A_proj = np.ones(params.n_samples)

        elif params.shape == "flat_plate":
            # Single dominant frequency, full modulation
            A_proj = np.abs(np.cos(2 * math.pi * f * t + phi))

        elif params.shape == "cylinder":
            # Rotation modulates area but a baseline is always visible
            A_proj = 0.5 * np.abs(np.cos(2 * math.pi * f * t + phi)) + 0.5

        elif params.shape == "tumbling":
            # Two-axis tumble: two incommensurate frequencies
            f2 = f * 1.618  # golden-ratio offset avoids exact harmonics
            A_proj = (
                0.6 * np.abs(np.cos(2 * math.pi * f * t + phi))
                + 0.4 * np.abs(np.cos(2 * math.pi * f2 * t + phi * 0.7))
            )
            A_proj = A_proj / A_proj.max()  # normalise so peak = 1.0

        else:
            raise ValueError(f"Unknown shape: '{params.shape}'")

        # ── Specular highlight ────────────────────────────────────────────────
        # Narrow Gaussian spike at the phase where the surface faces the observer
        if params.shape != "sphere":
            peak_phase = 2 * math.pi * f * t + phi
            # Highlight fires when peak_phase ≈ 0 (mod 2π)
            wrapped = (peak_phase % (2 * math.pi)) / (2 * math.pi)  # in [0, 1)
            sigma_wrap = SPECULAR_WIDTH
            # Evaluate Gaussian centred at 0 (and 1 for wrap-around)
            gaussian = np.exp(-0.5 * ((wrapped) / sigma_wrap) ** 2) + \
                       np.exp(-0.5 * ((wrapped - 1.0) / sigma_wrap) ** 2)
            specular = SPECULAR_COEFF * gaussian
        else:
            specular = np.zeros(params.n_samples)

        # ── Combined brightness ───────────────────────────────────────────────
        B = params.albedo * A_proj * (LAMBERTIAN_COEFF + specular)

        # ── Gaussian noise ────────────────────────────────────────────────────
        if params.snr > 0:
            sigma_noise = B.max() / params.snr if B.max() > 0 else 0.0
            B = B + rng.normal(0.0, sigma_noise, size=B.shape)

        # ── Normalise to [0, 1] ───────────────────────────────────────────────
        b_min, b_max = B.min(), B.max()
        if b_max > b_min:
            B = (B - b_min) / (b_max - b_min)
        else:
            B = np.zeros_like(B)

        return B.astype(np.float64)


def generate_dataset(
    n_samples: int = 2000,
    output_path: str = "data/synthetic/light_curves.csv",
    seed: int = 0,
    n_time_points: int = N_SAMPLES_DEFAULT,
    cadence_s: float = CADENCE_S_DEFAULT,
    snr_range: tuple[float, float] = (8.0, 40.0),
) -> None:
    """Generate a labeled dataset of synthetic light curves and write to CSV.

    CSV format
    ----------
    - Header row: t_0, t_1, ..., t_{n_time_points-1}, size_class, shape, rotation_rate_hz
    - Each data row: brightness values + labels
    - First line comment: # SYNTHETIC DATA — see README

    Parameters
    ----------
    n_samples : int
        Number of light curves to generate.
    output_path : str
        Destination CSV file path.
    seed : int
        Master RNG seed for reproducibility.
    n_time_points : int
        Length of each time-series.
    cadence_s : float
        Seconds between samples.
    snr_range : tuple[float, float]
        (min_snr, max_snr) — sampled uniformly per curve.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rng = np.random.default_rng(seed)
    gen = LightCurveGenerator()

    rows = []
    for i in range(n_samples):
        # Randomly sample size and shape
        size_class = rng.choice(_SIZES)
        shape = rng.choice(_SHAPES)
        size_p = _SIZE_PARAMS[size_class]

        # Sample continuous parameters
        rot_lo, rot_hi = size_p["rotation_rate_range"]
        rotation_rate_hz = float(rng.uniform(rot_lo, rot_hi))
        alb_lo, alb_hi = size_p["albedo_range"]
        albedo = float(rng.uniform(alb_lo, alb_hi))
        phase_offset = float(rng.uniform(0.0, 2 * math.pi))
        snr = float(rng.uniform(snr_range[0], snr_range[1]))

        params = LightCurveParams(
            size_class=size_class,
            shape=shape,
            rotation_rate_hz=rotation_rate_hz,
            phase_offset=phase_offset,
            albedo=albedo,
            snr=snr,
            n_samples=n_time_points,
            cadence_s=cadence_s,
            seed=int(rng.integers(0, 2**31)),
        )
        curve = gen.generate(params)

        row = list(curve) + [size_class, shape, rotation_rate_hz]
        rows.append(row)

    # Column headers
    time_cols = [f"t_{i}" for i in range(n_time_points)]
    header = time_cols + ["size_class", "shape", "rotation_rate_hz"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        f.write("# SYNTHETIC DATA — physics-based light-curve generator. See README.\n")
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"[generator] Wrote {n_samples} synthetic light curves → {output_path}")
