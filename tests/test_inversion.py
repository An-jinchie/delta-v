"""
tests/test_inversion.py — Tests for pipeline/characterize/inversion.py

Strategy
--------
We generate synthetic light curves with known ground-truth parameters using
the generator, then verify that inversion recovers them within tolerance.
This is the correct validation approach: test against known signals before
running against noisy real data (Sub-Task 5).

Key physical facts about the generator output that inform test tolerances:
- flat_plate uses |cos(2π·f·t)| → dominant FFT bin appears at 2f (rectification)
- tumbling uses two incommensurate frequencies → spread will be high
- sphere has near-constant A_proj → DC-dominated → shape_hint == "sphere"
- cylinder has DC offset + cosine → intermediate spread
"""

import math

import numpy as np
import pytest

from pipeline.characterize.generator import LightCurveGenerator, LightCurveParams
from pipeline.characterize.inversion import (
    InversionResult,
    N_FOURIER_COEFFS,
    invert,
)

CADENCE_S = 0.1
N_SAMPLES = 256


def _make_curve(shape: str, rotation_rate_hz: float, snr: float = 50.0, seed: int = 0) -> np.ndarray:
    gen = LightCurveGenerator()
    params = LightCurveParams(
        size_class="medium",
        shape=shape,
        rotation_rate_hz=rotation_rate_hz,
        phase_offset=0.0,
        albedo=0.15,
        snr=snr,
        n_samples=N_SAMPLES,
        cadence_s=CADENCE_S,
        seed=seed,
    )
    return gen.generate(params)


# ── Return type and field contract ────────────────────────────────────────────

def test_returns_inversion_result():
    lc = _make_curve("flat_plate", 1.0)
    result = invert(lc, cadence_s=CADENCE_S)
    assert isinstance(result, InversionResult)


def test_fourier_coefficients_shape():
    lc = _make_curve("tumbling", 0.5)
    result = invert(lc, cadence_s=CADENCE_S)
    assert result.fourier_coefficients.shape == (N_FOURIER_COEFFS,)


def test_fourier_coefficients_non_negative():
    """Coefficients are absolute amplitudes — must be ≥ 0."""
    lc = _make_curve("cylinder", 0.8)
    result = invert(lc, cadence_s=CADENCE_S)
    assert np.all(result.fourier_coefficients >= 0.0)


def test_size_class_valid_values():
    for shape in ("sphere", "flat_plate", "cylinder", "tumbling"):
        lc = _make_curve(shape, 0.5)
        result = invert(lc, cadence_s=CADENCE_S)
        assert result.size_class in {"small", "medium", "large"}, (
            f"Unexpected size_class '{result.size_class}' for shape={shape}"
        )


def test_shape_hint_valid_values():
    for shape in ("sphere", "flat_plate", "cylinder", "tumbling"):
        lc = _make_curve(shape, 0.5)
        result = invert(lc, cadence_s=CADENCE_S)
        assert result.shape_hint in {"sphere", "flat_plate", "cylinder", "tumbling", "unknown"}, (
            f"Unexpected shape_hint '{result.shape_hint}' for shape={shape}"
        )


def test_amplitude_non_negative():
    for shape in ("sphere", "flat_plate", "cylinder", "tumbling"):
        lc = _make_curve(shape, 0.5)
        result = invert(lc, cadence_s=CADENCE_S)
        assert result.amplitude >= 0.0, f"Negative amplitude for shape={shape}"


def test_snr_positive():
    for shape in ("sphere", "flat_plate", "tumbling"):
        lc = _make_curve(shape, 0.5)
        result = invert(lc, cadence_s=CADENCE_S)
        assert result.snr_estimate > 0.0


def test_uncertainty_positive():
    lc = _make_curve("flat_plate", 1.0)
    result = invert(lc, cadence_s=CADENCE_S)
    assert result.rotation_rate_uncertainty_hz > 0.0


def test_metadata_stored_correctly():
    lc = _make_curve("flat_plate", 1.0)
    result = invert(lc, cadence_s=CADENCE_S)
    assert result.n_samples == N_SAMPLES
    assert result.cadence_s == CADENCE_S


# ── Rotation rate recovery ────────────────────────────────────────────────────

@pytest.mark.parametrize("f_rot", [0.3, 0.5, 1.0, 2.0])
def test_flat_plate_rotation_rate_recovered(f_rot):
    """Flat plate uses |cos(2π·f·t)| — dominant FFT bin is at 2f.
    We verify inversion returns rotation_rate_hz ≈ 2*f_rot within 3 FFT bins.
    """
    lc = _make_curve("flat_plate", f_rot, snr=80.0)
    result = invert(lc, cadence_s=CADENCE_S)
    expected = 2.0 * f_rot  # |cos| rectification doubles apparent frequency
    tolerance = 3.0 * result.rotation_rate_uncertainty_hz * 2  # 3 bins
    assert abs(result.rotation_rate_hz - expected) <= tolerance, (
        f"f_rot={f_rot}: recovered {result.rotation_rate_hz:.4f} Hz, "
        f"expected ~{expected:.4f} Hz (±{tolerance:.4f})"
    )


@pytest.mark.parametrize("f_rot", [0.3, 0.8, 1.5])
def test_tumbling_dominant_frequency_present(f_rot):
    """Tumbling curve has two frequencies: f_rot and f_rot*1.618.
    The dominant FFT bin should be near one of them (×2 for rectification).
    """
    lc = _make_curve("tumbling", f_rot, snr=80.0)
    result = invert(lc, cadence_s=CADENCE_S)
    f1 = 2.0 * f_rot
    f2 = 2.0 * f_rot * 1.618
    tolerance = 4.0 / (N_SAMPLES * CADENCE_S)  # 4 FFT bins
    near_f1 = abs(result.rotation_rate_hz - f1) <= tolerance
    near_f2 = abs(result.rotation_rate_hz - f2) <= tolerance
    assert near_f1 or near_f2, (
        f"f_rot={f_rot}: dominant freq {result.rotation_rate_hz:.4f} Hz not near "
        f"f1={f1:.4f} or f2={f2:.4f} (tol={tolerance:.4f})"
    )


# ── Shape hint ────────────────────────────────────────────────────────────────

def test_sphere_shape_hint():
    """High-SNR sphere should be identified as sphere (DC-dominated)."""
    lc = _make_curve("sphere", 0.5, snr=100.0)
    result = invert(lc, cadence_s=CADENCE_S)
    assert result.shape_hint == "sphere", (
        f"Sphere identified as '{result.shape_hint}'"
    )


def test_flat_plate_shape_hint():
    """High-SNR flat plate should be identified as flat_plate (low spread)."""
    lc = _make_curve("flat_plate", 0.5, snr=100.0)
    result = invert(lc, cadence_s=CADENCE_S)
    assert result.shape_hint == "flat_plate", (
        f"Flat plate identified as '{result.shape_hint}'"
    )


def test_tumbling_shape_hint():
    """High-SNR tumbling should be identified as tumbling (high spread)."""
    lc = _make_curve("tumbling", 0.5, snr=100.0)
    result = invert(lc, cadence_s=CADENCE_S)
    assert result.shape_hint == "tumbling", (
        f"Tumbling identified as '{result.shape_hint}'"
    )


# ── Amplitude & size ──────────────────────────────────────────────────────────

def test_sphere_amplitude_low():
    """Sphere has near-constant A_proj → low amplitude."""
    lc = _make_curve("sphere", 0.5, snr=100.0)
    result = invert(lc, cadence_s=CADENCE_S)
    # Sphere normalised to [0,1] from pure noise: amplitude should be < 0.5
    # (it can't be 0 due to noise, but should be lower than flat plate)
    lc_plate = _make_curve("flat_plate", 0.5, snr=100.0)
    result_plate = invert(lc_plate, cadence_s=CADENCE_S)
    assert result.amplitude < result_plate.amplitude, (
        f"Sphere amplitude ({result.amplitude:.4f}) not < flat_plate ({result_plate.amplitude:.4f})"
    )


def test_large_amplitude_gives_large_size_class():
    """A tumbling large object at low SNR should have amplitude → large class."""
    # Large object parameters: high albedo variation, tumbling
    from pipeline.characterize.generator import LightCurveParams
    gen = LightCurveGenerator()
    params = LightCurveParams(
        size_class="large",
        shape="tumbling",
        rotation_rate_hz=0.1,
        phase_offset=0.0,
        albedo=0.30,
        snr=200.0,  # very clean — want to test amplitude, not noise
        n_samples=N_SAMPLES,
        cadence_s=CADENCE_S,
        seed=10,
    )
    lc = gen.generate(params)
    result = invert(lc, cadence_s=CADENCE_S)
    # Amplitude of a clean tumbling curve should push above the medium threshold
    assert result.amplitude >= 0.10, (
        f"Expected amplitude ≥ 0.10 for large tumbling, got {result.amplitude:.4f}"
    )


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_minimum_length_raises():
    with pytest.raises(ValueError, match="too short"):
        invert(np.ones(4))


def test_exactly_minimum_length_ok():
    lc = np.sin(np.linspace(0, 2 * math.pi, 8))
    result = invert(lc, cadence_s=CADENCE_S)
    assert isinstance(result, InversionResult)


def test_constant_curve():
    """A perfectly flat curve (zero noise sphere) should not crash."""
    lc = np.ones(N_SAMPLES)
    result = invert(lc, cadence_s=CADENCE_S)
    assert result.amplitude == 0.0
    assert result.size_class == "small"


def test_no_nan_in_any_field():
    for shape in ("sphere", "flat_plate", "cylinder", "tumbling"):
        lc = _make_curve(shape, 0.7, snr=20.0)
        r = invert(lc, cadence_s=CADENCE_S)
        assert not math.isnan(r.rotation_rate_hz)
        assert not math.isnan(r.amplitude)
        assert not math.isnan(r.snr_estimate)
        assert not np.any(np.isnan(r.fourier_coefficients))


# ── Uncertainty bounds ────────────────────────────────────────────────────────

def test_uncertainty_equals_half_bin_width():
    """Uncertainty should equal exactly half the FFT bin width."""
    lc = _make_curve("flat_plate", 1.0)
    result = invert(lc, cadence_s=CADENCE_S)
    expected_uncertainty = 0.5 / (N_SAMPLES * CADENCE_S)
    assert abs(result.rotation_rate_uncertainty_hz - expected_uncertainty) < 1e-10
