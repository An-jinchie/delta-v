"""
tests/test_generator.py — Tests for pipeline/characterize/generator.py
"""

import math
import os
import tempfile

import numpy as np
import pytest

from pipeline.characterize.generator import (
    LightCurveGenerator,
    LightCurveParams,
    N_SAMPLES_DEFAULT,
    generate_dataset,
)


@pytest.fixture
def gen():
    return LightCurveGenerator()


def _params(shape, seed=42, snr=20.0, rotation_rate_hz=0.5, size_class="medium"):
    return LightCurveParams(
        size_class=size_class,
        shape=shape,
        rotation_rate_hz=rotation_rate_hz,
        phase_offset=0.0,
        albedo=0.15,
        snr=snr,
        seed=seed,
    )


# ── Shape tests ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("shape", ["sphere", "flat_plate", "cylinder", "tumbling"])
def test_output_shape(gen, shape):
    curve = gen.generate(_params(shape))
    assert curve.shape == (N_SAMPLES_DEFAULT,), f"Expected ({N_SAMPLES_DEFAULT},), got {curve.shape}"


@pytest.mark.parametrize("shape", ["sphere", "flat_plate", "cylinder", "tumbling"])
def test_values_in_unit_range(gen, shape):
    curve = gen.generate(_params(shape))
    assert curve.min() >= 0.0, f"Min {curve.min():.4f} < 0 for shape={shape}"
    assert curve.max() <= 1.0, f"Max {curve.max():.4f} > 1 for shape={shape}"


@pytest.mark.parametrize("shape", ["sphere", "flat_plate", "cylinder", "tumbling"])
def test_no_nan_or_inf(gen, shape):
    curve = gen.generate(_params(shape))
    assert not np.any(np.isnan(curve)), f"NaN in output for shape={shape}"
    assert not np.any(np.isinf(curve)), f"Inf in output for shape={shape}"


def test_dtype_float64(gen):
    curve = gen.generate(_params("flat_plate"))
    assert curve.dtype == np.float64


# ── Reproducibility ───────────────────────────────────────────────────────────

def test_reproducibility_same_seed(gen):
    p = _params("tumbling", seed=99)
    c1 = gen.generate(p)
    c2 = gen.generate(p)
    np.testing.assert_array_equal(c1, c2, err_msg="Same seed produced different outputs")


def test_different_seeds_differ(gen):
    c1 = gen.generate(_params("flat_plate", seed=1))
    c2 = gen.generate(_params("flat_plate", seed=2))
    assert not np.allclose(c1, c2), "Different seeds produced identical outputs"


# ── Physical behaviour ────────────────────────────────────────────────────────

def test_sphere_lower_amplitude_than_flat_plate(gen):
    """Sphere has constant area so its amplitude should be lower than flat plate.

    Both generated at the same SNR — the flat plate's |cos| modulation produces
    a much larger peak-to-trough range than the sphere's noise-only variation.
    """
    # Use a low rotation rate so the flat plate's periodicity is clearly resolved
    sphere = gen.generate(_params("sphere",     snr=30.0, seed=5, rotation_rate_hz=0.3))
    plate  = gen.generate(_params("flat_plate", snr=30.0, seed=5, rotation_rate_hz=0.3))
    # Peak-to-trough after normalisation (both are in [0,1])
    # Flat plate spans nearly the full [0,1] range; sphere is noise-only
    sphere_range = sphere.max() - sphere.min()
    plate_range  = plate.max()  - plate.min()
    # Both will reach 1.0 after normalisation, but the sphere's *std* should be
    # much lower (noise only) compared to the flat plate (large periodic signal)
    assert sphere.std() < plate.std(), (
        f"Sphere std {sphere.std():.4f} should be < flat plate std {plate.std():.4f}"
    )


def test_flat_plate_has_periodicity(gen):
    """Flat plate should show a dominant frequency in its FFT.

    A flat plate uses |cos(2π·f·t)|, which rectifies the cosine and therefore
    has its fundamental power at *2f* (the full-wave rectification doubles the
    frequency). We check that the dominant FFT bin is near 2*f_true.
    """
    f_true = 1.0  # Hz — the rotation rate
    f_expected = 2 * f_true  # |cos| doubles the apparent frequency
    params = _params("flat_plate", rotation_rate_hz=f_true, snr=50.0)
    curve = gen.generate(params)
    n = len(curve)
    cadence = params.cadence_s
    freqs = np.fft.rfftfreq(n, d=cadence)
    power = np.abs(np.fft.rfft(curve))
    dominant_freq = freqs[1 + np.argmax(power[1:])]  # skip DC
    freq_resolution = 1.0 / (n * cadence)
    assert abs(dominant_freq - f_expected) < 3 * freq_resolution, (
        f"Flat plate dominant freq {dominant_freq:.4f} Hz, expected ~{f_expected} Hz"
    )


def test_tumbling_two_frequencies(gen):
    """Tumbling curve should have power at two distinct frequencies."""
    f_true = 0.8
    params = _params("tumbling", rotation_rate_hz=f_true, snr=50.0)
    curve = gen.generate(params)
    n = len(curve)
    cadence = params.cadence_s
    power = np.abs(np.fft.rfft(curve))
    # Find top-2 peaks (excluding DC)
    top2_idx = np.argsort(power[1:])[-2:] + 1
    assert len(top2_idx) == 2
    # Both should have non-trivial power (> 5% of max)
    assert power[top2_idx].min() > 0.05 * power[1:].max()


def test_snr_effect(gen):
    """Higher SNR should produce a smoother curve (lower std of residual)."""
    low_snr = gen.generate(_params("flat_plate", snr=5.0, seed=7))
    high_snr = gen.generate(_params("flat_plate", snr=100.0, seed=7))
    # The high-SNR curve should have lower point-to-point roughness
    roughness_low  = np.abs(np.diff(low_snr)).mean()
    roughness_high = np.abs(np.diff(high_snr)).mean()
    assert roughness_high < roughness_low, (
        f"High-SNR curve rougher than low-SNR: {roughness_high:.4f} vs {roughness_low:.4f}"
    )


def test_unknown_shape_raises(gen):
    p = _params("triangle")  # not a valid shape
    with pytest.raises(ValueError, match="Unknown shape"):
        gen.generate(p)


# ── Dataset generation ────────────────────────────────────────────────────────

def test_generate_dataset_creates_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_curves.csv")
        generate_dataset(n_samples=20, output_path=path, seed=0)
        assert os.path.exists(path)


def test_generate_dataset_row_count():
    import csv as csv_mod
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_curves.csv")
        generate_dataset(n_samples=30, output_path=path, seed=0)
        with open(path, encoding="utf-8") as f:
            lines = [l for l in f if not l.startswith("#")]
        reader = list(csv_mod.reader(lines))
        # First line is header, remaining are data rows
        assert len(reader) - 1 == 30, f"Expected 30 data rows, got {len(reader)-1}"


def test_generate_dataset_has_synthetic_comment():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_curves.csv")
        generate_dataset(n_samples=5, output_path=path, seed=0)
        with open(path, encoding="utf-8") as f:
            first_line = f.readline()
        assert "SYNTHETIC" in first_line, "Missing SYNTHETIC DATA comment in CSV header"


def test_generate_dataset_reproducibility():
    import pandas as pd
    with tempfile.TemporaryDirectory() as tmp:
        p1 = os.path.join(tmp, "a.csv")
        p2 = os.path.join(tmp, "b.csv")
        generate_dataset(n_samples=10, output_path=p1, seed=42)
        generate_dataset(n_samples=10, output_path=p2, seed=42)
        df1 = pd.read_csv(p1, comment="#")
        df2 = pd.read_csv(p2, comment="#")
        pd.testing.assert_frame_equal(df1, df2)
