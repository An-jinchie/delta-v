"""
train.py — One-time training script for the Delta-V characterization model.

Pipeline:
    1. Generate 2000 synthetic light curves (physics-based)
    2. Run inversion on each curve (deterministic Fourier math)
    3. Extract 17-dimensional feature vectors (inversion output + statistics)
    4. Train/test split (80/20)
    5. Train two RandomForest classifiers (size class + shape)
    6. Print classification reports
    7. Save model to data/models/characterize_model.pkl

Run this once before starting the Streamlit app:
    python train.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from pipeline.characterize.generator import (
    LightCurveGenerator,
    LightCurveParams,
    generate_dataset,
    N_SAMPLES_DEFAULT,
    CADENCE_S_DEFAULT,
    _SIZE_PARAMS,
    _SHAPES,
    _SIZES,
)
from pipeline.characterize.inversion import invert
from pipeline.characterize.features import extract_features, N_FEATURES
from pipeline.characterize.model import CharacterizeModel

import math

N_TRAIN_SAMPLES = 2000
DATASET_PATH    = "data/synthetic/light_curves.csv"
MODEL_PATH      = "data/models/characterize_model.pkl"
SEED            = 42
TEST_SPLIT      = 0.20
TARGET_ACCURACY = 0.80


def build_training_data(n_samples: int = N_TRAIN_SAMPLES, seed: int = SEED):
    """Generate light curves, invert each, extract features, return (X, y_size, y_shape)."""
    rng = np.random.default_rng(seed)
    gen = LightCurveGenerator()

    X        = np.zeros((n_samples, N_FEATURES), dtype=np.float64)
    y_size   = []
    y_shape  = []

    print(f"[train] Generating {n_samples} synthetic light curves + inversion...")
    t0 = time.time()

    for i in range(n_samples):
        size_class = str(rng.choice(_SIZES))
        shape      = str(rng.choice(_SHAPES))
        size_p     = _SIZE_PARAMS[size_class]

        rot_lo, rot_hi = size_p["rotation_rate_range"]
        rotation_rate_hz = float(rng.uniform(rot_lo, rot_hi))
        alb_lo, alb_hi = size_p["albedo_range"]
        albedo = float(rng.uniform(alb_lo, alb_hi))
        phase_offset = float(rng.uniform(0.0, 2 * math.pi))
        snr = float(rng.uniform(8.0, 40.0))

        params = LightCurveParams(
            size_class=size_class,
            shape=shape,
            rotation_rate_hz=rotation_rate_hz,
            phase_offset=phase_offset,
            albedo=albedo,
            snr=snr,
            n_samples=N_SAMPLES_DEFAULT,
            cadence_s=CADENCE_S_DEFAULT,
            seed=int(rng.integers(0, 2**31)),
        )
        lc  = gen.generate(params)
        inv = invert(lc, cadence_s=CADENCE_S_DEFAULT)
        X[i] = extract_features(lc, inv)
        y_size.append(size_class)
        y_shape.append(shape)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{n_samples}  ({elapsed:.1f}s elapsed)")

    print(f"[train] Data generation complete in {time.time()-t0:.1f}s")
    return X, np.array(y_size), np.array(y_shape)


def main():
    print("=" * 60)
    print("Delta-V — Characterization Model Training")
    print("=" * 60)

    # ── 1. Build feature matrix ───────────────────────────────────────────────
    X, y_size, y_shape = build_training_data()

    # ── 2. Train/test split ───────────────────────────────────────────────────
    (X_tr, X_te,
     ys_tr, ys_te,
     ysh_tr, ysh_te) = train_test_split(
        X, y_size, y_shape,
        test_size=TEST_SPLIT,
        random_state=SEED,
        stratify=y_size,  # stratify on size to ensure balanced split
    )
    print(f"\n[train] Train: {len(X_tr)} | Test: {len(X_te)}")

    # ── 3. Train model ────────────────────────────────────────────────────────
    print("\n[train] Training RandomForest classifiers...")
    t0 = time.time()
    model = CharacterizeModel(n_estimators=100, random_state=SEED)
    model.train(X_tr, ys_tr, ysh_tr)
    print(f"[train] Training complete in {time.time()-t0:.1f}s")

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    size_preds  = model._size_clf.predict(X_te)
    shape_preds = model._shape_clf.predict(X_te)

    print("\n--- Size Class Classification Report ---")
    print(classification_report(ys_te, size_preds, zero_division=0))

    print("--- Shape Classification Report ---")
    print(classification_report(ysh_te, shape_preds, zero_division=0))

    # Check accuracy threshold
    from sklearn.metrics import accuracy_score
    size_acc  = accuracy_score(ys_te, size_preds)
    shape_acc = accuracy_score(ysh_te, shape_preds)
    print(f"Size accuracy:  {size_acc*100:.1f}%")
    print(f"Shape accuracy: {shape_acc*100:.1f}%")

    if size_acc < TARGET_ACCURACY or shape_acc < TARGET_ACCURACY:
        print(
            f"\nWARN:  One or both classifiers below {TARGET_ACCURACY*100:.0f}% target. "
            "Consider increasing n_samples or tuning parameters."
        )
    else:
        print(f"\nOK:  Both classifiers meet the {TARGET_ACCURACY*100:.0f}% accuracy target.")

    # ── 5. Save ───────────────────────────────────────────────────────────────
    model.save(MODEL_PATH)
    print(f"\n[train] Model saved → {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()

