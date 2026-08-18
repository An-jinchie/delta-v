"""
pipeline/characterize/model.py — RandomForest Classifier (Secondary Refinement)

This classifier is SECONDARY to the deterministic inversion in inversion.py.
It refines the shape and size labels on top of physically-extracted features —
it does not replace the inversion math.

Architecture
------------
Two independent RandomForestClassifiers:
  1. size_clf   → predicts size_class ("small" | "medium" | "large")
  2. shape_clf  → predicts shape      ("sphere" | "flat_plate" | "cylinder" | "tumbling")

Both are trained on the 17-dimensional feature vector from features.extract_features().

The predict() method always returns the InversionResult alongside the ML output,
so every prediction is inspectable — you can see both what the math said and
what the classifier said.
"""

from __future__ import annotations

import os
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from pipeline.characterize.features import extract_features, N_FEATURES
from pipeline.characterize.inversion import InversionResult, invert

# Label sets (fixed order for reproducibility)
SIZE_CLASSES  = ["large", "medium", "small"]
SHAPE_CLASSES = ["cylinder", "flat_plate", "sphere", "tumbling"]


class CharacterizeModel:
    """Wrapper around two RandomForest classifiers for size and shape prediction.

    Usage
    -----
    Train and save:
        model = CharacterizeModel()
        model.train(X, y_size, y_shape)
        model.save("data/models/characterize_model.pkl")

    Load and predict:
        model = CharacterizeModel.load("data/models/characterize_model.pkl")
        result = model.predict(light_curve, inversion_result)
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._size_clf:  Optional[RandomForestClassifier] = None
        self._shape_clf: Optional[RandomForestClassifier] = None
        self._trained = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        y_size: np.ndarray,
        y_shape: np.ndarray,
    ) -> None:
        """Fit both classifiers.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, N_FEATURES)
            Feature matrix from extract_features().
        y_size : np.ndarray, shape (n_samples,)
            Size class labels: "small" | "medium" | "large"
        y_shape : np.ndarray, shape (n_samples,)
            Shape labels: "sphere" | "flat_plate" | "cylinder" | "tumbling"
        """
        self._size_clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._shape_clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._size_clf.fit(X, y_size)
        self._shape_clf.fit(X, y_shape)
        self._trained = True

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist both classifiers to a single joblib file."""
        if not self._trained:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "size_clf":    self._size_clf,
            "shape_clf":   self._shape_clf,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str) -> "CharacterizeModel":
        """Load a previously saved model from a joblib file."""
        payload = joblib.load(path)
        instance = cls(
            n_estimators=payload["n_estimators"],
            random_state=payload["random_state"],
        )
        instance._size_clf  = payload["size_clf"]
        instance._shape_clf = payload["shape_clf"]
        instance._trained = True
        return instance

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        lc: np.ndarray,
        inv: Optional[InversionResult] = None,
        cadence_s: float = 0.1,
    ) -> dict:
        """Predict size class and shape from a light curve.

        The InversionResult is always included in the returned dict so every
        prediction is inspectable — callers can see the underlying math.

        Parameters
        ----------
        lc : np.ndarray
            Brightness time-series.
        inv : InversionResult, optional
            Pre-computed inversion result. If None, invert() is called internally.
        cadence_s : float
            Cadence in seconds (used only if inv is None).

        Returns
        -------
        dict with keys:
            size_class         : str   — ML-predicted size class
            shape              : str   — ML-predicted shape
            size_confidence    : float — probability of predicted size class [0, 1]
            shape_confidence   : float — probability of predicted shape [0, 1]
            inversion_result   : InversionResult — underlying math output
            features           : np.ndarray — the feature vector used for prediction
        """
        if not self._trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        if inv is None:
            inv = invert(lc, cadence_s=cadence_s)

        features = extract_features(lc, inv).reshape(1, -1)

        size_pred  = self._size_clf.predict(features)[0]
        shape_pred = self._shape_clf.predict(features)[0]

        size_proba  = self._size_clf.predict_proba(features)[0]
        shape_proba = self._shape_clf.predict_proba(features)[0]

        size_conf  = float(np.max(size_proba))
        shape_conf = float(np.max(shape_proba))

        return {
            "size_class":        str(size_pred),
            "shape":             str(shape_pred),
            "size_confidence":   size_conf,
            "shape_confidence":  shape_conf,
            "inversion_result":  inv,
            "features":          features.flatten(),
        }
