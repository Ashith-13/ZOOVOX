"""
Adapter that wraps a joblib-persisted scikit-learn artifact and exposes it
through the AnimalClassifierModel protocol (see app/ml/model_interface.py).

This is the only place in the codebase that knows the artifact is a
scikit-learn Pipeline with predict_proba(). AudioAnalysisService never
imports sklearn/joblib directly — it only ever talks to this adapter's
predict() method.
"""

import logging

import numpy as np

from app.ml.model_interface import AnimalPrediction

logger = logging.getLogger(__name__)


class SklearnZooVoxClassifier:
    """Adapts a loaded artifact dict to the AnimalClassifierModel protocol.

    Expected artifact schema (see model_interface.py docstring for the full
    contract):
        {
            "pipeline": <fitted estimator with predict_proba(X)>,
            "classes": list[str],
            "model_version": str,   # optional
            "feature_dim": int,     # optional
        }

    Raises ValueError at construction time if the artifact does not satisfy
    this shape — callers should treat that as a load failure, not attempt to
    use a partially-valid instance.
    """

    def __init__(self, artifact: dict):
        if not isinstance(artifact, dict):
            raise ValueError(f"Expected a dict artifact, got {type(artifact).__name__}")

        pipeline = artifact.get("pipeline")
        classes = artifact.get("classes")

        if pipeline is None or not hasattr(pipeline, "predict_proba"):
            raise ValueError("Artifact is missing a 'pipeline' exposing predict_proba(X)")
        if not classes or not isinstance(classes, (list, tuple)):
            raise ValueError("Artifact is missing a non-empty 'classes' list")

        self._pipeline = pipeline
        self.classes = [str(c) for c in classes]
        self.version = str(artifact.get("model_version", "unknown"))
        self._feature_dim = artifact.get("feature_dim")

    def predict(self, feature_vector) -> AnimalPrediction:
        vector = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)

        if self._feature_dim is not None and vector.shape[1] != self._feature_dim:
            raise ValueError(
                f"Feature vector length {vector.shape[1]} does not match "
                f"artifact feature_dim {self._feature_dim}"
            )

        proba = self._pipeline.predict_proba(vector)[0]
        if len(proba) != len(self.classes):
            raise ValueError(
                f"predict_proba returned {len(proba)} scores but artifact "
                f"declares {len(self.classes)} classes"
            )

        class_probabilities = {label: float(p) for label, p in zip(self.classes, proba)}
        best_idx = int(np.argmax(proba))

        return AnimalPrediction(
            animal=self.classes[best_idx],
            confidence=float(proba[best_idx]),
            class_probabilities=class_probabilities,
            source="zoovox_classifier",
            model_version=self.version,
        )
