"""
Model-agnostic interface between audio preprocessing and animal-type inference.

AudioAnalysisService depends only on the AnimalClassifierModel protocol below —
never on scikit-learn, joblib, or any specific model artifact format directly.
Swapping the underlying model later (a different classical classifier, a CNN on
spectrograms, a hosted inference API) means writing a new adapter that satisfies
this protocol; nothing in AudioAnalysisService needs to change.

Artifact contract (what a training script must produce for the ZOOVOX
classifier to be loadable — see app/ml/zoovox_classifier.py):

    {
        "pipeline": <fitted estimator exposing predict_proba(X) -> (n_samples, n_classes)>,
        "classes": list[str],          # required, index-aligned to predict_proba columns
        "model_version": str,          # optional, defaults to "unknown"
        "feature_dim": int,            # optional; sanity-checked against input length if present
    }

Neither scripts/train_classifier.py nor scripts/train_model.py currently
produce this exact schema — bringing one of them into compliance is a
separate, not-yet-approved step.
"""

from dataclasses import dataclass
from typing import Dict, List, Protocol


@dataclass(frozen=True)
class AnimalPrediction:
    """A single model's animal-type prediction. Self-validating: constructing
    an instance with an invalid shape raises immediately, so any caller that
    successfully receives one can trust it without re-checking."""

    animal: str
    confidence: float
    class_probabilities: Dict[str, float]
    source: str  # "zoovox_classifier" | "yamnet" | "heuristic"
    model_version: str = "unknown"

    def __post_init__(self):
        if not self.animal:
            raise ValueError("AnimalPrediction.animal must be non-empty")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"AnimalPrediction.confidence out of range: {self.confidence!r}")
        if not self.class_probabilities:
            raise ValueError("AnimalPrediction.class_probabilities must be non-empty")
        if self.animal not in self.class_probabilities:
            raise ValueError(
                f"AnimalPrediction.animal {self.animal!r} not present in class_probabilities"
            )


class AnimalClassifierModel(Protocol):
    """Structural interface any animal-type classifier must satisfy to be
    consumed by AudioAnalysisService. No inheritance required — any object
    with matching attributes/methods satisfies this by duck typing."""

    classes: List[str]
    version: str

    def predict(self, feature_vector) -> AnimalPrediction: ...
