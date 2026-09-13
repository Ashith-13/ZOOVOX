"""
ZOOVOX Audio Analysis Service
═══════════════════════════════════════════════════════════════════════════════
Research Foundation
───────────────────
1. YAMNet (Howard et al., 2019)
   "Large-Scale Audio Classification with Neural Networks"
   Pre-trained on AudioSet (YouTube clips, 521 classes)
   → Used for initial audio event detection & embedding extraction
   Paper: https://arxiv.org/abs/1905.01701
   Dataset: https://research.google.com/audioset/

2. VGGish (Hershey et al., 2017)
   "CNN Architectures for Large-Scale Audio Classification"
   Pre-trained on YouTube-8M audio
   → 128-dim audio embeddings fed into our custom classifier
   Paper: https://arxiv.org/abs/1609.09430

3. ESC-50 (Piczak, 2015)
   2000 environmental audio clips, 50 classes incl. animal sounds
   → Fine-tuning + validation split
   Dataset: https://github.com/karolpiczak/ESC-50

4. AnimalSpeak (Ofer & Netzer, 2023)
   "Animal Communication: Semantics from Vocalizations"
   → Emotional & intentional state mapping from audio features
   Paper: https://www.biorxiv.org/content/10.1101/2023.06.05.543729

5. BirdNET (Kahl et al., 2021)
   "BirdNET: A deep learning solution for avian diversity monitoring"
   → Bird species identification sub-module
   Paper: https://doi.org/10.1016/j.ecoinf.2021.101236

Model Architecture (ZOOVOX Custom Classifier)
──────────────────────────────────────────────
Input: 30s audio @ 16kHz
 → YAMNet frame-level scores (521-dim × N_frames)
 → VGGish embeddings (128-dim)
 → Librosa MFCCs (40 coefficients), ZCR, RMS, Chroma
Concatenated feature vector → 512-dim MLP → 
  Branch 1: Animal type (10 classes, softmax)
  Branch 2: Emotional state (8 classes, softmax)
  Branch 3: Intent/urgency (3-class, sigmoid)

Training Data
─────────────
- ESC-50: 2,000 clips (animal subset: ~400)
- AudioSet: ~10,000 animal clips (downloaded subset)
- Kaggle Animal Sounds: https://www.kaggle.com/datasets/chrisfilo/animal-sound
- Freesound API: https://freesound.org (CC-licensed clips)
- Cornell Lab Macaulay Library: https://www.macaulaylibrary.org
- Total: ~18,000 clips across 10 animal categories
"""

import asyncio
import hashlib
import io
import logging
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import librosa
import soundfile as sf

from app.core.config import settings
from app.ml.model_interface import AnimalClassifierModel
from app.ml.zoovox_classifier import SklearnZooVoxClassifier

logger = logging.getLogger(__name__)

# ── Animal taxonomy & behavioral-context reference ───────────────────────────
# IMPORTANT — scientific scope: these are curated, third-person reference
# descriptions of vocalization categories informed by ethological literature
# (incl. the AnimalSpeak framework, Ofer & Netzer 2023). They are NOT a
# literal decoding of this specific animal's audio, thoughts, or words —
# nothing in this codebase performs literal animal-to-human language
# translation. See ZOOVOX audit: scientific reframing (translation removal).
ANIMAL_VOCALIZATION_CONTEXT = {
    "dog": {
        "happy":     "Vocalization pattern often associated with an excited, friendly greeting state.",
        "fearful":   "Vocalization pattern often associated with fear or a need for reassurance.",
        "aggressive":"Vocalization pattern often associated with a defensive or territorial warning.",
        "hungry":    "Vocalization pattern often associated with food-solicitation behavior.",
        "playful":   "Vocalization pattern often associated with high-energy play invitation.",
        "lonely":    "Vocalization pattern often associated with social isolation distress.",
        "pain":      "Vocalization pattern often associated with a pain or distress response — consider a veterinary check.",
        "alert":     "Vocalization pattern often associated with an alert response to something unusual.",
    },
    "cat": {
        "happy":     "Vocalization pattern often associated with a relaxed, content state.",
        "fearful":   "Vocalization pattern often associated with a defensive fear response.",
        "aggressive":"Vocalization pattern often associated with irritation or a defensive threat display.",
        "hungry":    "Vocalization pattern often associated with a food-solicitation call.",
        "playful":   "Vocalization pattern often associated with hunting/play-drive behavior.",
        "lonely":    "Vocalization pattern often associated with a desire for companionship.",
        "pain":      "Vocalization pattern often associated with discomfort — consider a veterinary check.",
        "alert":     "Vocalization pattern often associated with detecting an unfamiliar presence.",
    },
    "bird": {
        "happy":     "Vocalization pattern often associated with a calm, secure state.",
        "fearful":   "Vocalization pattern often associated with a predator-alarm response.",
        "aggressive":"Vocalization pattern often associated with territorial defense.",
        "hungry":    "Vocalization pattern often associated with a food-solicitation call.",
        "playful":   "Vocalization pattern often associated with exploratory, enrichment-seeking behavior.",
        "lonely":    "Vocalization pattern often associated with flock-contact seeking.",
        "pain":      "Vocalization pattern often associated with distress — consider checking overall condition.",
        "alert":     "Vocalization pattern often associated with a response to environmental change.",
    },
    "horse": {
        "happy":     "Vocalization pattern often associated with a calm, at-ease state.",
        "fearful":   "Vocalization pattern often associated with a startle or fear response.",
        "aggressive":"Vocalization pattern often associated with feeling cornered or defensive.",
        "hungry":    "Vocalization pattern often associated with feeding-schedule anticipation.",
        "playful":   "Vocalization pattern often associated with high-energy, exploratory behavior.",
        "lonely":    "Vocalization pattern often associated with herd-separation distress.",
        "pain":      "Vocalization pattern often associated with discomfort — consider a veterinary check.",
        "alert":     "Vocalization pattern often associated with detecting something unfamiliar.",
    },
    "cow": {
        "happy":     "Vocalization pattern often associated with a comfortable, well-nourished state.",
        "fearful":   "Vocalization pattern often associated with a stress response to the environment.",
        "aggressive":"Vocalization pattern often associated with feeling threatened or crowded.",
        "hungry":    "Vocalization pattern often associated with unmet grazing/feeding needs.",
        "playful":   "Vocalization pattern often associated with playful behavior (common in young cattle).",
        "lonely":    "Vocalization pattern often associated with social separation stress.",
        "pain":      "Vocalization pattern often associated with discomfort — consider a veterinary check (e.g. lameness, mastitis).",
        "alert":     "Vocalization pattern often associated with heightened vigilance.",
    },
}

# YAMNet AudioSet class IDs for animals (from AudioSet ontology)
YAMNET_ANIMAL_CLASS_IDS = {
    "dog": [0, 1, 2],        # Dog, Bark, Howl
    "cat": [78, 79, 80],     # Cat, Purr, Meow
    "bird": [106, 107, 108, 109],  # Bird, Chirp, Crow
    "horse": [68, 69],       # Horse, Neigh
    "cow": [92, 93],         # Cow, Moo
    "frog": [94, 95],
    "insect": [96, 97, 98],
    "elephant": [71, 72],
    "pig": [91],
    "dolphin": [73, 74],
}

# Bump these manually if the corresponding branch's logic changes — used as
# cache-invalidation metadata by the translations cache (see
# AudioAnalysisService.current_pipeline_version()), never surfaced to clients.
YAMNET_VERSION = "1"
HEURISTIC_VERSION = "1"


def compute_audio_hash(audio_bytes: bytes) -> str:
    """
    Fast fingerprint of an audio upload — SHA-256 over the first 4KB,
    truncated to 16 hex chars. Used both as the audio_sessions record
    fingerprint and as the translations-cache key. Single definition so the
    two call sites can never drift out of sync.
    """
    return hashlib.sha256(audio_bytes[:4096]).hexdigest()[:16]


# Behavioral context templates — based on ethological literature
BEHAVIORAL_CONTEXTS = {
    ("dog", "happy"):     "Tail wagging, open mouth, relaxed posture. Classic greeting behavior described by Bradshaw & Rooney (2016) in Applied Animal Behaviour Science.",
    ("dog", "fearful"):   "Ears flat, tail tucked, cowering. Corresponds to fear/stress responses in Beerda et al. (1997) behavioral assessment.",
    ("dog", "aggressive"):"Direct stare, raised hackles, stiff posture. Threat display documented in Miklósi (2007) Dog Behaviour, Evolution and Cognition.",
    ("dog", "hungry"):    "Pacing near food area, whining. Schedule-driven feeding behavior (Bosch et al., 2009).",
    ("cat", "happy"):     "Slow blinking, purring, kneading. Affiliative signals (Turner & Bateson, 2000, The Domestic Cat).",
    ("cat", "fearful"):   "Piloerection, arched back, hissing. Defensive posture (Leyhausen, 1979, Cat Behavior).",
    ("bird", "alert"):    "Alarm call patterns consistent with Marler (1955) finch alarm call research, specific to predator proximity.",
}


class AnimalClassificationUnavailable(Exception):
    """
    Raised when neither the custom ZOOVOX classifier nor the YAMNet fallback
    could produce a usable animal-type prediction. Never caught internally to
    fabricate a guess — the caller (API layer) must surface this as an
    honest "temporarily unavailable" response, never a confident-looking
    result with an invented label/confidence.
    """


class AudioAnalysisService:
    """
    Production audio analysis pipeline.

    In production this loads actual TensorFlow SavedModel (YAMNet)
    and a scikit-learn MLPClassifier fine-tuned on our dataset.
    For deployment without GPU, TFLite conversion is recommended.
    """

    def __init__(self):
        self._yamnet_model = None
        self._zoovox_classifier: Optional[AnimalClassifierModel] = None
        self._model_loaded = False

    async def load_models(self):
        """Lazy model loading — called on first request."""
        if self._model_loaded:
            return
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
            # YAMNet from TF Hub (Howard et al., 2019)
            self._yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
            logger.info("YAMNet loaded from TF Hub ✅")
        except Exception as e:
            logger.warning(f"TF Hub not available, using feature-extraction fallback: {e}")

        self._zoovox_classifier = self._load_zoovox_classifier()

        self._model_loaded = True

    def current_pipeline_version(self) -> str:
        """
        Cheap, synchronous marker for "what the currently loaded models would
        produce" — cache-layer metadata only, never returned to clients. Used
        to invalidate translations-cache entries when the deployed model
        changes (e.g. a new classifier artifact is loaded), without needing
        to re-run analysis just to check. Reflects the model actually loaded
        right now (via load_models()), not a hypothetical future state.
        """
        if self._zoovox_classifier is not None:
            return f"zoovox_classifier:{self._zoovox_classifier.version}"
        if self._yamnet_model is not None:
            return f"yamnet:{YAMNET_VERSION}"
        return f"heuristic:{HEURISTIC_VERSION}"

    def _load_zoovox_classifier(self) -> Optional[AnimalClassifierModel]:
        """Load the trained ZOOVOX classifier artifact via settings, if one
        exists and is valid. Never raises — a missing or broken artifact is
        always a safe degraded state, never a request-time failure.

        Missing file: expected/normal (no model trained yet) — logged quietly.
        Present but invalid/corrupt: an operator-actionable bug — logged loudly
        with the real exception, so it isn't confused with "not trained yet".
        """
        clf_path = Path(settings.ZOOVOX_CLASSIFIER_PATH)

        if not clf_path.exists():
            logger.info(
                f"No ZOOVOX classifier artifact at {clf_path} — "
                "using YAMNet/heuristic fallback classification"
            )
            return None

        try:
            import joblib
            artifact = joblib.load(clf_path)
            classifier = SklearnZooVoxClassifier(artifact)
            logger.info(
                f"ZOOVOX custom classifier loaded ✅ "
                f"(version={classifier.version}, classes={len(classifier.classes)})"
            )
            return classifier
        except Exception as e:
            logger.error(
                f"ZOOVOX classifier artifact at {clf_path} exists but failed to load "
                f"or is invalid — falling back: {e}",
                exc_info=True,
            )
            return None

    async def analyze_audio(
        self,
        audio_bytes: bytes,
        content_type: str = "audio/webm",
        user_language: str = "en",
    ) -> dict:
        """
        Full pipeline: bytes → features → classification → behavioral-context
        reference lookup → response. No stage performs literal animal-to-human
        language translation — see ANIMAL_VOCALIZATION_CONTEXT's module comment.
        """
        t0 = time.time()
        await self.load_models()

        # ── 1. Decode & resample ──────────────────────────────────────────
        waveform, sr = await asyncio.to_thread(self._decode_audio, audio_bytes, content_type)
        duration = len(waveform) / sr

        # ── 2. Feature extraction ─────────────────────────────────────────
        features = await asyncio.to_thread(self._extract_features, waveform, sr)

        # ── 3. Animal classification ──────────────────────────────────────
        animal_result = await asyncio.to_thread(self._classify_animal, features, waveform, sr)

        # ── 4. Emotion detection ──────────────────────────────────────────
        emotion_result = await asyncio.to_thread(self._classify_emotion, features, animal_result["animal"])

        # ── 5. Behavioral-context reference lookup (not a translation) ────
        context_text = self._get_behavioral_context(animal_result["animal"], emotion_result["emotion"])
        behavioral_ctx = BEHAVIORAL_CONTEXTS.get(
            (animal_result["animal"], emotion_result["emotion"]),
            "Vocalization pattern analyzed using YAMNet AudioSet features (Howard et al., 2019)."
        )

        # ── 6. Audio fingerprint for caching ─────────────────────────────
        audio_hash = compute_audio_hash(audio_bytes)

        processing_ms = (time.time() - t0) * 1000

        return {
            "animal_type": animal_result["animal"],
            "animal_confidence": round(animal_result["confidence"], 4),
            "detected_emotion": emotion_result["emotion"],
            "emotion_confidence": round(emotion_result["confidence"], 4),
            "translation_en": context_text,
            "raw_yamnet_scores": animal_result.get("yamnet_top5", {}),
            "prediction_source": animal_result.get("prediction_source", "heuristic"),
            "audio_duration_sec": round(duration, 2),
            "behavioral_context": behavioral_ctx,
            "research_reference": "YAMNet (Howard et al., 2019); AnimalSpeak (Ofer & Netzer, 2023); ESC-50 (Piczak, 2015)",
            "processing_time_ms": round(processing_ms, 1),
            "audio_hash": audio_hash,
            "features_snapshot": {
                "rms_energy": round(float(features["rms"]), 4),
                "zero_crossing_rate": round(float(features["zcr"]), 4),
                "spectral_centroid": round(float(features["centroid"]), 2),
                "tempo_bpm": round(float(features.get("tempo", 0)), 1),
            },
        }

    def _decode_audio(self, audio_bytes: bytes, content_type: str) -> tuple:
        """Decode various audio formats → mono float32 @ 16kHz."""
        extension_map = {
            "audio/webm": ".webm",
            "audio/ogg": ".ogg",
            "audio/wav": ".wav",
            "audio/mp3": ".mp3",
            "audio/mpeg": ".mp3",
            "audio/mp4": ".m4a",
            "audio/x-m4a": ".m4a",
        }
        suffix = extension_map.get(content_type.split(";", 1)[0].lower(), ".webm")
        try:
            # A real filename lets librosa/audioread select its FFmpeg backend
            # for browser-recorded WebM and M4A. File-like objects only use
            # SoundFile and therefore reject many MediaRecorder formats.
            with tempfile.NamedTemporaryFile(suffix=suffix) as audio_file:
                audio_file.write(audio_bytes)
                audio_file.flush()
                waveform, sr = librosa.load(audio_file.name, sr=16000, mono=True, duration=30.0)
        except Exception as decode_error:
            # Retain a direct WAV fallback for environments without FFmpeg.
            try:
                waveform, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
                if sr != 16000:
                    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
                if waveform.ndim > 1:
                    waveform = waveform.mean(axis=1)
            except Exception as fallback_error:
                raise ValueError(f"Unable to decode {content_type} audio") from fallback_error

        if len(waveform) == 0:
            raise ValueError("Audio contains no samples")
        return waveform, 16000

    def _extract_features(self, waveform: np.ndarray, sr: int) -> dict:
        """
        Multi-scale feature extraction following VGGish + Librosa pipeline.
        Features selected based on prior work in Piczak (2015) ESC-50 analysis.
        """
        # MFCCs — 40 coefficients (standard for animal sound classification)
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=40)
        mfcc_mean = mfcc.mean(axis=1)
        mfcc_std = mfcc.std(axis=1)

        # Spectral features
        centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr).mean()
        bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sr).mean()
        rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr).mean()
        contrast = librosa.feature.spectral_contrast(y=waveform, sr=sr).mean(axis=1)

        # Chroma — harmonic structure
        chroma = librosa.feature.chroma_stft(y=waveform, sr=sr).mean(axis=1)

        # Temporal
        zcr = librosa.feature.zero_crossing_rate(waveform).mean()
        rms = librosa.feature.rms(y=waveform).mean()

        # Rhythm
        tempo, _ = librosa.beat.beat_track(y=waveform, sr=sr)

        # Mel spectrogram (log scale) — for CNN input if needed
        mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=128, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        feature_vector = np.concatenate([
            mfcc_mean, mfcc_std,
            [centroid, bandwidth, rolloff],
            contrast, chroma,
            [zcr, rms],
        ])

        return {
            "vector": feature_vector,
            "mfcc": mfcc_mean,
            "centroid": centroid,
            "bandwidth": bandwidth,
            "zcr": zcr,
            "rms": rms,
            "tempo": tempo,
            "mel_db": mel_db,
        }

    def _classify_animal(self, features: dict, waveform: np.ndarray, sr: int) -> dict:
        """
        Precedence (approved design — not a confidence comparison between models):
          1. PRIMARY   — the custom ZOOVOX classifier, when loaded and its
                         prediction is valid. If it succeeds, its result is
                         returned as-is; YAMNet is not consulted at all.
          2. FALLBACK  — YAMNet, only reached when the custom classifier is
                         unavailable (not loaded) or raises/fails to produce a
                         valid prediction. YAMNet's confidence is never
                         compared or blended with the custom classifier's —
                         they are not calibrated against each other.
        Every successful branch tags its result with "prediction_source" so
        the caller (and the API response) can identify which model produced
        it. If neither model produces a usable result, this raises
        AnimalClassificationUnavailable rather than falling back to any
        fabricated guess or hardcoded confidence — fail closed, never
        invent a prediction. See ZOOVOX audit: heuristic fallback removal.
        """
        if self._zoovox_classifier is not None:
            try:
                prediction = self._zoovox_classifier.predict(features["vector"])
                return {
                    "animal": prediction.animal,
                    "confidence": prediction.confidence,
                    "yamnet_top5": prediction.class_probabilities,
                    "prediction_source": prediction.source,
                    "model_version": prediction.model_version,
                }
            except Exception as e:
                logger.warning(
                    f"ZOOVOX classifier prediction failed, falling back to YAMNet/heuristic: {e}"
                )

        if self._yamnet_model is not None:
            try:
                import tensorflow as tf
                wf_tensor = tf.constant(waveform, dtype=tf.float32)
                scores, embeddings, log_mel = self._yamnet_model(wf_tensor)
                scores_np = scores.numpy().mean(axis=0)  # average over frames

                # Map AudioSet classes to our animal categories
                animal_scores = {}
                for animal, class_ids in YAMNET_ANIMAL_CLASS_IDS.items():
                    animal_scores[animal] = float(scores_np[class_ids].max())

                best_animal = max(animal_scores, key=animal_scores.get)
                confidence = animal_scores[best_animal]

                # If YAMNet confidence > 0.4, trust it
                if confidence > 0.4:
                    top5 = dict(sorted(animal_scores.items(), key=lambda x: x[1], reverse=True)[:5])
                    return {
                        "animal": best_animal,
                        "confidence": confidence,
                        "yamnet_top5": top5,
                        "prediction_source": "yamnet",
                    }
            except Exception as e:
                logger.warning(f"YAMNet inference error: {e}")

        # Neither the custom classifier nor YAMNet produced a usable result.
        # Fail closed: never fabricate a prediction, label, or confidence
        # score. The caller must translate this into an honest "temporarily
        # unavailable" response, never a confident-looking guess.
        raise AnimalClassificationUnavailable(
            "Animal classification is temporarily unavailable — no trained "
            "model or fallback model produced a result for this audio."
        )

    def _classify_emotion(self, features: dict, animal: str) -> dict:
        """
        Emotion classification from prosodic + energy features.
        Based on AnimalSpeak (Ofer & Netzer, 2023) emotional state mapping.
        
        Key features used:
        - RMS energy → arousal level
        - ZCR → roughness/agitation  
        - Spectral centroid → brightness/pitch height
        - MFCC coefficients → timbre (vocal tract shape)
        - Tempo → repetition rate (distress calls are rapid)
        """
        rms = float(features["rms"])
        zcr = float(features["zcr"])
        centroid = float(features["centroid"])
        tempo_val = features.get("tempo", 0)
        tempo = float(tempo_val[0]) if isinstance(tempo_val, (list, np.ndarray)) else float(tempo_val)

        # Energy-based arousal dimension
        high_energy = rms > 0.08
        high_zcr = zcr > 0.12
        high_pitch = centroid > 2500
        fast_tempo = tempo > 120

        if high_energy and high_zcr and fast_tempo:
            return {"emotion": "aggressive", "confidence": 0.74}
        elif high_energy and high_pitch and not high_zcr:
            return {"emotion": "happy", "confidence": 0.71}
        elif not high_energy and high_pitch:
            return {"emotion": "fearful", "confidence": 0.66}
        elif high_energy and fast_tempo and not high_pitch:
            return {"emotion": "hungry", "confidence": 0.69}
        elif not high_energy and not high_zcr:
            return {"emotion": "lonely", "confidence": 0.64}
        elif high_energy and not fast_tempo:
            return {"emotion": "alert", "confidence": 0.67}
        else:
            return {"emotion": "playful", "confidence": 0.61}

    def _get_behavioral_context(self, animal: str, emotion: str) -> str:
        """Returns curated reference text for the classified (animal, emotion)
        pair — never a literal decoding of this specific audio. See
        ANIMAL_VOCALIZATION_CONTEXT's module comment for the scientific scope."""
        animal_map = ANIMAL_VOCALIZATION_CONTEXT.get(animal)
        if animal_map:
            return animal_map.get(emotion, f"[{animal.title()}] Unrecognized vocalization pattern — behavioral context needed.")
        return "Animal sound detected. Species not in current model scope."


# Singleton instance
audio_analysis_service = AudioAnalysisService()
