"""
Face Recognition Service — ZOOVOX
═══════════════════════════════════════════════════════════════════════════════
Library: DeepFace (Serengil & Ozpinar, 2020)
  "LightFace: A Hybrid Deep Face Recognition Framework"
  https://github.com/serengil/deepface

Model: ArcFace (Deng et al., 2019)
  "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
  CVPR 2019 — https://arxiv.org/abs/1801.07698
  Achieves 99.83% accuracy on LFW benchmark

Strategy:
  Enrollment: capture 3 frames → average embedding → store in MongoDB
  Verification: capture 1 frame → cosine similarity vs stored embedding
  Threshold: cosine distance < 0.40 (ArcFace recommended — Deng et al., 2019)

Liveness / Anti-Spoofing:
  Kept as a check fully separate from identity verification (see
  check_liveness() vs extract_embedding()/verify() below). Uses DeepFace's
  built-in Fasnet model — a single-frame, texture/artifact-based
  presentation-attack classifier (github.com/minivision-ai/
  Silent-Face-Anti-Spoofing), NOT blink/motion/interactive liveness. It
  raises the bar against casual spoofing (e.g. a phone photo) but is not a
  guarantee against a determined attacker with a high-quality print or
  screen replay.

  Face login is a CONVENIENCE authentication path, not a full security
  boundary — password login remains the authoritative path and stays fully
  available regardless of face-login/anti-spoofing availability. If the
  anti-spoofing model cannot be loaded, face login fails closed (503) rather
  than falling back to identity verification alone.

Security Notes:
  - Embeddings (128-dim float vector) stored, NOT raw images
  - New embeddings are encrypted at rest at the application layer (Fernet
    authenticated encryption — see encrypt_embedding()/decrypt_embedding()
    below), keyed by FACE_EMBEDDING_ENCRYPTION_KEY. Embeddings enrolled
    before this feature shipped remain plaintext and continue to work
    (backward-compatible reads) until the user re-enrolls or an operator
    runs a separate, explicit migration — nothing here rewrites them
    automatically.
  - GDPR: embeddings can be purged via /auth/face/delete endpoint
"""

import base64
import io
import json
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

COSINE_THRESHOLD = 0.40   # ArcFace recommended threshold
EMBEDDING_DIM = 128

# Literal, inspectable format/version tag checked before any decryption is
# attempted — an unrecognized stored value is rejected as such rather than
# fed into Fernet. Bump this (zvfe2$, ...) if the scheme ever changes.
_ENCRYPTED_FORMAT_PREFIX = "zvfe1$"


class EmbeddingEncryptionUnavailable(Exception):
    """No valid FACE_EMBEDDING_ENCRYPTION_KEY is configured — encryption or
    decryption cannot be attempted at all. Never includes the key or any
    embedding data in its message."""


class EmbeddingDecryptionError(Exception):
    """Decryption was attempted but failed — wrong key, corrupted/tampered
    ciphertext, or an unrecognized stored format. Never includes the key,
    ciphertext, or any recovered plaintext in its message."""


class FaceRecognitionService:

    def __init__(self):
        self._deepface_available = False
        self._checked = False
        # None = not yet checked. True/False cached after the first real
        # attempt, since a failed model load (e.g. a broken torch install)
        # will keep failing for the process lifetime — no point retrying an
        # expensive load on every request.
        self._anti_spoofing_available: Optional[bool] = None
        self._encryption_available: Optional[bool] = None

    def _ensure_deepface(self):
        if self._checked:
            return self._deepface_available
        try:
            from deepface import DeepFace  # noqa: F401
            self._deepface_available = True
            logger.info("DeepFace (ArcFace) loaded ✅")
        except ImportError:
            logger.warning("deepface not installed — face auth in simulation mode")
            self._deepface_available = False
        self._checked = True
        return self._deepface_available

    def _ensure_anti_spoofing(self) -> bool:
        """
        Verify — not assume — that DeepFace's anti-spoofing model (Fasnet)
        can actually be built in this environment. This is deliberately a
        separate check from _ensure_deepface(): DeepFace itself can be fully
        functional for face verification while anti-spoofing is unavailable
        (Fasnet has its own dependency, torch, which can be installed-but-
        broken independently of the rest of DeepFace).
        """
        if self._anti_spoofing_available is not None:
            return self._anti_spoofing_available
        if not self._ensure_deepface():
            self._anti_spoofing_available = False
            return False
        try:
            from deepface.modules.modeling import build_model
            build_model(task="spoofing", model_name="Fasnet")
            self._anti_spoofing_available = True
            logger.info("Anti-spoofing model (Fasnet) is available ✅")
        except Exception as e:
            self._anti_spoofing_available = False
            logger.error(
                f"Anti-spoofing model unavailable ({e}) — face login will fail "
                "closed until this is resolved. Password login is unaffected."
            )
        return self._anti_spoofing_available

    def _ensure_encryption_available(self) -> bool:
        """
        Verify — not assume — that a valid FACE_EMBEDDING_ENCRYPTION_KEY is
        configured, by actually constructing a Fernet instance from it.
        Cached after the first check (mirrors _ensure_anti_spoofing()).
        Never logs the key value itself, only whether it's usable.
        """
        if self._encryption_available is not None:
            return self._encryption_available
        from app.core.config import settings
        key = settings.FACE_EMBEDDING_ENCRYPTION_KEY
        if not key:
            self._encryption_available = False
            logger.error(
                "FACE_EMBEDDING_ENCRYPTION_KEY is not configured — face "
                "embedding encryption/decryption is unavailable. New "
                "enrollments will be rejected rather than stored in plaintext."
            )
            return False
        try:
            from cryptography.fernet import Fernet
            Fernet(key.encode() if isinstance(key, str) else key)
            self._encryption_available = True
        except Exception:
            self._encryption_available = False
            logger.error(
                "FACE_EMBEDDING_ENCRYPTION_KEY is configured but invalid "
                "(not a valid Fernet key) — face embedding encryption/"
                "decryption is unavailable."
            )
        return self._encryption_available

    # ── Embedding encryption at rest ──────────────────────────────────────
    #
    # New embeddings are always encrypted before being written to MongoDB.
    # Embeddings enrolled before this feature existed remain stored as plain
    # float lists; decrypt_embedding() recognizes and passes those through
    # unchanged rather than treating them as ciphertext. Nothing here
    # rewrites a legacy record — that would be a separate, explicit
    # migration decision, not something to do silently on read.

    def encrypt_embedding(self, embedding: list) -> str:
        """
        Encrypt a plaintext embedding for storage. Raises
        EmbeddingEncryptionUnavailable if no valid key is configured —
        callers must treat this as fail-closed and never store the
        plaintext embedding as a fallback.
        """
        if not self._ensure_encryption_available():
            raise EmbeddingEncryptionUnavailable(
                "Face embedding encryption key is not configured or invalid."
            )
        from app.core.config import settings
        from cryptography.fernet import Fernet
        f = Fernet(settings.FACE_EMBEDDING_ENCRYPTION_KEY.encode())
        token = f.encrypt(json.dumps(embedding).encode())
        return f"{_ENCRYPTED_FORMAT_PREFIX}{token.decode()}"

    def decrypt_embedding(self, stored) -> list:
        """
        Normalize whatever is stored in face_embedding into a usable
        plaintext vector, or raise a clear exception:
          - a list (legacy, pre-encryption record) is returned as-is
          - a string with the recognized format prefix is decrypted
          - anything else raises EmbeddingDecryptionError immediately,
            before any attempt to decrypt it
        Raises EmbeddingEncryptionUnavailable if no valid key is configured
        and the record actually needs decrypting. Never returns ciphertext
        as if it were a vector.
        """
        if isinstance(stored, list):
            return stored
        if not isinstance(stored, str) or not stored.startswith(_ENCRYPTED_FORMAT_PREFIX):
            raise EmbeddingDecryptionError(
                "Stored face embedding is not in a recognized format."
            )
        if not self._ensure_encryption_available():
            raise EmbeddingEncryptionUnavailable(
                "Face embedding encryption key is not configured or invalid."
            )
        from app.core.config import settings
        from cryptography.fernet import Fernet, InvalidToken
        f = Fernet(settings.FACE_EMBEDDING_ENCRYPTION_KEY.encode())
        token = stored[len(_ENCRYPTED_FORMAT_PREFIX):].encode()
        try:
            plaintext = f.decrypt(token)
        except InvalidToken as e:
            raise EmbeddingDecryptionError(
                "Failed to decrypt stored face embedding "
                "(wrong key or corrupted/tampered data)."
            ) from e
        return json.loads(plaintext)

    # ── Liveness / anti-spoofing ─────────────────────────────────────────
    #
    # Deliberately kept separate from extract_embedding()/verify() below.
    # Face verification answers "is this the enrolled person's face?".
    # This answers a different question: "does this input look like it came
    # from a live person rather than a photo/screen replay?". A frame can
    # pass one check and fail the other; callers must treat them as
    # independent gates, not a single combined score.

    def check_liveness(self, frame_b64: str) -> dict:
        """
        Anti-spoofing check via DeepFace's Fasnet model (a real, verified
        capability of the installed deepface==0.0.100 — not blink/motion
        detection, a single-frame texture/artifact classifier).

        Never raises. Returns {"status": ..., "score": float | None}, where
        status is one of:
          "unavailable"      — the anti-spoofing model could not be loaded
                                (e.g. a broken torch install). Callers MUST
                                treat this as fail-closed for authentication.
          "no_face_detected" — no face found in the frame.
          "spoof_detected"   — a face was found but flagged as not live.
          "live"             — passed the anti-spoofing check.
        """
        if not self._ensure_anti_spoofing():
            return {"status": "unavailable", "score": None}

        try:
            from deepface import DeepFace
            img_bytes = base64.b64decode(frame_b64)
            import cv2
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            faces = DeepFace.extract_faces(
                img_path=img,
                detector_backend="retinaface",
                enforce_detection=True,
                align=True,
                anti_spoofing=True,
            )
            is_real = bool(faces[0].get("is_real", True))
            score = float(faces[0].get("antispoof_score", 0.0))
            return {"status": "live" if is_real else "spoof_detected", "score": round(score, 4)}

        except Exception as e:
            logger.warning(f"Liveness check failed (treated as no face detected): {e}")
            return {"status": "no_face_detected", "score": None}

    # ── Embedding extraction (identity — no liveness signal here) ────────

    def extract_embedding(self, frame_b64: str) -> Optional[np.ndarray]:
        """
        Decode base64 JPEG, detect face, return 128-dim ArcFace embedding.
        Returns None if no face detected. Pure identity extraction — carries
        no information about liveness; see check_liveness() for that.
        """
        if not self._ensure_deepface():
            # Simulation mode — return deterministic pseudo-embedding from image hash
            import hashlib
            h = hashlib.sha256(frame_b64[:200].encode()).digest()
            vec = np.frombuffer(h * (EMBEDDING_DIM // 32 + 1), dtype=np.uint8)[:EMBEDDING_DIM].astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            return vec

        try:
            from deepface import DeepFace
            img_bytes = base64.b64decode(frame_b64)
            import cv2
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            result = DeepFace.represent(
                img_path=img,
                model_name="ArcFace",
                detector_backend="retinaface",   # best accuracy per DeepFace benchmarks
                enforce_detection=True,
                align=True,
            )
            embedding = np.array(result[0]["embedding"], dtype=np.float32)
            # L2-normalize for cosine similarity
            embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
            return embedding

        except Exception as e:
            logger.warning(f"Face embedding extraction failed: {e}")
            return None

    # ── Enrollment ────────────────────────────────────────────────────────

    def enroll(self, frames_b64: list[str]) -> Optional[list]:
        """
        Average embedding over multiple frames for robustness.
        Minimum 1 frame, recommended 3–5.
        """
        embeddings = []
        for frame in frames_b64:
            emb = self.extract_embedding(frame)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            return None

        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-9)
        return avg_embedding.tolist()

    # ── Verification ─────────────────────────────────────────────────────

    def verify(self, frame_b64: str, stored_embedding: list) -> dict:
        """
        Compare live frame embedding against stored enrollment embedding.
        Uses cosine distance (ArcFace optimal metric — Deng et al., 2019).
        """
        live_emb = self.extract_embedding(frame_b64)
        if live_emb is None:
            return {"verified": False, "reason": "no_face_detected", "distance": None}

        stored = np.array(stored_embedding, dtype=np.float32)
        stored = stored / (np.linalg.norm(stored) + 1e-9)

        # Cosine distance = 1 - cosine_similarity
        cosine_sim = float(np.dot(live_emb, stored))
        cosine_dist = 1.0 - cosine_sim

        verified = cosine_dist < COSINE_THRESHOLD
        confidence = max(0.0, min(1.0, 1.0 - cosine_dist / COSINE_THRESHOLD))

        return {
            "verified": verified,
            "distance": round(cosine_dist, 4),
            "confidence": round(confidence, 4),
            "threshold": COSINE_THRESHOLD,
            "reason": "match" if verified else "distance_exceeds_threshold",
        }


face_recognition_service = FaceRecognitionService()
