"""
Pydantic v2 schemas — request / response models for all API endpoints
"""

from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional, List, Literal
from datetime import datetime
import re


# ── Auth ──────────────────────────────────────────────────────────────────────

class UserRegisterRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=80)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)

    @field_validator("password")
    @classmethod
    def password_strength(cls, v):
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserLoginRequest(BaseModel):
    email: EmailStr
    password: str


class FaceLoginRequest(BaseModel):
    """Base64-encoded JPEG frame from webcam"""
    frame_b64: str = Field(..., description="Base64 encoded JPEG image from MediaRecorder")


class FaceEnrollRequest(BaseModel):
    frame_b64: str
    user_id: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: "UserPublicProfile"


class RefreshRequest(BaseModel):
    refresh_token: str


# ── User ──────────────────────────────────────────────────────────────────────

class UserPublicProfile(BaseModel):
    id: str
    name: str
    email: str
    avatar_url: Optional[str] = None
    face_enrolled: bool = False
    plan: Literal["free", "pro", "enterprise"] = "free"
    animals_analyzed: int = 0
    created_at: datetime


# ── Audio & Translation ───────────────────────────────────────────────────────

class AudioAnalysisResponse(BaseModel):
    session_id: str
    animal_type: str
    animal_confidence: float = Field(..., ge=0.0, le=1.0)
    detected_emotion: str
    emotion_confidence: float = Field(..., ge=0.0, le=1.0)
    translation_en: str = Field(
        ...,
        description=(
            "Curated behavioral/contextual reference text for the classified "
            "vocalization — NOT a literal translation or decoding of this "
            "specific animal's audio, thoughts, or words."
        ),
    )
    translation_local: Optional[str] = None
    raw_yamnet_scores: dict
    prediction_source: str = Field(
        "heuristic",
        description="Which model produced the primary animal-type prediction: "
        "'zoovox_classifier', 'yamnet', or 'heuristic'.",
    )
    audio_duration_sec: float
    mel_spectrogram_url: Optional[str] = None
    reverse_cue_url: Optional[str] = None    # TTS audio for animal feedback
    behavioral_context: str = Field(
        ...,
        description=(
            "General reference information about typical behavior associated "
            "with the classified category — not derived from this specific "
            "recording."
        ),
    )
    research_reference: str
    processing_time_ms: float


class HumanToAnimalRequest(BaseModel):
    text: str = Field(..., max_length=500)
    target_animal: Literal["dog", "cat", "bird", "horse", "cow", "elephant", "dolphin", "pig"]
    language: str = Field("en", max_length=10)


class HumanToAnimalResponse(BaseModel):
    session_id: str
    original_text: str
    animal_cue_description: str
    audio_url: str               # Coqui TTS synthesized audio
    recommended_actions: List[str]
    scientific_basis: str


class TranslationHistoryItem(BaseModel):
    session_id: str
    direction: Literal["animal_to_human", "human_to_animal"]
    animal_type: str
    translation: str = Field(
        ...,
        description="Curated behavioral/contextual reference text — not a literal translation.",
    )
    confidence: float
    created_at: datetime


# ── Veterinary ────────────────────────────────────────────────────────────────

class VetSearchRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    radius_km: float = Field(10.0, ge=0.1, le=100.0)
    type: Literal["hospital", "clinic", "pharmacy", "store"] = "hospital"


class VetService(BaseModel):
    id: str
    name: str
    type: str
    address: str
    distance_km: float
    latitude: float
    longitude: float
    phone: Optional[str]
    website: Optional[str]
    rating: Optional[float]
    open_now: Optional[bool]
    hours: Optional[str]
    google_place_id: Optional[str]


class VetSearchResponse(BaseModel):
    results: List[VetService]
    total: int
    radius_km: float


# ── Analytics ─────────────────────────────────────────────────────────────────

class AnalyticsSummary(BaseModel):
    total_sessions: int
    animal_breakdown: dict
    emotion_breakdown: dict
    avg_confidence: float
    top_animals: List[str]
    weekly_trend: List[dict]


TokenResponse.model_rebuild()
