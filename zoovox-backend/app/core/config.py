"""
Central configuration — all env vars validated at startup via Pydantic Settings.
Copy .env.example to .env and fill in your values.
"""

from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, validator
from typing import List, Optional
import secrets


class Settings(BaseSettings):
    # ── App ───────────────────────────────────────────────────────────────
    APP_NAME: str = "ZOOVOX"
    ENVIRONMENT: str = "development"          # development | staging | production
    DEBUG: bool = False
    SECRET_KEY: str = secrets.token_urlsafe(64)

    # ── MongoDB Atlas ────────────────────────────────────────────────────
    MONGODB_URL: str = "mongodb+srv://<user>:<pass>@cluster0.mongodb.net/zoovox?retryWrites=true&w=majority"
    MONGODB_DB_NAME: str = "zoovox"
    # Disables MongoDB TLS certificate validation. Defaults to False (secure).
    # Only takes effect when ENVIRONMENT == "development" — see db/mongodb.py.
    # Never set true in staging/production.
    MONGODB_ALLOW_INSECURE_TLS: bool = False

    # ── JWT ───────────────────────────────────────────────────────────────
    JWT_SECRET_KEY: str = secrets.token_urlsafe(64)
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    # ── CORS ─────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",
        "https://zoovox.app",      # production frontend
    ]

    # ── Google APIs ──────────────────────────────────────────────────────
    GOOGLE_MAPS_API_KEY: str = ""             # Places API for vet geolocation
    GOOGLE_TRANSLATE_API_KEY: str = ""        # Multilingual translation

    # ── ML Model Paths ───────────────────────────────────────────────────
    YAMNET_MODEL_PATH: str = "models/yamnet"
    VGGISH_MODEL_PATH: str = "models/vggish"
    ZOOVOX_CLASSIFIER_PATH: str = "models/zoovox_classifier.pkl"
    FACE_EMBEDDINGS_DIM: int = 128            # DeepFace embedding dimension

    # ── Face Embedding Encryption ────────────────────────────────────────
    # Fernet key (urlsafe-base64, 32 bytes) protecting stored face
    # embeddings at rest. Deliberately NOT auto-generated like SECRET_KEY/
    # JWT_SECRET_KEY above — an auto-generated key would silently change on
    # every process restart, permanently losing the ability to decrypt any
    # previously-encrypted embedding. Must be explicitly set; unset means
    # encryption is unavailable (new enrollments fail closed rather than
    # falling back to plaintext).
    # Generate with:
    #   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    FACE_EMBEDDING_ENCRYPTION_KEY: Optional[str] = None

    # ── Audio ────────────────────────────────────────────────────────────
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_MAX_DURATION_SECONDS: int = 30
    AUDIO_MAX_FILE_MB: int = 10
    AUDIO_TEMP_DIR: str = "/tmp/zoovox_audio"

    # ── Cloudinary (audio/image storage) ────────────────────────────────
    CLOUDINARY_CLOUD_NAME: str = ""
    CLOUDINARY_API_KEY: str = ""
    CLOUDINARY_API_SECRET: str = ""

    # ── Redis (rate-limiting & session cache) ────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── Email (optional — for verification) ─────────────────────────────
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    EMAILS_FROM: str = "noreply@zoovox.app"

    # ── Rate Limiting ────────────────────────────────────────────────────
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
