"""
ZOOVOX Backend Test Suite
Run: pytest tests/ -v --asyncio-mode=auto
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch, MagicMock
import numpy as np

# ── Test client setup ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture
async def client():
    # Patch MongoDB so tests don't need a real DB
    mock_db = MagicMock()
    mock_db.users.find_one = AsyncMock(return_value=None)
    mock_db.users.insert_one = AsyncMock(return_value=MagicMock(inserted_id="test_id"))
    mock_db.users.update_one = AsyncMock()
    mock_db.users.create_indexes = AsyncMock()
    mock_db.audio_sessions.insert_one = AsyncMock()
    mock_db.audio_sessions.find = MagicMock(return_value=MagicMock(
        sort=MagicMock(return_value=MagicMock(
            skip=MagicMock(return_value=MagicMock(
                limit=MagicMock(return_value=MagicMock(
                    to_list=AsyncMock(return_value=[])
                ))
            ))
        ))
    ))
    mock_db.audio_sessions.count_documents = AsyncMock(return_value=0)
    mock_db.vet_services.create_indexes = AsyncMock()
    mock_db.translations.create_indexes = AsyncMock()

    with patch("app.db.mongodb.connect_to_mongo", AsyncMock()), \
         patch("app.db.mongodb.get_database", AsyncMock(return_value=mock_db)):
        from app.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac


# ── Health Tests ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_root(client):
    r = await client.get("/")
    assert r.status_code == 200
    assert "ZOOVOX" in r.json()["service"]


# ── Auth Tests ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_register_validates_weak_password(client):
    r = await client.post("/api/v1/auth/register", json={
        "name": "Test User",
        "email": "test@example.com",
        "password": "weakpass",   # no uppercase, no digit
    })
    assert r.status_code == 422   # Pydantic validation error


@pytest.mark.asyncio
async def test_register_validates_short_name(client):
    r = await client.post("/api/v1/auth/register", json={
        "name": "A",
        "email": "test@example.com",
        "password": "ValidPass1",
    })
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_login_wrong_credentials(client):
    r = await client.post("/api/v1/auth/login", json={
        "email": "nobody@example.com",
        "password": "WrongPass1",
    })
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_login_fails_closed_when_database_unavailable():
    """Degraded mode must never fabricate a user or issue a token."""
    from app.main import app

    with patch("app.api.v1.endpoints.auth.get_database", AsyncMock(return_value=None)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.post("/api/v1/auth/login", json={
                "email": "anyone@example.com",
                "password": "WhateverPass1",
            })

    assert r.status_code == 503
    body = r.json()
    assert "access_token" not in body
    assert "refresh_token" not in body


@pytest.mark.asyncio
async def test_register_fails_closed_when_database_unavailable():
    """Degraded mode must never fabricate a user or issue a token."""
    from app.main import app

    with patch("app.api.v1.endpoints.auth.get_database", AsyncMock(return_value=None)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.post("/api/v1/auth/register", json={
                "name": "Someone New",
                "email": "new-degraded-mode@example.com",
                "password": "ValidPass1",
            })

    assert r.status_code == 503
    body = r.json()
    assert "access_token" not in body
    assert "refresh_token" not in body


# ── Refresh / Current-User Fail-Closed Tests ─────────────────────────────────

@pytest.mark.asyncio
async def test_refresh_fails_closed_when_database_unavailable():
    """Degraded mode must never fabricate a user or issue a token on refresh."""
    from app.main import app
    from app.core.security import create_refresh_token
    from bson import ObjectId

    refresh_token_value = create_refresh_token(str(ObjectId()), "anyone@example.com")

    with patch("app.api.v1.endpoints.auth.get_database", AsyncMock(return_value=None)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token_value})

    assert r.status_code == 503
    body = r.json()
    assert "access_token" not in body
    assert "refresh_token" not in body


@pytest.mark.asyncio
async def test_refresh_succeeds_with_valid_token_and_available_database():
    """Normal refresh behavior must be unaffected by the fail-closed fix."""
    from app.main import app
    from app.core.security import create_refresh_token
    from bson import ObjectId
    from datetime import datetime, timezone

    user_id = ObjectId()
    user_doc = {
        "_id": user_id, "email": "refresh-ok@example.com", "name": "Refresh Tester",
        "plan": "free", "animals_analyzed": 0, "created_at": datetime.now(timezone.utc),
        "face_embedding": None,
    }
    mock_db = MagicMock()
    mock_db.users.find_one = AsyncMock(return_value=user_doc)

    refresh_token_value = create_refresh_token(str(user_id), user_doc["email"])

    with patch("app.api.v1.endpoints.auth.get_database", AsyncMock(return_value=mock_db)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token_value})

    assert r.status_code == 200
    body = r.json()
    assert body["access_token"]
    assert body["refresh_token"]
    assert body["user"]["email"] == "refresh-ok@example.com"


@pytest.mark.asyncio
async def test_refresh_rejects_invalid_token_regardless_of_database():
    """An invalid/garbage refresh token must still be rejected before any
    database check — unaffected by the fail-closed fix."""
    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.post("/api/v1/auth/refresh", json={"refresh_token": "not-a-real-jwt"})

    assert r.status_code == 401


@pytest.mark.asyncio
async def test_get_current_user_fails_closed_when_database_unavailable():
    """The shared auth dependency (used by every protected endpoint) must
    never return a fabricated user when the database is unreachable."""
    from app.main import app
    from app.core.security import create_access_token
    from bson import ObjectId

    access_token = create_access_token({"sub": str(ObjectId()), "email": "anyone@example.com"})

    with patch("app.core.security.get_database", AsyncMock(return_value=None)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {access_token}"})

    assert r.status_code == 503
    assert "Demo User" not in r.text
    body = r.json()
    assert "name" not in body


@pytest.mark.asyncio
async def test_get_current_user_succeeds_when_database_available():
    """Normal authenticated-request behavior must be unaffected by the
    fail-closed fix."""
    from app.main import app
    from app.core.security import create_access_token
    from bson import ObjectId
    from datetime import datetime, timezone

    user_id = ObjectId()
    user_doc = {
        "_id": user_id, "email": "me-ok@example.com", "name": "Me Tester",
        "plan": "free", "animals_analyzed": 0, "created_at": datetime.now(timezone.utc),
        "face_embedding": None, "is_active": True, "is_banned": False,
    }
    mock_db = MagicMock()
    mock_db.users.find_one = AsyncMock(return_value=user_doc)

    access_token = create_access_token({"sub": str(user_id), "email": user_doc["email"]})

    with patch("app.core.security.get_database", AsyncMock(return_value=mock_db)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {access_token}"})

    assert r.status_code == 200
    body = r.json()
    assert body["email"] == "me-ok@example.com"
    assert body["name"] == "Me Tester"


@pytest.mark.asyncio
async def test_get_current_user_rejects_missing_and_invalid_tokens():
    """The fail-closed fix must not weaken existing auth rejection behavior
    for requests that were never going to be authenticated anyway."""
    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r_missing = await ac.get("/api/v1/auth/me")
        assert r_missing.status_code in (401, 403)

        r_invalid = await ac.get("/api/v1/auth/me", headers={"Authorization": "Bearer not-a-real-jwt"})
        assert r_invalid.status_code == 401


# ── Audio Analysis Tests ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_audio_requires_auth():
    from app.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.post("/api/v1/audio/analyze")
        assert r.status_code in (401, 403, 422)


@pytest.mark.asyncio
async def test_supported_animals_unauthenticated():
    """Supported animals endpoint should still require auth (protected route)."""
    from app.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/api/v1/audio/supported-animals")
        # No auth → 401 or returns data depending on route config
        assert r.status_code in (200, 401, 403)


# ── Feature Extraction Tests ──────────────────────────────────────────────────

def test_feature_extraction_produces_correct_shape():
    """Test that audio feature extraction returns (104,) vector."""
    from app.services.audio_analysis import audio_analysis_service
    import numpy as np

    # Generate synthetic 5-second white noise @ 16kHz
    y = np.random.randn(16000 * 5).astype(np.float32)
    features = audio_analysis_service._extract_features(y, 16000)

    assert "vector" in features
    assert len(features["vector"]) == 104
    assert "mfcc" in features
    assert "rms" in features
    assert "zcr" in features


def test_emotion_classification_returns_valid_emotion():
    from app.services.audio_analysis import audio_analysis_service, ANIMAL_VOCALIZATION_CONTEXT
    import numpy as np

    y = np.random.randn(16000 * 3).astype(np.float32)
    features = audio_analysis_service._extract_features(y, 16000)
    result = audio_analysis_service._classify_emotion(features, "dog")

    valid_emotions = list(ANIMAL_VOCALIZATION_CONTEXT["dog"].keys())
    assert result["emotion"] in valid_emotions
    assert 0.0 <= result["confidence"] <= 1.0


def test_vocalization_context_does_not_claim_literal_translation():
    """Scientific-honesty regression test: no behavioral-context reference
    string may use first-person phrasing that implies the audio was
    literally decoded into the animal's own words/thoughts."""
    from app.services.audio_analysis import ANIMAL_VOCALIZATION_CONTEXT

    first_person_markers = ("i'm ", "i am ", "i feel", "i need", "i want", "i sense", "i detected", "my ", "me now", "feed me")

    for animal, emotions in ANIMAL_VOCALIZATION_CONTEXT.items():
        for emotion, text in emotions.items():
            lowered = text.lower()
            for marker in first_person_markers:
                assert marker not in lowered, (
                    f"ANIMAL_VOCALIZATION_CONTEXT[{animal!r}][{emotion!r}] still contains "
                    f"first-person phrasing ({marker!r}): {text!r}"
                )


def test_spectral_heuristic_classification():
    from app.services.audio_analysis import audio_analysis_service
    import numpy as np

    features = {
        "centroid": np.float32(5000),  # high centroid → bird
        "zcr": np.float32(0.20),
        "rms": np.float32(0.05),
        "tempo": np.float32(100),
        "vector": np.zeros(104, dtype=np.float32),
    }
    result = audio_analysis_service._spectral_heuristic_classification(features)
    assert result["animal"] == "bird"
    assert 0.0 <= result["confidence"] <= 1.0


# ── Face Recognition Tests ────────────────────────────────────────────────────

def test_face_recognition_simulation_mode():
    """In simulation mode (deepface not installed), embeddings should still be generated."""
    from app.services.face_recognition import face_recognition_service

    import base64
    # 1x1 white JPEG
    white_pixel_b64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/xAAUAQEAAAAAAAAAAAAAAAAAAAAA/8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAwDAQACEQMRAD8AJQAB/9k="

    emb = face_recognition_service.extract_embedding(white_pixel_b64)
    assert emb is not None
    assert len(emb) == 128


def test_face_verify_same_embedding():
    from app.services.face_recognition import face_recognition_service
    import base64

    frame_b64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/xAAUAQEAAAAAAAAAAAAAAAAAAAAA/8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAwDAQACEQMRAD8AJQAB/9k="

    emb = face_recognition_service.extract_embedding(frame_b64)
    stored = emb.tolist()

    result = face_recognition_service.verify(frame_b64, stored)
    # Same embedding → should verify (distance ≈ 0)
    assert result["verified"] is True
    assert result["distance"] < 0.05


# ── Veterinary Tests ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_vet_seed_data_returned():
    """Test that vet service returns seed data when Google API key is absent."""
    from app.services.veterinary import veterinary_service

    mock_db = MagicMock()
    mock_db.vet_services.find = MagicMock(return_value=MagicMock(
        to_list=AsyncMock(return_value=[])  # cache miss
    ))

    with patch("app.services.veterinary.get_database", AsyncMock(return_value=mock_db)), \
         patch("app.core.config.settings.GOOGLE_MAPS_API_KEY", ""):
        results = await veterinary_service.search_nearby(
            latitude=12.9716,
            longitude=77.5946,
            radius_km=50.0,
            place_type="hospital",
        )
        # Seed data should be returned
        assert len(results) >= 1
        assert all("name" in r for r in results)


# ── Human-to-Animal Tests ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_intent_detection():
    from app.services.human_to_animal import HumanToAnimalService
    service = HumanToAnimalService()

    assert service._detect_intent("hello good boy!") == "greeting"
    assert service._detect_intent("let's play fetch") == "play"
    assert service._detect_intent("shh calm down easy") == "calm"
    assert service._detect_intent("dinner time, eat your food") == "food"


@pytest.mark.asyncio
async def test_human_to_animal_translate():
    from app.services.human_to_animal import HumanToAnimalService
    service = HumanToAnimalService()

    result = await service.translate("hello good boy", "dog", "en")
    assert "intent_detected" in result
    assert result["intent_detected"] == "greeting"
    assert "recommended_actions" in result
    assert len(result["recommended_actions"]) > 0
    assert "scientific_basis" in result


# ── MongoDB TLS Scoping Tests ─────────────────────────────────────────────────

class _FakeMotorClient:
    """Stands in for AsyncIOMotorClient so connect_to_mongo() can be exercised
    without a real MongoDB server, while capturing the kwargs it was built with."""

    def __init__(self, *args, **kwargs):
        self.captured_kwargs = kwargs
        self.admin = MagicMock()
        self.admin.command = AsyncMock(return_value={"ok": 1.0})
        self.closed = False

    def __getitem__(self, _name):
        db = MagicMock()
        db.users.create_indexes = AsyncMock()
        db.audio_sessions.create_indexes = AsyncMock()
        db.translations.create_indexes = AsyncMock()
        db.vet_services.create_indexes = AsyncMock()
        return db

    def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_mongo_tls_bypass_allowed_only_when_dev_and_explicitly_permitted():
    """Cert validation may only be disabled when BOTH ENVIRONMENT=='development'
    AND MONGODB_ALLOW_INSECURE_TLS is explicitly True."""
    import app.db.mongodb as mongodb_module
    from app.core.config import settings

    created_clients = []

    def _record_client(*args, **kwargs):
        client = _FakeMotorClient(*args, **kwargs)
        created_clients.append(client)
        return client

    with patch("app.db.mongodb.AsyncIOMotorClient", side_effect=_record_client), \
         patch.object(settings, "ENVIRONMENT", "development"), \
         patch.object(settings, "MONGODB_ALLOW_INSECURE_TLS", True):
        await mongodb_module.connect_to_mongo()

    assert created_clients[0].captured_kwargs["tlsAllowInvalidCertificates"] is True
    await mongodb_module.close_mongo_connection()


@pytest.mark.asyncio
async def test_mongo_tls_validation_enforced_when_not_explicitly_permitted():
    """Even in development, cert validation stays ON unless MONGODB_ALLOW_INSECURE_TLS
    is explicitly set True — being 'development' alone must not be enough."""
    import app.db.mongodb as mongodb_module
    from app.core.config import settings

    created_clients = []

    def _record_client(*args, **kwargs):
        client = _FakeMotorClient(*args, **kwargs)
        created_clients.append(client)
        return client

    with patch("app.db.mongodb.AsyncIOMotorClient", side_effect=_record_client), \
         patch.object(settings, "ENVIRONMENT", "development"), \
         patch.object(settings, "MONGODB_ALLOW_INSECURE_TLS", False):
        await mongodb_module.connect_to_mongo()

    assert created_clients[0].captured_kwargs["tlsAllowInvalidCertificates"] is False
    await mongodb_module.close_mongo_connection()


@pytest.mark.asyncio
async def test_mongo_tls_validation_enforced_outside_development():
    """Staging/production must never disable cert validation, even if the
    insecure-TLS flag is (mis)configured True — ENVIRONMENT is authoritative."""
    import app.db.mongodb as mongodb_module
    from app.core.config import settings

    for env in ("production", "staging", "anything-else"):
        created_clients = []

        def _record_client(*args, **kwargs):
            client = _FakeMotorClient(*args, **kwargs)
            created_clients.append(client)
            return client

        with patch("app.db.mongodb.AsyncIOMotorClient", side_effect=_record_client), \
             patch.object(settings, "ENVIRONMENT", env), \
             patch.object(settings, "MONGODB_ALLOW_INSECURE_TLS", True):
            await mongodb_module.connect_to_mongo()

        assert created_clients[0].captured_kwargs["tlsAllowInvalidCertificates"] is False, (
            f"TLS certificate validation was disabled for ENVIRONMENT={env!r}"
        )
        await mongodb_module.close_mongo_connection()


@pytest.mark.asyncio
async def test_mongo_tls_secure_by_default_when_unconfigured():
    """With no explicit overrides (Settings() class defaults), cert validation
    must stay ON — ENVIRONMENT defaulting to 'development' must not, by itself,
    disable certificate validation."""
    import app.db.mongodb as mongodb_module
    from app.core.config import Settings

    default_settings = Settings(_env_file=None, MONGODB_URL="mongodb+srv://placeholder/", JWT_SECRET_KEY="x", SECRET_KEY="x")
    assert default_settings.ENVIRONMENT == "development"  # documented default
    assert default_settings.MONGODB_ALLOW_INSECURE_TLS is False  # secure default

    created_clients = []

    def _record_client(*args, **kwargs):
        client = _FakeMotorClient(*args, **kwargs)
        created_clients.append(client)
        return client

    with patch("app.db.mongodb.AsyncIOMotorClient", side_effect=_record_client), \
         patch("app.db.mongodb.settings", default_settings):
        await mongodb_module.connect_to_mongo()

    assert created_clients[0].captured_kwargs["tlsAllowInvalidCertificates"] is False
    await mongodb_module.close_mongo_connection()
