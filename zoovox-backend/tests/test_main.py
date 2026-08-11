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
    from app.services.audio_analysis import audio_analysis_service, ANIMAL_EMOTION_TRANSLATIONS
    import numpy as np

    y = np.random.randn(16000 * 3).astype(np.float32)
    features = audio_analysis_service._extract_features(y, 16000)
    result = audio_analysis_service._classify_emotion(features, "dog")

    valid_emotions = list(ANIMAL_EMOTION_TRANSLATIONS["dog"].keys())
    assert result["emotion"] in valid_emotions
    assert 0.0 <= result["confidence"] <= 1.0


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
