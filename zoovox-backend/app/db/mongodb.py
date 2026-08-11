"""
MongoDB Atlas — async motor client
Collections:
  users           — auth, face embeddings, profile
  audio_sessions  — recording metadata + analysis results
  translations    — cached translation pairs
  vet_services    — nearby vet index (geospatial)
  analytics       — usage telemetry
"""

import logging
import certifi

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import IndexModel, ASCENDING, GEOSPHERE, TEXT
from app.core.config import settings

logger = logging.getLogger(__name__)

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


async def connect_to_mongo():
    global _client, _db

    _client = AsyncIOMotorClient(
        settings.MONGODB_URL,

        # TLS / SSL Fix for macOS + MongoDB Atlas
        tls=True,
        tlsCAFile=certifi.where(),
        tlsAllowInvalidCertificates=True,

        # Connection Settings
        retryWrites=True,
        serverSelectionTimeoutMS=30000,
        connectTimeoutMS=30000,
        socketTimeoutMS=30000,

        # Pool Settings
        maxPoolSize=20,
        minPoolSize=5,
    )

    _db = _client[settings.MONGODB_DB_NAME]

    # Test connection immediately
    await _client.admin.command("ping")

    await _ensure_indexes(_db)

    logger.info(f"Connected to MongoDB: {settings.MONGODB_DB_NAME}")


async def close_mongo_connection():
    global _client

    if _client:
        _client.close()
        logger.info("MongoDB connection closed")


async def get_database() -> AsyncIOMotorDatabase:
    if _db is None:
        raise RuntimeError(
            "Database not initialised — call connect_to_mongo() first"
        )

    return _db


async def _ensure_indexes(db: AsyncIOMotorDatabase):
    """Create all required indexes on startup."""

    # users
    await db.users.create_indexes([
        IndexModel(
            [("email", ASCENDING)],
            unique=True,
            name="unique_email"
        ),

        IndexModel(
            [("username", ASCENDING)],
            unique=True,
            sparse=True,
            name="unique_username"
        ),

        IndexModel(
            [("created_at", ASCENDING)],
            name="user_created"
        ),
    ])

    # audio_sessions
    await db.audio_sessions.create_indexes([
        IndexModel(
            [("user_id", ASCENDING)],
            name="session_user"
        ),

        IndexModel(
            [("created_at", ASCENDING)],
            name="session_created"
        ),

        IndexModel(
            [("animal_type", ASCENDING)],
            name="session_animal"
        ),
    ])

    # translations — cache with TTL (7 days)
    await db.translations.create_indexes([
        IndexModel(
            [("audio_hash", ASCENDING)],
            unique=True,
            name="audio_hash_unique"
        ),

        IndexModel(
            [("created_at", ASCENDING)],
            expireAfterSeconds=604800,
            name="ttl_7days"
        ),
    ])

    # vet_services — geolocation
    await db.vet_services.create_indexes([
        IndexModel(
            [("location", GEOSPHERE)],
            name="geo_location"
        ),

        IndexModel(
            [("name", TEXT)],
            name="text_search"
        ),
    ])

    logger.info("MongoDB indexes verified ✅")