#!/usr/bin/env python3
"""
Seed MongoDB with realistic development data.
Run: python scripts/seed_db.py
"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timezone, timedelta
import random
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings
from app.core.security import hash_password

SAMPLE_USERS = [
    {"name": "Ashith Shankar", "email": "ashith@zoovox.app"},
    {"name": "Demo User",       "email": "demo@zoovox.app"},
    {"name": "Test Researcher", "email": "research@zoovox.app"},
]

ANIMALS = ["dog", "cat", "bird", "horse", "cow"]
EMOTIONS = ["happy", "fearful", "aggressive", "hungry", "playful", "lonely", "alert"]

TRANSLATIONS = {
    "dog":   ["I'm happy and excited!", "I'm scared, comfort me.", "I'm hungry!", "Let's play!"],
    "cat":   ["I'm content.", "Don't touch me.", "Feed me now.", "I want to play."],
    "bird":  ["Environment is safe.", "Danger nearby!", "I need food.", "I'm singing!"],
    "horse": ["I'm calm.", "Something scared me.", "My feeding is overdue.", "Let me run!"],
    "cow":   ["Good nutrition today.", "I'm stressed.", "I'm hungry.", "Social bonds strong."],
}


async def seed():
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.MONGODB_DB_NAME]

    print("🌱 Seeding MongoDB...")

    # Clear existing dev data
    await db.users.delete_many({"email": {"$in": [u["email"] for u in SAMPLE_USERS]}})

    # Insert users
    user_ids = []
    for u in SAMPLE_USERS:
        doc = {
            **u,
            "email": u["email"].lower(),
            "password_hash": hash_password("ZooVox@123"),
            "is_active": True,
            "is_banned": False,
            "face_embedding": None,
            "plan": random.choice(["free", "pro"]),
            "animals_analyzed": 0,
            "created_at": datetime.now(timezone.utc) - timedelta(days=random.randint(1, 90)),
        }
        res = await db.users.insert_one(doc)
        user_ids.append(res.inserted_id)
        print(f"  ✅ User: {u['email']}")

    # Insert 50 sample audio sessions
    sessions = []
    for i in range(50):
        animal = random.choice(ANIMALS)
        emotion = random.choice(EMOTIONS)
        user_id = str(random.choice(user_ids))
        sessions.append({
            "session_id": f"seed-session-{i:04d}",
            "user_id": user_id,
            "direction": random.choice(["animal_to_human", "human_to_animal"]),
            "animal_type": animal,
            "detected_emotion": emotion,
            "animal_confidence": round(random.uniform(0.55, 0.95), 4),
            "emotion_confidence": round(random.uniform(0.50, 0.90), 4),
            "translation_en": random.choice(TRANSLATIONS[animal]),
            "audio_duration_sec": round(random.uniform(1.0, 15.0), 2),
            "processing_time_ms": round(random.uniform(120, 800), 1),
            "created_at": datetime.now(timezone.utc) - timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23)
            ),
        })

    await db.audio_sessions.insert_many(sessions)
    print(f"  ✅ Inserted {len(sessions)} audio sessions")

    # Update user stats
    for uid in user_ids:
        count = await db.audio_sessions.count_documents({"user_id": str(uid)})
        await db.users.update_one({"_id": uid}, {"$set": {"animals_analyzed": count}})

    # Insert Bengaluru vet services with geospatial index
    vet_services = [
        {
            "google_place_id": "SEED_BLR_001",
            "name": "Cessna Lifeline Veterinary Hospital",
            "type": "hospital",
            "address": "Nagarbhavi 2nd Stage, Bengaluru 560072",
            "latitude": 12.9719, "longitude": 77.5082,
            "location": {"type": "Point", "coordinates": [77.5082, 12.9719]},
            "phone": "+91 80 2328 7777",
            "website": "https://www.cessnalifeline.com",
            "rating": 4.6, "open_now": True, "hours": "24/7",
            "cached_at": datetime.now(timezone.utc).timestamp(),
        },
        {
            "google_place_id": "SEED_BLR_002",
            "name": "Bangalore Pet Hospital",
            "type": "hospital",
            "address": "Indiranagar, Bengaluru 560038",
            "latitude": 12.9784, "longitude": 77.6408,
            "location": {"type": "Point", "coordinates": [77.6408, 12.9784]},
            "phone": "+91 80 4112 0000",
            "rating": 4.4, "open_now": True, "hours": "8 AM – 10 PM",
            "cached_at": datetime.now(timezone.utc).timestamp(),
        },
        {
            "google_place_id": "SEED_BLR_003",
            "name": "Heads Up For Tails — Koramangala",
            "type": "store",
            "address": "Forum Mall, Koramangala, Bengaluru",
            "latitude": 12.9363, "longitude": 77.6101,
            "location": {"type": "Point", "coordinates": [77.6101, 12.9363]},
            "rating": 4.3, "open_now": True, "hours": "10 AM – 9 PM",
            "cached_at": datetime.now(timezone.utc).timestamp(),
        },
    ]
    for vet in vet_services:
        await db.vet_services.update_one(
            {"google_place_id": vet["google_place_id"]},
            {"$set": vet},
            upsert=True,
        )
    print(f"  ✅ Inserted {len(vet_services)} vet services (Bengaluru)")

    print("\n🎉 Seed complete!")
    print(f"   Login: demo@zoovox.app / ZooVox@123")
    client.close()


if __name__ == "__main__":
    asyncio.run(seed())
