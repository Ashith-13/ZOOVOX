"""
Veterinary Services — Real-world Geolocation
════════════════════════════════════════════════════════════════════
Uses Google Places API (Nearby Search + Place Details)
with MongoDB Atlas geospatial index as a write-through cache.

Flow:
  1. Check MongoDB for recent cache (< 24h) within requested radius
  2. If cache miss → call Google Places API
  3. Persist results to MongoDB with 24h TTL
  4. Return merged & deduplicated results
"""

import logging
import time
import httpx
from typing import List, Optional
from app.core.config import settings
from app.db.mongodb import get_database

logger = logging.getLogger(__name__)

GOOGLE_PLACES_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
GOOGLE_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"
CACHE_TTL_SECONDS = 86400   # 24 hours


VET_KEYWORD_MAP = {
    "hospital": "veterinary hospital",
    "clinic":   "animal clinic",
    "pharmacy": "pet pharmacy",
    "store":    "pet store",
}

# Seed data for offline/test mode (Bengaluru-specific)
SEED_VET_DATA = [
    {
        "name": "Cessna Lifeline Veterinary Hospital",
        "type": "hospital",
        "address": "2nd Stage, Nagarbhavi, Bengaluru, Karnataka 560072",
        "latitude": 12.9719, "longitude": 77.5082,
        "phone": "+91 80 2328 7777",
        "website": "https://www.cessnalifeline.com",
        "rating": 4.6,
        "open_now": True,
        "hours": "24/7",
        "google_place_id": "SEED_001",
    },
    {
        "name": "Bangalore Pet Hospital",
        "type": "hospital",
        "address": "Indiranagar, Bengaluru, Karnataka 560038",
        "latitude": 12.9784, "longitude": 77.6408,
        "phone": "+91 80 4112 0000",
        "rating": 4.4,
        "open_now": True,
        "hours": "8 AM – 10 PM",
        "google_place_id": "SEED_002",
    },
    {
        "name": "PetSutra Veterinary Clinic",
        "type": "clinic",
        "address": "Koramangala, Bengaluru, Karnataka 560034",
        "latitude": 12.9352, "longitude": 77.6245,
        "phone": "+91 80 4170 3636",
        "rating": 4.5,
        "open_now": False,
        "hours": "9 AM – 8 PM",
        "google_place_id": "SEED_003",
    },
    {
        "name": "Heads Up For Tails",
        "type": "store",
        "address": "Forum Mall, Koramangala, Bengaluru",
        "latitude": 12.9363, "longitude": 77.6101,
        "phone": "+91 80 4090 3030",
        "rating": 4.3,
        "open_now": True,
        "hours": "10 AM – 9 PM",
        "google_place_id": "SEED_004",
    },
]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance in km between two geo points."""
    import math
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class VeterinaryService:

    async def search_nearby(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 10.0,
        place_type: str = "hospital",
    ) -> List[dict]:
        """
        Main entry: cache-first, then Google Places, then seed data.
        """
        db = await get_database()
        radius_m = int(radius_km * 1000)

        # ── 1. MongoDB geospatial cache lookup ────────────────────────────
        cached = await db.vet_services.find({
            "location": {
                "$near": {
                    "$geometry": {"type": "Point", "coordinates": [longitude, latitude]},
                    "$maxDistance": radius_m,
                }
            },
            "type": place_type,
            "cached_at": {"$gt": time.time() - CACHE_TTL_SECONDS},
        }).to_list(length=20)

        if cached:
            logger.info(f"VetService: cache hit ({len(cached)} results)")
            return [self._format_result(r, latitude, longitude) for r in cached]

        # ── 2. Google Places API ──────────────────────────────────────────
        if settings.GOOGLE_MAPS_API_KEY:
            try:
                results = await self._fetch_google_places(
                    latitude, longitude, radius_m, place_type
                )
                if results:
                    await self._cache_results(db, results, place_type)
                    return [self._format_result(r, latitude, longitude) for r in results]
            except Exception as e:
                logger.error(f"Google Places API error: {e}")

        # ── 3. Seed data fallback ─────────────────────────────────────────
        logger.info("VetService: using seed data fallback")
        seed = [s for s in SEED_VET_DATA if s.get("type") == place_type]
        for s in seed:
            s["distance_km"] = _haversine_km(latitude, longitude, s["latitude"], s["longitude"])
        seed.sort(key=lambda x: x["distance_km"])
        return [s for s in seed if s["distance_km"] <= radius_km]

    async def _fetch_google_places(
        self,
        lat: float,
        lng: float,
        radius_m: int,
        place_type: str,
    ) -> List[dict]:
        keyword = VET_KEYWORD_MAP.get(place_type, "veterinary")
        params = {
            "location": f"{lat},{lng}",
            "radius": radius_m,
            "keyword": keyword,
            "key": settings.GOOGLE_MAPS_API_KEY,
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(GOOGLE_PLACES_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        results = []
        for place in data.get("results", [])[:15]:
            geo = place.get("geometry", {}).get("location", {})
            results.append({
                "google_place_id": place.get("place_id"),
                "name": place.get("name"),
                "address": place.get("vicinity"),
                "latitude": geo.get("lat"),
                "longitude": geo.get("lng"),
                "rating": place.get("rating"),
                "open_now": place.get("opening_hours", {}).get("open_now"),
                "location": {
                    "type": "Point",
                    "coordinates": [geo.get("lng"), geo.get("lat")],
                },
                "cached_at": time.time(),
            })
        return results

    async def _cache_results(self, db, results: List[dict], place_type: str):
        for r in results:
            r["type"] = place_type
            await db.vet_services.update_one(
                {"google_place_id": r["google_place_id"]},
                {"$set": r},
                upsert=True,
            )

    def _format_result(self, r: dict, user_lat: float, user_lng: float) -> dict:
        dist = _haversine_km(user_lat, user_lng, r.get("latitude", 0), r.get("longitude", 0))
        return {
            "id": str(r.get("_id", r.get("google_place_id", ""))),
            "name": r.get("name", ""),
            "type": r.get("type", "hospital"),
            "address": r.get("address", ""),
            "distance_km": round(dist, 2),
            "latitude": r.get("latitude"),
            "longitude": r.get("longitude"),
            "phone": r.get("phone"),
            "website": r.get("website"),
            "rating": r.get("rating"),
            "open_now": r.get("open_now"),
            "hours": r.get("hours"),
            "google_place_id": r.get("google_place_id"),
        }


veterinary_service = VeterinaryService()
