"""
Veterinary & Analytics Endpoints
  GET /api/v1/vet/search              — Nearby vet services (geolocation)
  GET /api/v1/vet/pet-care-tips       — AI-generated pet care by animal
  GET /api/v1/analytics/dashboard     — User analytics summary
  GET /api/v1/analytics/leaderboard   — Global usage leaderboard (anonymized)
"""

import logging
from fastapi import APIRouter, Depends, Query, HTTPException
from app.core.security import get_current_active_user
from app.db.mongodb import get_database
from app.schemas.schemas import VetSearchRequest, VetSearchResponse
from app.services.veterinary import veterinary_service

router = APIRouter(tags=["Veterinary & Analytics"])
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Veterinary
# ─────────────────────────────────────────────────────────────────────────────

vet_router = APIRouter(prefix="/vet")


@vet_router.get("/search", response_model=VetSearchResponse)
async def search_vet_services(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    radius_km: float = Query(10.0, ge=0.1, le=100.0),
    type: str = Query("hospital", regex="^(hospital|clinic|pharmacy|store)$"),
    current_user: dict = Depends(get_current_active_user),
):
    """
    Find nearby veterinary services using Google Places API (geospatial).
    Results cached in MongoDB Atlas for 24 hours to reduce API costs.
    """
    results = await veterinary_service.search_nearby(
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km,
        place_type=type,
    )
    return VetSearchResponse(results=results, total=len(results), radius_km=radius_km)


@vet_router.get("/pet-care-tips/{animal}")
async def pet_care_tips(
    animal: str,
    current_user: dict = Depends(get_current_active_user),
):
    """Curated pet care tips sourced from veterinary literature."""
    tips_db = {
        "dog": {
            "nutrition": [
                "Feed 2× daily for adult dogs; puppies need 3–4× (AAFCO guidelines 2023)",
                "Ensure clean water is always available — dogs need ~1 oz water per pound body weight",
                "Avoid chocolate, xylitol, grapes, onions — all acutely toxic (ASPCA toxin list)",
            ],
            "exercise": [
                "Adult dogs need 30–60 minutes of exercise daily (breed-dependent)",
                "Mental stimulation (puzzle toys, training) reduces destructive behavior",
                "Avoid exercise 1h before/after meals to prevent bloat (GDV risk — Glickman et al., 1994)",
            ],
            "veterinary": [
                "Annual wellness exam + boosters; bi-annual for seniors (>7 years)",
                "Monthly heartworm prevention if in endemic area (AHS recommendation)",
                "Dental cleaning every 1–3 years under anesthesia to prevent periodontal disease",
            ],
            "behavioral": [
                "Socialization window: 3–14 weeks critical for healthy development (Scott & Fuller, 1965)",
                "Positive reinforcement training shown superior to aversive methods (Hiby et al., 2004)",
                "Separation anxiety affects 17% of dogs — graduated desensitization is effective",
            ],
        },
        "cat": {
            "nutrition": [
                "Cats are obligate carnivores — taurine and arachidonic acid essential in diet",
                "Wet food recommended for urinary health — dry food can cause chronic dehydration",
                "Avoid dog food — lacks essential feline nutrients (AAFCO cat nutrient profiles)",
            ],
            "exercise": [
                "Cats need 15–20 min of active play daily to prevent obesity",
                "Interactive wand toys stimulate natural hunting behavior",
                "Vertical space (cat trees) reduces stress and increases environmental enrichment",
            ],
            "veterinary": [
                "Annual wellness exam; biannual for cats >10 years",
                "FeLV/FIV testing recommended for outdoor/new cats (AAFP 2020 guidelines)",
                "Spay/neuter at 5–6 months unless medical reason to delay",
            ],
            "behavioral": [
                "Litter box rule: N+1 boxes for N cats (Indoor Pet Initiative, OSU)",
                "Scratching is normal — provide appropriate surfaces to redirect",
                "Cats communicate primarily through body language and scent marking",
            ],
        },
        "bird": {
            "nutrition": [
                "Seed-only diets are nutritionally deficient — supplement with pellets and fresh produce",
                "Toxic foods: avocado, caffeine, alcohol, high-salt foods, onions",
                "Calcium and vitamin A are commonly deficient — leafy greens and cuttlebone help",
            ],
            "environment": [
                "Cage should be wider than tall — birds fly horizontally",
                "Avoid Teflon cookware near birds — PTFE fumes acutely fatal at 200°C+",
                "Natural light or full-spectrum UV lamp supports circadian health",
            ],
            "veterinary": [
                "Annual avian vet check — birds hide illness (prey species survival instinct)",
                "Wing/nail trim by vet or trained groomer every 2–3 months",
                "Quarantine new birds 30 days before introducing to existing flock",
            ],
        },
    }
    animal_tips = tips_db.get(animal.lower())
    if not animal_tips:
        raise HTTPException(status_code=404, detail=f"Care tips for '{animal}' not available yet")
    return {"animal": animal, "tips": animal_tips}


# ─────────────────────────────────────────────────────────────────────────────
# Analytics
# ─────────────────────────────────────────────────────────────────────────────

analytics_router = APIRouter(prefix="/analytics")


@analytics_router.get("/dashboard")
async def dashboard_analytics(
    current_user: dict = Depends(get_current_active_user),
):
    """User's personal analytics dashboard."""
    db = await get_database()
    user_id = str(current_user["_id"])

    # Animal breakdown
    pipeline_animal = [
        {"$match": {"user_id": user_id}},
        {"$group": {"_id": "$animal_type", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    animal_agg = await db.audio_sessions.aggregate(pipeline_animal).to_list(length=20)

    # Emotion breakdown
    pipeline_emotion = [
        {"$match": {"user_id": user_id}},
        {"$group": {"_id": "$detected_emotion", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    emotion_agg = await db.audio_sessions.aggregate(pipeline_emotion).to_list(length=20)

    # Average confidence
    pipeline_conf = [
        {"$match": {"user_id": user_id, "animal_confidence": {"$exists": True}}},
        {"$group": {"_id": None, "avg": {"$avg": "$animal_confidence"}}},
    ]
    conf_agg = await db.audio_sessions.aggregate(pipeline_conf).to_list(length=1)

    # Weekly trend (last 7 days)
    from datetime import timedelta, datetime, timezone
    week_ago = datetime.now(timezone.utc) - timedelta(days=7)
    pipeline_weekly = [
        {"$match": {"user_id": user_id, "created_at": {"$gte": week_ago}}},
        {"$group": {
            "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}},
            "count": {"$sum": 1},
        }},
        {"$sort": {"_id": 1}},
    ]
    weekly_agg = await db.audio_sessions.aggregate(pipeline_weekly).to_list(length=7)

    total = await db.audio_sessions.count_documents({"user_id": user_id})

    return {
        "total_sessions": total,
        "animals_analyzed": current_user.get("animals_analyzed", 0),
        "animal_breakdown": {a["_id"]: a["count"] for a in animal_agg if a["_id"]},
        "emotion_breakdown": {e["_id"]: e["count"] for e in emotion_agg if e["_id"]},
        "avg_confidence": round(conf_agg[0]["avg"], 4) if conf_agg else 0.0,
        "top_animals": [a["_id"] for a in animal_agg[:3] if a["_id"]],
        "weekly_trend": [{"date": w["_id"], "count": w["count"]} for w in weekly_agg],
        "plan": current_user.get("plan", "free"),
    }


@analytics_router.get("/leaderboard")
async def leaderboard():
    """Anonymized global leaderboard — top contributors."""
    db = await get_database()
    pipeline = [
        {"$group": {"_id": "$user_id", "sessions": {"$sum": 1}}},
        {"$sort": {"sessions": -1}},
        {"$limit": 10},
        {"$lookup": {
            "from": "users",
            "localField": "_id",
            "foreignField": "_id",
            "as": "user_info",
        }},
        {"$project": {
            "sessions": 1,
            "display_name": {"$ifNull": [
                {"$arrayElemAt": ["$user_info.name", 0]}, "Anonymous"
            ]},
        }},
    ]
    from bson import ObjectId
    results = await db.audio_sessions.aggregate(pipeline).to_list(length=10)
    return {
        "leaderboard": [
            {
                "rank": i + 1,
                "name": r.get("display_name", "Anonymous")[:2] + "***",  # Anonymize
                "sessions": r["sessions"],
            }
            for i, r in enumerate(results)
        ]
    }
