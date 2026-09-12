"""
Authentication Endpoints
  POST /api/v1/auth/register          — email + password signup
  POST /api/v1/auth/login             — email + password → JWT pair
  POST /api/v1/auth/face/enroll       — store face embedding (requires JWT)
  POST /api/v1/auth/face/login        — face-only login
  POST /api/v1/auth/refresh           — refresh JWT pair
  POST /api/v1/auth/logout            — revoke refresh token
  GET  /api/v1/auth/me                — current user profile
  DELETE /api/v1/auth/face/delete     — GDPR: remove face embedding
"""

import logging
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Depends, status, Body
from bson import ObjectId

from app.db.mongodb import get_database
from app.core.security import (
    hash_password, verify_password,
    create_access_token, create_refresh_token, decode_token,
    get_current_active_user,
)
from app.schemas.schemas import (
    UserRegisterRequest, UserLoginRequest,
    FaceLoginRequest, FaceEnrollRequest,
    TokenResponse, RefreshRequest, UserPublicProfile,
)
from app.services.face_recognition import face_recognition_service

router = APIRouter(prefix="/auth", tags=["Authentication"])
logger = logging.getLogger(__name__)


def _serialize_user(user: dict) -> UserPublicProfile:
    return UserPublicProfile(
        id=str(user["_id"]),
        name=user["name"],
        email=user["email"],
        avatar_url=user.get("avatar_url"),
        face_enrolled=bool(user.get("face_embedding")),
        plan=user.get("plan", "free"),
        animals_analyzed=user.get("animals_analyzed", 0),
        created_at=user.get("created_at", datetime.now(timezone.utc)),
    )


def _make_token_response(user: dict) -> dict:
    user_id = str(user["_id"])
    access = create_access_token({"sub": user_id, "email": user["email"]})
    refresh = create_refresh_token(user_id)
    return {
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "bearer",
        "expires_in": 3600,
        "user": _serialize_user(user),
    }


# ── Register ──────────────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(payload: UserRegisterRequest):
    db = await get_database()

    # Fail closed: never fabricate a user or issue a token when the database
    # is unreachable. Credentials cannot be verified without it.
    if db is None:
        logger.error("Registration rejected: database unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Registration is temporarily unavailable. Please try again shortly.",
        )

    if await db.users.find_one({"email": payload.email}):
        raise HTTPException(status_code=409, detail="Email already registered")

    user_doc = {
        "name": payload.name,
        "email": payload.email.lower(),
        "password_hash": hash_password(payload.password),
        "is_active": True,
        "is_banned": False,
        "face_embedding": None,
        "plan": "free",
        "animals_analyzed": 0,
        "avatar_url": None,
        "created_at": datetime.now(timezone.utc),
        "last_login": None,
    }
    result = await db.users.insert_one(user_doc)
    user_doc["_id"] = result.inserted_id

    logger.info(f"New user registered: {payload.email}")
    return _make_token_response(user_doc)


# ── Login ─────────────────────────────────────────────────────────────────────

@router.post("/login", response_model=TokenResponse)
async def login(payload: UserLoginRequest):
    db = await get_database()

    # Fail closed: never fabricate a user or issue a token when the database
    # is unreachable. Credentials cannot be verified without it.
    if db is None:
        logger.error("Login rejected: database unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Login is temporarily unavailable. Please try again shortly.",
        )

    user = await db.users.find_one({"email": payload.email.lower()})

    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    if user.get("is_banned"):
        raise HTTPException(status_code=403, detail="Account suspended")

    await db.users.update_one(
        {"_id": user["_id"]},
        {"$set": {"last_login": datetime.now(timezone.utc)}},
    )
    logger.info(f"User login: {payload.email}")
    return _make_token_response(user)


# ── Face Enroll ───────────────────────────────────────────────────────────────

@router.post("/face/enroll", status_code=200)
async def enroll_face(
    payload: FaceEnrollRequest,
    current_user: dict = Depends(get_current_active_user),
):
    """
    Store face embedding for authenticated user.
    Accepts a single base64 frame; production clients should send 3+ frames
    for better accuracy (averaged by the service).
    """
    embedding = face_recognition_service.enroll([payload.frame_b64])
    if embedding is None:
        raise HTTPException(
            status_code=422,
            detail="No face detected in the provided frame. Ensure good lighting and face visibility.",
        )

    db = await get_database()
    await db.users.update_one(
        {"_id": current_user["_id"]},
        {"$set": {"face_embedding": embedding, "face_enrolled_at": datetime.now(timezone.utc)}},
    )
    return {"success": True, "message": "Face enrolled successfully"}


# ── Face Login ────────────────────────────────────────────────────────────────

@router.post("/face/login", response_model=TokenResponse)
async def face_login(payload: FaceLoginRequest):
    """
    Step 1: extract embedding from frame
    Step 2: scan all enrolled users (production: use ANN index like FAISS)
    Step 3: return best match if distance < threshold
    """
    db = await get_database()

    live_emb = face_recognition_service.extract_embedding(payload.frame_b64)
    if live_emb is None:
        raise HTTPException(status_code=422, detail="No face detected in frame")

    # Load all enrolled users (production: replace with FAISS ANN search)
    enrolled_users = await db.users.find(
        {"face_embedding": {"$ne": None}}
    ).to_list(length=1000)

    best_user = None
    best_result = {"distance": 1.0, "verified": False}

    for user in enrolled_users:
        result = face_recognition_service.verify(payload.frame_b64, user["face_embedding"])
        if result["verified"] and result["distance"] < best_result["distance"]:
            best_user = user
            best_result = result

    if not best_user:
        raise HTTPException(
            status_code=401,
            detail="Face not recognized. Please use email/password login.",
        )

    await db.users.update_one(
        {"_id": best_user["_id"]},
        {"$set": {"last_login": datetime.now(timezone.utc)}},
    )
    logger.info(f"Face login success: {best_user['email']} (distance={best_result['distance']})")
    return _make_token_response(best_user)


# ── Refresh Token ─────────────────────────────────────────────────────────────

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(payload: RefreshRequest):
    decoded = decode_token(payload.refresh_token)
    if decoded.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    db = await get_database()
    user = await db.users.find_one({"_id": ObjectId(decoded["sub"])})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return _make_token_response(user)


# ── Me ────────────────────────────────────────────────────────────────────────

@router.get("/me", response_model=UserPublicProfile)
async def get_me(current_user: dict = Depends(get_current_active_user)):
    return _serialize_user(current_user)


# ── Delete Face Data (GDPR) ───────────────────────────────────────────────────

@router.delete("/face/delete", status_code=200)
async def delete_face_data(current_user: dict = Depends(get_current_active_user)):
    db = await get_database()
    await db.users.update_one(
        {"_id": current_user["_id"]},
        {"$set": {"face_embedding": None, "face_enrolled_at": None}},
    )
    return {"success": True, "message": "Face data deleted per GDPR request"}
