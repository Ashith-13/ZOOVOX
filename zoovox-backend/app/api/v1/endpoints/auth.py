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
from app.services.face_recognition import (
    face_recognition_service,
    EmbeddingEncryptionUnavailable,
    EmbeddingDecryptionError,
    FaceRecognitionUnavailable,
)

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

    Liveness is checked as a data-quality signal (reject an obviously
    spoofed enrollment frame) but, unlike face/login, enrollment does not
    fail closed when the anti-spoofing model is merely unavailable — the
    caller already holds a valid JWT to reach this endpoint, so this isn't
    an authentication-bypass surface the way face/login is.
    """
    liveness = face_recognition_service.check_liveness(payload.frame_b64)
    if liveness["status"] == "spoof_detected":
        raise HTTPException(
            status_code=422,
            detail="This looks like a photo or screen, not a live camera. Please try again using your camera directly.",
        )

    # Fail closed: never fabricate an embedding when the real face
    # recognition model is unavailable — enrollment must be rejected, not
    # silently stored against a non-biometric placeholder.
    try:
        embedding = face_recognition_service.enroll([payload.frame_b64])
    except FaceRecognitionUnavailable:
        raise HTTPException(
            status_code=503,
            detail="Face enrollment is temporarily unavailable. Please try again later.",
        )
    if embedding is None:
        raise HTTPException(
            status_code=422,
            detail="No face detected in the provided frame. Ensure good lighting and face visibility.",
        )

    # Fail closed: never store a new embedding in plaintext. If encryption
    # isn't configured, enrollment is unavailable rather than degrading.
    try:
        encrypted_embedding = face_recognition_service.encrypt_embedding(embedding)
    except EmbeddingEncryptionUnavailable:
        raise HTTPException(
            status_code=503,
            detail="Face enrollment is temporarily unavailable. Please try again later.",
        )

    db = await get_database()
    if db is None:
        raise HTTPException(status_code=503, detail="Face enrollment unavailable in demo mode (requires DB)")

    await db.users.update_one(
        {"_id": current_user["_id"]},
        {"$set": {"face_embedding": encrypted_embedding, "face_enrolled_at": datetime.now(timezone.utc)}},
    )
    return {"success": True, "message": "Face enrolled successfully"}


# ── Face Login ────────────────────────────────────────────────────────────────

@router.post("/face/login", response_model=TokenResponse)
async def face_login(payload: FaceLoginRequest):
    """
    Face login is convenience-tier authentication, not a full security
    boundary — password login remains the authoritative path and is
    unaffected by anything below.

    Step 0: liveness/anti-spoofing check — deliberately separate from and
            gating identity verification. Fails closed (503) if the
            anti-spoofing model itself is unavailable; identity verification
            never runs on a frame that hasn't passed this check.
    Step 1: extract embedding from frame ONCE (identity only, no liveness
            signal) — never re-extracted per enrolled user below.
    Step 2: scan all enrolled users, comparing the same live embedding via
            compare_embeddings() (production: use ANN index like FAISS)
    Step 3: return best match if distance < threshold
    """
    db = await get_database()

    liveness = face_recognition_service.check_liveness(payload.frame_b64)
    if liveness["status"] == "unavailable":
        raise HTTPException(
            status_code=503,
            detail="Face login is temporarily unavailable. Please sign in with your password.",
        )
    if liveness["status"] == "no_face_detected":
        raise HTTPException(status_code=422, detail="No face detected in frame")
    if liveness["status"] == "spoof_detected":
        raise HTTPException(
            status_code=401,
            detail="Face login failed. Please use a live camera and try again, or sign in with your password.",
        )
    # liveness["status"] == "live" — proceed to identity verification.

    live_emb = face_recognition_service.extract_embedding(payload.frame_b64)
    if live_emb is None:
        raise HTTPException(status_code=422, detail="No face detected in frame")

    if db is None:
        raise HTTPException(status_code=503, detail="Face login unavailable in demo mode (requires DB)")

    # Load all enrolled users (production: replace with FAISS ANN search)
    enrolled_users = await db.users.find(
        {"face_embedding": {"$ne": None}}
    ).to_list(length=1000)

    best_user = None
    best_result = {"distance": 1.0, "verified": False}
    encryption_unavailable_encountered = False

    for user in enrolled_users:
        try:
            stored_embedding = face_recognition_service.decrypt_embedding(user["face_embedding"])
        except EmbeddingEncryptionUnavailable:
            # Key missing/invalid — this record genuinely can't be checked,
            # distinct from "checked and it didn't match".
            encryption_unavailable_encountered = True
            continue
        except EmbeddingDecryptionError:
            logger.warning(f"Skipping unreadable face embedding for user {user['_id']}")
            continue

        # Compare the already-extracted live embedding (computed once,
        # above) against each stored embedding — never re-run face
        # detection/ArcFace inference per enrolled user.
        result = face_recognition_service.compare_embeddings(live_emb, stored_embedding)
        if result["verified"] and result["distance"] < best_result["distance"]:
            best_user = user
            best_result = result

    if not best_user:
        if encryption_unavailable_encountered:
            raise HTTPException(
                status_code=503,
                detail="Face login is temporarily unavailable. Please sign in with your password.",
            )
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
    if db is None:
        # Fail closed: never fabricate a user or issue a token when the
        # database is unreachable. Credentials/identity cannot be verified
        # without it.
        raise HTTPException(
            status_code=503,
            detail="Token refresh is temporarily unavailable. Please try again shortly.",
        )
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
