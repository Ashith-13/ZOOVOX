"""
Audio Translation Endpoints
  POST   /api/v1/audio/analyze        — Upload audio file → analysis + translation
  POST   /api/v1/audio/human-to-animal — Text input → animal communication cue
  GET    /api/v1/audio/history        — User's translation history
  GET    /api/v1/audio/session/{id}   — Single session detail
  WS     /api/v1/audio/stream         — Real-time WebSocket streaming analysis
  GET    /api/v1/audio/supported-animals — List supported species
"""

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import (
    APIRouter, Depends, File, Form, HTTPException,
    UploadFile, WebSocket, WebSocketDisconnect, status
)
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.security import get_current_active_user
from app.db.mongodb import get_database
from app.schemas.schemas import (
    AudioAnalysisResponse, HumanToAnimalRequest, HumanToAnimalResponse,
    TranslationHistoryItem,
)
from app.services.audio_analysis import (
    AnimalClassificationUnavailable, audio_analysis_service, compute_audio_hash,
)
from app.services.human_to_animal import human_to_animal_service

router = APIRouter(prefix="/audio", tags=["Audio Translation"])
logger = logging.getLogger(__name__)

MAX_AUDIO_BYTES = settings.AUDIO_MAX_FILE_MB * 1024 * 1024
SUPPORTED_MIME = {
    "audio/webm", "audio/ogg", "audio/wav", "audio/mp3",
    "audio/mpeg", "audio/mp4", "audio/x-m4a",
}

SUPPORTED_ANIMALS = [
    {"id": "dog",      "name": "Dog",      "emoji": "🐕", "emotions": 8, "research": "Bradshaw & Rooney (2016)"},
    {"id": "cat",      "name": "Cat",      "emoji": "🐈", "emotions": 8, "research": "Turner & Bateson (2000)"},
    {"id": "bird",     "name": "Bird",     "emoji": "🐦", "emotions": 8, "research": "Marler (1955); Kahl et al. (2021)"},
    {"id": "horse",    "name": "Horse",    "emoji": "🐎", "emotions": 8, "research": "Briefer & McElligott (2011)"},
    {"id": "cow",      "name": "Cow",      "emoji": "🐄", "emotions": 8, "research": "Manteuffel et al. (2004)"},
    {"id": "elephant", "name": "Elephant", "emoji": "🐘", "emotions": 6, "research": "Poole et al. (1988)"},
    {"id": "pig",      "name": "Pig",      "emoji": "🐷", "emotions": 6, "research": "Weary & Fraser (1995)"},
    {"id": "dolphin",  "name": "Dolphin",  "emoji": "🐬", "emotions": 6, "research": "Herzing (2010)"},
]


# ── Analyze Audio (cache-first) ────────────────────────────────────────────────

async def _get_or_compute_analysis(
    db,
    audio_bytes: bytes,
    content_type: str,
    language: str,
) -> dict:
    """
    Cache-first wrapper around audio_analysis_service.analyze_audio(), shared
    by the REST endpoint and the WebSocket "end" path so both get identical
    caching behavior.

    Cache key: the existing audio_hash (unique-indexed in db.translations).
    Cache validity: the stored pipeline_version must match
    audio_analysis_service.current_pipeline_version() — invalidates stale
    results automatically when the deployed model changes, without any
    manual cache-flush step.

    Cache failures (read or write) never block or crash the primary
    analysis — any error while touching the cache is logged and the request
    proceeds as if the cache didn't exist. If db is None (MongoDB
    unavailable), the cache is skipped entirely and behavior is identical to
    before this feature existed.
    """
    t0 = time.time()
    await audio_analysis_service.load_models()
    audio_hash = compute_audio_hash(audio_bytes)
    current_version = audio_analysis_service.current_pipeline_version()

    if db is not None:
        cached = None
        try:
            cached = await db.translations.find_one({"audio_hash": audio_hash})
        except Exception as e:
            logger.warning(f"Translation cache read failed, proceeding without cache: {e}")

        if cached and cached.get("pipeline_version") == current_version:
            logger.info(f"Translation cache hit (audio_hash={audio_hash})")
            result = dict(cached["result"])
            result["processing_time_ms"] = round((time.time() - t0) * 1000, 1)
            return result

    analysis = await audio_analysis_service.analyze_audio(
        audio_bytes=audio_bytes,
        content_type=content_type,
        user_language=language,
    )

    if db is not None:
        try:
            await db.translations.update_one(
                {"audio_hash": audio_hash},
                {"$set": {
                    "audio_hash": audio_hash,
                    "pipeline_version": current_version,
                    "result": analysis,
                    "created_at": datetime.now(timezone.utc),
                }},
                upsert=True,
            )
        except Exception as e:
            logger.warning(f"Translation cache write failed (non-fatal): {e}")

    return analysis


@router.post("/analyze", response_model=AudioAnalysisResponse)
async def analyze_audio(
    audio: UploadFile = File(..., description="Animal audio recording (WebM/WAV/MP3)"),
    language: str = Form("en", description="Target language for translation (ISO 639-1)"),
    current_user: dict = Depends(get_current_active_user),
):
    """
    Main animal sound analysis endpoint.

    Pipeline:
      1. Validate file type & size
      2. YAMNet feature extraction + animal classification
      3. Emotion/intent detection (AnimalSpeak framework)
      4. Translation lookup
      5. Persist session to MongoDB
      6. Return full analysis response
    """
    # ── Validation ────────────────────────────────────────────────────────
    content_type = (audio.content_type or "").split(";", 1)[0].strip().lower()
    if content_type not in SUPPORTED_MIME:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio type: {audio.content_type or 'unknown'}. Supported: {', '.join(SUPPORTED_MIME)}",
        )

    audio_bytes = await audio.read()
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file too large. Maximum {settings.AUDIO_MAX_FILE_MB} MB.",
        )
    if len(audio_bytes) < 1024:
        raise HTTPException(status_code=400, detail="Audio file too short or empty")

    # ── ML Analysis (cache-first) ────────────────────────────────────────────
    db = await get_database()
    try:
        analysis = await _get_or_compute_analysis(
            db=db,
            audio_bytes=audio_bytes,
            content_type=content_type,
            language=language,
        )
    except AnimalClassificationUnavailable as e:
        logger.warning(f"Animal classification unavailable: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Audio analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Audio analysis service error")

    # ── Persist to MongoDB ────────────────────────────────────────────────
    session_id = str(uuid.uuid4())
    session_doc = {
        "session_id": session_id,
        "user_id": str(current_user["_id"]),
        "direction": "animal_to_human",
        "animal_type": analysis["animal_type"],
        "detected_emotion": analysis["detected_emotion"],
        "animal_confidence": analysis["animal_confidence"],
        "emotion_confidence": analysis["emotion_confidence"],
        "translation_en": analysis["translation_en"],
        "audio_duration_sec": analysis["audio_duration_sec"],
        "processing_time_ms": analysis["processing_time_ms"],
        "audio_hash": analysis["audio_hash"],
        "language": language,
        "created_at": datetime.now(timezone.utc),
    }
    if db is not None:
        await db.audio_sessions.insert_one(session_doc)

    # Increment user stats
    if db is not None:
        await db.users.update_one(
            {"_id": current_user["_id"]},
            {"$inc": {"animals_analyzed": 1}},
        )

    return AudioAnalysisResponse(
        session_id=session_id,
        **{k: v for k, v in analysis.items() if k != "audio_hash"},
    )


# ── Human → Animal ────────────────────────────────────────────────────────────

@router.post("/human-to-animal", response_model=HumanToAnimalResponse)
async def human_to_animal(
    payload: HumanToAnimalRequest,
    current_user: dict = Depends(get_current_active_user),
):
    """
    Translate human speech/text into animal-appropriate communication cues.
    Returns synthesized audio URL + recommended behavioral actions.
    """
    try:
        result = await human_to_animal_service.translate(
            text=payload.text,
            target_animal=payload.target_animal,
            language=payload.language,
        )
    except Exception as e:
        logger.error(f"H2A translation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Translation service error")

    session_id = str(uuid.uuid4())
    db = await get_database()
    await db.audio_sessions.insert_one({
        "session_id": session_id,
        "user_id": str(current_user["_id"]),
        "direction": "human_to_animal",
        "animal_type": payload.target_animal,
        "original_text": payload.text,
        "intent_detected": result.get("intent_detected"),
        "animal_cue": result["animal_cue_description"],
        "created_at": datetime.now(timezone.utc),
    })

    return HumanToAnimalResponse(
        session_id=session_id,
        original_text=payload.text,
        animal_cue_description=result["animal_cue_description"],
        audio_url=result["audio_url"],
        recommended_actions=result["recommended_actions"],
        scientific_basis=result["scientific_basis"],
    )


# ── History ───────────────────────────────────────────────────────────────────

@router.get("/history")
async def get_history(
    page: int = 1,
    limit: int = 20,
    animal: Optional[str] = None,
    current_user: dict = Depends(get_current_active_user),
):
    """Paginated translation history for the current user."""
    db = await get_database()
    if db is None:
        raise HTTPException(
            status_code=503,
            detail="Translation history is temporarily unavailable. Please try again shortly.",
        )

    query = {"user_id": str(current_user["_id"])}
    if animal:
        query["animal_type"] = animal

    skip = (page - 1) * limit
    total = await db.audio_sessions.count_documents(query)
    sessions = await db.audio_sessions.find(query).sort(
        "created_at", -1
    ).skip(skip).limit(limit).to_list(length=limit)

    items = []
    for s in sessions:
        items.append({
            "session_id": s["session_id"],
            "direction": s.get("direction", "animal_to_human"),
            "animal_type": s.get("animal_type", "unknown"),
            "translation": s.get("translation_en") or s.get("animal_cue", ""),
            "confidence": s.get("animal_confidence", 0.0),
            "created_at": s["created_at"].isoformat(),
        })

    return {
        "items": items,
        "total": total,
        "page": page,
        "pages": (total + limit - 1) // limit,
    }


# ── Session Detail ────────────────────────────────────────────────────────────

@router.get("/session/{session_id}")
async def get_session(
    session_id: str,
    current_user: dict = Depends(get_current_active_user),
):
    db = await get_database()
    if db is None:
        raise HTTPException(
            status_code=503,
            detail="Session history is temporarily unavailable. Please try again shortly.",
        )

    session = await db.audio_sessions.find_one({
        "session_id": session_id,
        "user_id": str(current_user["_id"]),
    })
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    session["_id"] = str(session["_id"])
    session["created_at"] = session["created_at"].isoformat()
    return session


# ── Supported Animals ─────────────────────────────────────────────────────────

@router.get("/supported-animals")
async def supported_animals():
    return {"animals": SUPPORTED_ANIMALS}


# ── WebSocket Real-time Stream ────────────────────────────────────────────────

@router.websocket("/stream")
async def audio_stream_ws(websocket: WebSocket):
    """
    Authenticated real-time recording stream.

    The browser sends `auth` first, then MediaRecorder blobs as binary frames,
    followed by `end`. JSON/base64 `audio_chunk` messages are supported for
    backwards compatibility. The final response is persisted exactly like
    POST /audio/analyze and includes its `session_id`.
    """
    await websocket.accept()
    user = None
    audio_buffer = bytearray()
    content_type = "audio/webm"
    language = "en"
    last_interim_size = 0

    try:
        import base64
        from app.core.security import decode_token
        from bson import ObjectId

        while True:
            frame = await websocket.receive()
            if frame.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(code=status.WS_1000_NORMAL_CLOSURE)

            # React sends MediaRecorder slices as binary WebSocket frames. A
            # JSON/base64 payload remains available for non-browser clients.
            if frame.get("bytes") is not None:
                msg = {"type": "audio_chunk"}
                chunk_bytes = frame["bytes"]
            else:
                try:
                    msg = json.loads(frame.get("text") or "{}")
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "message": "Invalid WebSocket message"})
                    continue
                chunk_bytes = None

            msg_type = msg.get("type")

            if msg_type == "auth":
                try:
                    payload = decode_token(msg["token"])
                    if payload.get("type") != "access":
                        raise ValueError("Access token required")
                    db = await get_database()
                    if db is None:
                        # Fail closed: never authenticate against a
                        # fabricated user. This is a distinct, honest
                        # outcome from "invalid token" — the database is
                        # unavailable, so identity cannot be verified at all.
                        await websocket.send_json({
                            "type": "error",
                            "message": "Live audio authentication is temporarily unavailable. Please try again shortly.",
                        })
                        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
                        return
                    user = await db.users.find_one({"_id": ObjectId(payload["sub"])})
                    if user:
                        requested_type = (msg.get("content_type") or "audio/webm").split(";", 1)[0].strip().lower()
                        if requested_type not in SUPPORTED_MIME:
                            await websocket.send_json({"type": "error", "message": "Unsupported audio format"})
                            await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
                            return
                        content_type = requested_type
                        language = str(msg.get("language") or "en")[:10]
                        await websocket.send_json({"type": "auth_ok", "user": user["name"]})
                    else:
                        await websocket.send_json({"type": "error", "message": "Invalid user"})
                        await websocket.close()
                        return
                except Exception:
                    await websocket.send_json({"type": "error", "message": "Auth failed"})
                    await websocket.close()
                    return

            elif msg_type == "audio_chunk":
                if not user:
                    await websocket.send_json({"type": "error", "message": "Not authenticated"})
                    continue
                if chunk_bytes is None:
                    try:
                        chunk_bytes = base64.b64decode(msg.get("data", ""), validate=True)
                    except (ValueError, TypeError):
                        await websocket.send_json({"type": "error", "message": "Invalid base64 audio chunk"})
                        continue

                if not chunk_bytes:
                    continue
                if len(audio_buffer) + len(chunk_bytes) > MAX_AUDIO_BYTES:
                    await websocket.send_json({"type": "error", "message": f"Audio exceeds {settings.AUDIO_MAX_FILE_MB} MB limit"})
                    await websocket.close(code=status.WS_1009_MESSAGE_TOO_BIG)
                    return

                audio_buffer.extend(chunk_bytes)

                # MediaRecorder chunks are compressed containers, not raw
                # 16-bit PCM. Analyze the complete recording so the decoder
                # receives a valid WebM/OGG stream.
                if len(audio_buffer) - last_interim_size >= 24 * 1024:
                    last_interim_size = len(audio_buffer)
                    try:
                        interim = await audio_analysis_service.analyze_audio(
                            audio_bytes=bytes(audio_buffer),
                            content_type=content_type,
                            user_language=language,
                        )
                        await websocket.send_json({
                            "type": "interim",
                            "animal_type": interim["animal_type"],
                            "confidence": interim["animal_confidence"],
                            "emotion": interim["detected_emotion"],
                        })
                    except Exception:
                        pass

            elif msg_type == "end":
                if not user or not audio_buffer:
                    await websocket.send_json({"type": "error", "message": "No audio received"})
                    await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                    return

                try:
                    db = await get_database()
                    final = await _get_or_compute_analysis(
                        db=db,
                        audio_bytes=bytes(audio_buffer),
                        content_type=content_type,
                        language=language,
                    )
                    session_id = str(uuid.uuid4())
                    if db is not None:
                        await db.audio_sessions.insert_one({
                            "session_id": session_id,
                            "user_id": str(user["_id"]),
                            "direction": "animal_to_human",
                            "animal_type": final["animal_type"],
                            "detected_emotion": final["detected_emotion"],
                            "animal_confidence": final["animal_confidence"],
                            "emotion_confidence": final["emotion_confidence"],
                            "translation_en": final["translation_en"],
                            "audio_duration_sec": final["audio_duration_sec"],
                            "processing_time_ms": final["processing_time_ms"],
                            "audio_hash": final["audio_hash"],
                            "language": language,
                            "created_at": datetime.now(timezone.utc),
                        })
                        await db.users.update_one(
                            {"_id": user["_id"]},
                            {"$inc": {"animals_analyzed": 1}},
                        )

                    await websocket.send_json({
                        "type": "final",
                        "session_id": session_id,
                        **{key: value for key, value in final.items() if key not in {"audio_hash", "features_snapshot"}},
                    })
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": str(e)})
                await websocket.close()
                return

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": "Server error"})
        except Exception:
            pass
