"""
Audio Translation Endpoints
  POST   /api/v1/audio/analyze        — Upload audio file → analysis + translation
  POST   /api/v1/audio/human-to-animal — Text input → animal communication cue
  GET    /api/v1/audio/history        — User's translation history
  GET    /api/v1/audio/session/{id}   — Single session detail
  WS     /api/v1/audio/stream         — Real-time WebSocket streaming analysis
  GET    /api/v1/audio/supported-animals — List supported species
"""

import logging
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
from app.services.audio_analysis import audio_analysis_service
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


# ── Analyze Audio ─────────────────────────────────────────────────────────────

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
    if audio.content_type not in SUPPORTED_MIME:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio type: {audio.content_type}. Supported: {', '.join(SUPPORTED_MIME)}",
        )

    audio_bytes = await audio.read()
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file too large. Maximum {settings.AUDIO_MAX_FILE_MB} MB.",
        )
    if len(audio_bytes) < 1024:
        raise HTTPException(status_code=400, detail="Audio file too short or empty")

    # ── ML Analysis ───────────────────────────────────────────────────────
    try:
        analysis = await audio_analysis_service.analyze_audio(
            audio_bytes=audio_bytes,
            content_type=audio.content_type,
            user_language=language,
        )
    except Exception as e:
        logger.error(f"Audio analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Audio analysis service error")

    # ── Persist to MongoDB ────────────────────────────────────────────────
    session_id = str(uuid.uuid4())
    db = await get_database()
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
    await db.audio_sessions.insert_one(session_doc)

    # Increment user stats
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
    Real-time audio streaming analysis via WebSocket.

    Protocol (client → server):
      { "type": "auth", "token": "<JWT>" }
      { "type": "audio_chunk", "data": "<base64 PCM chunk>", "animal_hint": "dog" }
      { "type": "end" }

    Protocol (server → client):
      { "type": "interim", "animal_type": "...", "confidence": 0.7 }
      { "type": "final", ...full AudioAnalysisResponse... }
      { "type": "error", "message": "..." }
    """
    await websocket.accept()
    user = None
    audio_buffer = b""

    try:
        import base64
        from app.core.security import decode_token
        from bson import ObjectId

        while True:
            msg = await websocket.receive_json()
            msg_type = msg.get("type")

            if msg_type == "auth":
                try:
                    payload = decode_token(msg["token"])
                    db = await get_database()
                    user = await db.users.find_one({"_id": ObjectId(payload["sub"])})
                    if user:
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
                chunk_b64 = msg.get("data", "")
                chunk_bytes = base64.b64decode(chunk_b64)
                audio_buffer += chunk_bytes

                # Send interim analysis every ~1 second of audio
                if len(audio_buffer) > 32000:  # ~2s at 16kHz mono int16
                    try:
                        interim = await audio_analysis_service.analyze_audio(
                            audio_bytes=audio_buffer[-32000:],
                            content_type="audio/wav",
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
                    break

                try:
                    final = await audio_analysis_service.analyze_audio(
                        audio_bytes=audio_buffer,
                        content_type="audio/wav",
                    )
                    await websocket.send_json({"type": "final", **final})
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": str(e)})
                break

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": "Server error"})
        except Exception:
            pass
