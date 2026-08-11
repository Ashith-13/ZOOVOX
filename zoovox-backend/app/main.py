"""
ZOOVOX – Speak Beyond Species
Production-Grade FastAPI Backend
Author: Senior ML/Backend Architecture
Research Basis: YAMNet (Howard et al., 2019), VGGish (Hershey et al., 2017),
               ESC-50 Dataset (Piczak, 2015), AnimalSpeak (Ofer & Netzer, 2023)
"""
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import time
import logging
import uuid

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.api.v1.router import api_router
from app.db.mongodb import connect_to_mongo, close_mongo_connection
from app.core.rate_limiter import RateLimitMiddleware

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ZOOVOX API",
    description="""
    ## ZOOVOX – Speak Beyond Species
    
    AI-powered animal-human communication platform.
    
    ### Research Foundation
    - **YAMNet** (Howard et al., 2019) — Audio event classification
    - **VGGish** (Hershey et al., 2017) — Audio embedding generation
    - **ESC-50** Dataset (Piczak, 2015) — Environmental sound classification
    - **AnimalSpeak** (Ofer & Netzer, 2023) — Animal vocalization semantics
    - **BirdNET** (Kahl et al., 2021) — Bird sound identification
    
    ### Key Features
    - Real-time animal sound classification & translation
    - Two-way animal ↔ human communication
    - Face recognition login (DeepFace + MongoDB face embeddings)
    - Veterinary geolocation services
    - WebSocket-based live audio streaming
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ── Middleware ──────────────────────────────────────────────────────────────
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware)

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
    logger.info(f"[{request_id}] {request.method} {request.url.path} → {response.status_code} ({process_time:.1f}ms)")
    return response

# ── Lifecycle ───────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    logger.info("🚀 ZOOVOX Backend starting up...")
    await connect_to_mongo()
    logger.info("✅ MongoDB Atlas connected")

@app.on_event("shutdown")
async def shutdown():
    await close_mongo_connection()
    logger.info("👋 ZOOVOX Backend shut down cleanly")

# ── Routers ─────────────────────────────────────────────────────────────────
app.include_router(api_router, prefix="/api/v1")

@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "ZOOVOX API",
        "version": "1.0.0",
        "status": "operational",
        "tagline": "Speak Beyond Species",
    }

@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy", "timestamp": time.time()}
