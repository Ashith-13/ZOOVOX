from fastapi import APIRouter
from app.api.v1.endpoints.auth import router as auth_router
from app.api.v1.endpoints.audio import router as audio_router
from app.api.v1.endpoints.vet_analytics import vet_router, analytics_router

api_router = APIRouter()
api_router.include_router(auth_router)
api_router.include_router(audio_router)
api_router.include_router(vet_router)
api_router.include_router(analytics_router)
