"""FastAPI application entrypoint for FineTuneOps."""

from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.core.config import settings
from backend.app.routes import evaluation_router, health_router, incident_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("finetuneops")

app = FastAPI(
    title="FineTuneOps API",
    description="Domain-specialized LLM fine-tuning platform for DevOps incident diagnosis.",
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for Next.js frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach API routers
app.include_router(health_router, prefix=settings.API_PREFIX)
app.include_router(incident_router, prefix=settings.API_PREFIX)
app.include_router(evaluation_router, prefix=settings.API_PREFIX)


@app.on_event("startup")
async def startup_event() -> None:
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    logger.info(f"Base model configured: {settings.BASE_MODEL_NAME}")
    logger.info(f"Fine-tuned adapter target: {settings.FINETUNED_ADAPTER_PATH}")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    logger.info(f"Shutting down {settings.PROJECT_NAME}")


@app.get("/", tags=["Root"])
async def root() -> dict[str, str]:
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "api_v1": settings.API_PREFIX,
    }
