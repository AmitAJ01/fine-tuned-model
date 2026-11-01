"""System health and hardware diagnostics router."""

from __future__ import annotations

import os
from fastapi import APIRouter
from backend.app.core.config import settings
from scripts.check_env import check_pytorch_cuda, get_disk_storage, get_nvidia_smi, get_system_ram

router = APIRouter(tags=["Health & Hardware"])


@router.get("/health", summary="Service Health Check")
async def health_check():
    """Verify backend availability and configuration."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "base_model": settings.BASE_MODEL_NAME,
        "finetuned_adapter_path": settings.FINETUNED_ADAPTER_PATH,
    }


@router.get("/hardware", summary="Hardware Diagnostics")
async def hardware_status():
    """Return live system memory, GPU VRAM, and storage availability."""
    return {
        "ram": get_system_ram(),
        "storage": get_disk_storage(),
        "gpu": get_nvidia_smi(),
        "pytorch": check_pytorch_cuda(),
    }
