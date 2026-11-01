"""Application configuration and core settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict
import yaml
from pydantic import BaseModel, Field


def find_project_root() -> Path:
    """Resolve the project root directory whether running from root or backend/."""
    current = Path.cwd().resolve()
    for parent in [current] + list(current.parents):
        if (parent / "ml" / "config" / "base_config.yaml").exists():
            return parent
    return current


def resolve_default_device() -> str:
    """Detect CUDA GPU availability unless forced to CPU."""
    if os.environ.get("FORCE_CPU") == "1":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def resolve_adapter_path() -> str:
    """Prefer explicit env var, then Task 7 completion-masked adapter if present."""
    if os.environ.get("FINETUNED_ADAPTER_PATH"):
        return os.environ["FINETUNED_ADAPTER_PATH"]
    root = find_project_root()
    if (root / "models" / "finetuned_v2" / "adapter" / "best").exists():
        return "models/finetuned_v2/adapter/best"
    return "models/finetuned/adapter/best"


class Settings(BaseModel):
    """FineTuneOps runtime settings."""
    PROJECT_NAME: str = "FineTuneOps"
    VERSION: str = "0.1.0"
    API_PREFIX: str = "/api/v1"
    PROJECT_ROOT: Path = Field(default_factory=find_project_root)

    # Model Configuration
    BASE_MODEL_NAME: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    FINETUNED_ADAPTER_PATH: str = Field(default_factory=resolve_adapter_path)
    DEVICE: str = Field(default_factory=resolve_default_device)

    # CORS
    BACKEND_CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
    ]

    def load_base_yaml(self) -> Dict[str, Any]:
        """Read base_config.yaml if present."""
        cfg_path = self.PROJECT_ROOT / "ml" / "config" / "base_config.yaml"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}


settings = Settings()
