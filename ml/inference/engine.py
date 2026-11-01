"""Production-grade LLM inference engine for DevOps incident diagnosis.

Supports:
1. Baseline configuration: Pretrained base model with zero-shot domain prompt.
2. Fine-Tuned configuration: Same base model merged or layered with QLoRA PEFT adapters.
3. Fallback dry-run mode for local development without active model weights.
"""

from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .prompts import (
    SYSTEM_PROMPT,
    build_diagnosis_prompt,
    extract_json_payload,
    extract_json_payload_with_diagnostics,
)

logger = logging.getLogger(__name__)


class IncidentInferenceEngine:
    """Unified inference engine supporting both baseline and fine-tuned QLoRA models."""

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        finetuned_adapter_path: Optional[str] = "models/finetuned/qwen2.5-coder-1.5b-incident-diagnosis",
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
    ) -> None:
        self.base_model_name = base_model_name
        self.finetuned_adapter_path = finetuned_adapter_path
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.mock_mode = (
            os.environ.get("USE_MOCK_INFERENCE") == "1"
            or os.environ.get("FINETUNEOPS_MOCK") == "1"
        )

        # Lazy initialized models
        self.tokenizer = None
        self.baseline_model = None
        self.finetuned_model = None

    def _ensure_tokenizer(self) -> Any:
        """Load tokenizer lazily."""
        if self.tokenizer is None:
            try:
                from transformers import AutoTokenizer

                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.base_model_name,
                        trust_remote_code=True,
                        local_files_only=True,
                    )
                except Exception:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.base_model_name,
                        trust_remote_code=True,
                    )
            except ImportError:
                logger.warning("transformers not installed. Tokenizer unavailable.")
            except Exception as e:
                logger.error(f"Error loading tokenizer {self.base_model_name}: {e}")
        return self.tokenizer

    def _load_baseline_model(self) -> Any:
        """Load the baseline pretrained model."""
        if self.baseline_model is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM

                dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
                logger.info(f"Loading baseline model: {self.base_model_name} on {self.device_map}")
                try:
                    self.baseline_model = AutoModelForCausalLM.from_pretrained(
                        self.base_model_name,
                        device_map=self.device_map,
                        torch_dtype=dtype,
                        trust_remote_code=True,
                        local_files_only=True,
                    )
                except Exception:
                    self.baseline_model = AutoModelForCausalLM.from_pretrained(
                        self.base_model_name,
                        device_map=self.device_map,
                        torch_dtype=dtype,
                        trust_remote_code=True,
                    )
            except Exception as e:
                logger.error(f"Failed to load baseline model: {e}")
                raise
        return self.baseline_model

    def _load_finetuned_model(self) -> Any:
        """Load base model layered with fine-tuned PEFT LoRA adapter."""
        if self.finetuned_model is None:
            if not self.finetuned_adapter_path:
                raise ValueError("No finetuned_adapter_path specified.")

            adapter_path = Path(self.finetuned_adapter_path)
            has_weights = (
                (adapter_path / "adapter_model.safetensors").exists()
                or (adapter_path / "adapter_model.bin").exists()
            )
            if not adapter_path.exists() or not has_weights:
                msg = (
                    f"Fine-tuned adapter weights (adapter_model.safetensors) not found at '{self.finetuned_adapter_path}'. "
                    "For full inference, please download the adapter checkpoint into this directory, "
                    "or set the FINETUNED_ADAPTER_PATH environment variable to a custom path. "
                    "To run lightweight demo inference without local weights, set mock=True or use baseline mode."
                )
                logger.error(msg)
                raise FileNotFoundError(msg)

            try:
                from peft import PeftModel

                base = self._load_baseline_model()
                logger.info(f"Attaching LoRA adapter from {self.finetuned_adapter_path}")
                try:
                    self.finetuned_model = PeftModel.from_pretrained(
                        base,
                        self.finetuned_adapter_path,
                        local_files_only=True,
                    )
                except Exception:
                    self.finetuned_model = PeftModel.from_pretrained(
                        base,
                        self.finetuned_adapter_path,
                    )
            except Exception as e:
                logger.error(f"Failed to attach fine-tuned adapter: {e}")
                raise
        return self.finetuned_model

    def generate_diagnosis(
        self,
        log_text: str,
        system_context: Optional[str] = None,
        use_finetuned: bool = True,
        temperature: float = 0.1,
        max_new_tokens: int = 768,
        mock: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Execute diagnosis inference and return structured report."""
        start_time = time.perf_counter()
        user_prompt = build_diagnosis_prompt(log_text, system_context)
        is_mock = mock if mock is not None else self.mock_mode

        # Attempt real LLM generation if transformers & model are available and not in mock mode
        try:
            if not is_mock:
                tokenizer = self._ensure_tokenizer()
                model = self._load_finetuned_model() if use_finetuned else self._load_baseline_model()
            else:
                tokenizer = None
                model = None

            if tokenizer and model:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
                formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer([formatted], return_tensors="pt").to(model.device)

                import torch
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=(temperature > 0.0),
                        pad_token_id=tokenizer.eos_token_id,
                    )
                new_tokens = output_ids[0][inputs.input_ids.shape[1] :]
                raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                tokens_count = len(new_tokens)
            else:
                raw_text, tokens_count = self._simulate_diagnosis(log_text, use_finetuned)
        except Exception as e:
            if not is_mock:
                logger.error(f"Real model inference failed: {e}")
                elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
                return {
                    "report": None,
                    "model_used": (
                        f"{self.base_model_name} + LoRA ({Path(self.finetuned_adapter_path).name})"
                        if use_finetuned and self.finetuned_adapter_path
                        else self.base_model_name
                    ),
                    "is_fine_tuned": use_finetuned,
                    "latency_ms": elapsed_ms,
                    "raw_response": f"Inference failure: {str(e)}",
                    "tokens_generated": 0,
                    "json_valid": False,
                    "error_detail": f"Real model inference failed: {str(e)}",
                }
            logger.info(f"Model generation fell back to simulation/mock mode ({e})")
            raw_text, tokens_count = self._simulate_diagnosis(log_text, use_finetuned)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        parsed_json, is_valid, error_msg = extract_json_payload_with_diagnostics(raw_text)
        error_detail = None
        report = None

        if not is_valid or parsed_json is None:
            is_valid = False
            error_detail = error_msg or "Failed to parse structured JSON from model output."
        else:
            # Validate required schema fields without silent fallback fabrication
            req_keys = [
                "incident_category",
                "severity_level",
                "root_cause_analysis",
                "contributing_factors",
                "recommended_debugging_steps",
                "suggested_remediation_actions",
            ]
            missing_keys = [k for k in req_keys if k not in parsed_json]
            if missing_keys:
                is_valid = False
                error_detail = f"Schema validation error: Missing required fields: {missing_keys}"
            else:
                report = parsed_json

        return {
            "report": report,
            "model_used": (
                f"{self.base_model_name} + LoRA ({Path(self.finetuned_adapter_path).name})"
                if use_finetuned and self.finetuned_adapter_path
                else self.base_model_name
            ),
            "is_fine_tuned": use_finetuned,
            "latency_ms": elapsed_ms,
            "raw_response": raw_text,
            "tokens_generated": tokens_count,
            "json_valid": is_valid,
            "error_detail": error_detail,
        }

    def _simulate_diagnosis(self, log_text: str, is_finetuned: bool) -> Tuple[str, int]:
        """Deterministic simulation for local testing prior to weight downloads."""
        import json

        lower = log_text.lower()
        if "oom" in lower or "out of memory" in lower or "killed process" in lower:
            category = "Infrastructure"
            severity = "SEV-1" if "critical" in lower or "production" in lower else "SEV-2"
            root_cause = "Linux OOM-killer terminated the container due to memory cgroup limits exceeded."
            factors = ["High concurrent query traffic", "Absence of JVM/Node heap limits inside cgroup"]
            steps = ["dmesg -T | grep -i oom", "kubectl top pod", "Check memory leak in heap profile"]
            remediation = ["Increase pod memory request/limit", "Tune application max heap size"]
        elif "connection refused" in lower or "pg_hba" in lower or "postgres" in lower or "timeout" in lower:
            category = "Database"
            severity = "SEV-2"
            root_cause = "Database connection pool exhaustion and connection timeout to primary PostgreSQL cluster."
            factors = ["Connection leak in backend worker threads", "Max client connections limit reached"]
            steps = ["SELECT count(*) FROM pg_stat_activity;", "Check PgBouncer queue depth"]
            remediation = ["Enable PgBouncer connection pooling", "Set idle connection timeout"]
        else:
            category = "Application"
            severity = "SEV-3"
            root_cause = "Unhandled runtime exception during request serialization."
            factors = ["Missing schema validation on external webhook payload"]
            steps = ["Check application error logs with request trace-id", "Inspect input payload schema"]
            remediation = ["Implement strict Pydantic payload validation", "Deploy patch release"]

        simulated_data = {
            "incident_category": category,
            "severity_level": severity,
            "root_cause_analysis": root_cause,
            "contributing_factors": factors,
            "recommended_debugging_steps": steps,
            "suggested_remediation_actions": remediation,
        }

        # Fine-tuned model outputs clean JSON directly; baseline without domain tuning
        # sometimes includes conversational chatter if temperature fluctuates
        if not is_finetuned:
            raw = f"Here is my diagnosis of the incident:\n```json\n{json.dumps(simulated_data, indent=2)}\n```\nLet me know if you need further help."
        else:
            raw = json.dumps(simulated_data, indent=2)

        return raw, 220
