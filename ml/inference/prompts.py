"""Prompt engineering and structured format templates for DevOps Incident Diagnosis."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple


SYSTEM_PROMPT = """You are an expert DevOps and Site Reliability Engineering (SRE) incident diagnostician.
Analyze the provided system logs, stack traces, or incident description and output a rigorous diagnostic report.

You MUST respond ONLY with a single valid JSON object following this exact schema:
{
  "incident_category": "Infrastructure" | "Database" | "Network" | "Application" | "Security" | "CI/CD Pipeline" | "Kubernetes / Container Orchestration",
  "severity_level": "SEV-1" | "SEV-2" | "SEV-3" | "SEV-4",
  "root_cause_analysis": "Comprehensive explanation of the root cause identified from the logs",
  "contributing_factors": [
    "Contributing environmental or architectural factor 1",
    "Contributing factor 2"
  ],
  "recommended_debugging_steps": [
    "Concrete command, metric check, or log query to inspect",
    "Next diagnostic verification step"
  ],
  "suggested_remediation_actions": [
    "Immediate mitigation action to restore service",
    "Long-term fix or preventive measure"
  ]
}

Do not include any conversational preamble, explanations, markdown fences, or postscripts outside the JSON object.
"""


def build_diagnosis_prompt(log_text: str, system_context: Optional[str] = None) -> str:
    """Format user prompt combining incident logs and optional architecture context."""
    context_block = f"System Context:\n{system_context.strip()}\n\n" if system_context else ""
    return (
        f"{context_block}"
        f"Incident Logs / Description:\n"
        f"```\n{log_text.strip()}\n```\n\n"
        f"Provide your structured incident diagnosis in the requested JSON format."
    )


def extract_json_payload_with_diagnostics(raw_text: str) -> Tuple[Optional[Dict[str, Any]], bool, Optional[str]]:
    """Parse raw LLM generation and extract JSON object with diagnostic error details."""
    if not raw_text or not raw_text.strip():
        return None, False, "Empty generation response."

    cleaned = raw_text.strip()

    # Case 1: Markdown code fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1)), True, None
        except json.JSONDecodeError as e:
            return None, False, f"JSONDecodeError inside code fence: {e}"

    # Case 2: Direct JSON object match
    direct_match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if direct_match:
        try:
            return json.loads(direct_match.group(1)), True, None
        except json.JSONDecodeError as e:
            if "Unterminated string" in str(e) or "Expecting" in str(e):
                return None, False, f"JSONDecodeError (likely truncated generation): {e}"
            return None, False, f"JSONDecodeError: {e}"

    # Case 3: Direct parsing
    try:
        return json.loads(cleaned), True, None
    except json.JSONDecodeError as e:
        if "Unterminated string" in str(e) or "Expecting" in str(e):
            return None, False, f"JSONDecodeError (likely truncated generation): {e}"
        return None, False, f"JSONDecodeError: No valid JSON found ({e})"


def extract_json_payload(raw_text: str) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Parse raw LLM generation and extract JSON object (backward-compatible 2-tuple)."""
    parsed, is_valid, _ = extract_json_payload_with_diagnostics(raw_text)
    return parsed, is_valid
