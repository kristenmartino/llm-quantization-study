"""Ollama HTTP wrapper with explicit sampling settings.

Defaults use temperature 0, seed 42 and shared sampling options across arms.
Requests record generated-token counts and timing for generation-speed
comparisons. Token-generation speed excludes prompt processing and loading.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_TIMEOUT = 300  # seconds
DEFAULT_SEED = 42

# Sampling held constant across arms — the single source of truth, also recorded
# in the experiment manifest. seed and num_predict are per-call (added below).
SAMPLING_OPTIONS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "repeat_penalty": 1.0,
}


@dataclass
class GenerationResult:
    """Model output with per-call token counts and timing."""
    text: str
    eval_count: int          # tokens generated
    eval_duration_ns: int    # generation time (excludes prompt processing)
    total_duration_ns: int
    prompt_eval_count: int   # prompt tokens
    model: str

    @property
    def tokens_per_second(self) -> float:
        if self.eval_duration_ns == 0:
            return 0.0
        return self.eval_count / (self.eval_duration_ns / 1e9)


def generate(
    model: str,
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 512,
    seed: int = DEFAULT_SEED,
    timeout: int = DEFAULT_TIMEOUT,
) -> GenerationResult:
    """Send one Ollama request with explicit sampling options and a seed.

    Shared defaults come from SAMPLING_OPTIONS. The caller supplies the model,
    prompt, optional system message, seed and output-token limit.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            **SAMPLING_OPTIONS,
            "seed": seed,
            "num_predict": max_tokens,
        },
    }
    if system is not None:
        payload["system"] = system

    response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()

    return GenerationResult(
        text=data["response"],
        eval_count=data.get("eval_count", 0),
        eval_duration_ns=data.get("eval_duration", 0),
        total_duration_ns=data.get("total_duration", 0),
        prompt_eval_count=data.get("prompt_eval_count", 0),
        model=model,
    )


def warmup(model: str) -> None:
    """Issue a short initialization request before evaluation.

    The runner calls this for each arm before starting the timed loop.
    Model residency is managed by Ollama.
    """
    generate(model=model, prompt="Hi", max_tokens=4)
    # Brief pause to let the runtime stabilize
    time.sleep(0.5)
