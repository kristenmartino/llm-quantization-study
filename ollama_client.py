"""Thin wrapper around the Ollama HTTP API.

Holds sampling params constant across arms — temperature=0, fixed seed, identical
options — so the only thing varying between runs is the model tag (which encodes
the quantization level). Also captures per-call inference timing for the cost
side of the cost/quality frontier.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_TIMEOUT = 300  # seconds; quantized 8B models should respond well within this
DEFAULT_SEED = 42


@dataclass
class GenerationResult:
    """One model call: the output plus enough metadata to compute cost."""
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
    """Single deterministic generation call to Ollama.

    Temperature is pinned to 0.0 so we measure the model, not sampling variance.
    All other options are held constant across arms — the only thing that should
    differ between arm runs is `model`.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "seed": seed,
            "num_predict": max_tokens,
            "top_p": 1.0,
            "top_k": 0,
            "repeat_penalty": 1.0,
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
    """Load the model into VRAM before timed runs so first-call latency doesn't
    skew per-call timing. Ollama unloads idle models, so call this immediately
    before each arm's eval loop."""
    generate(model=model, prompt="Hi", max_tokens=4)
    # Brief pause to let the runtime stabilize
    time.sleep(0.5)
