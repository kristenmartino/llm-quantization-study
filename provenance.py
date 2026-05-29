"""Capture experiment provenance into results/experiment_manifest.json.

The study's headline is a comparison *across quantization tiers*. For that to be
a clean comparison, the three Ollama tags must differ only in quantization — same
base weights, tokenizer, and chat template. This module records the evidence for
that assumption instead of leaving it implicit:

  - per-arm weight-blob sha256 (the three differ, as expected for distinct quants)
  - per-arm modelfile (shows identical stop tokens / template across arms)
  - Ollama version, host, git commit, dataset ids, seed, and sampling options

`update_manifest` is called by run_eval at run start; the same builder produced
the committed manifest for the original run (whose models were still local).
Everything degrades gracefully if the `ollama`/`git` CLIs are unavailable.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        ).stdout
    except Exception:
        return ""


def ollama_version() -> str:
    out = _run(["ollama", "--version"])
    m = re.search(r"(\d+\.\d+\.\d+)", out)
    return m.group(1) if m else (out.strip() or "unknown")


def _short_digests() -> dict[str, str]:
    """tag -> short manifest digest, parsed from `ollama list`."""
    digests: dict[str, str] = {}
    for line in _run(["ollama", "list"]).splitlines():
        parts = line.split()
        if len(parts) >= 2 and ":" in parts[0]:
            digests[parts[0]] = parts[1]
    return digests


def model_provenance(tag: str, short_digests: dict[str, str] | None = None) -> dict:
    """Capture identity for one model tag: digests, quantization, modelfile."""
    modelfile = _run(["ollama", "show", "--modelfile", tag])
    weight_blob = None
    for line in modelfile.splitlines():
        m = re.match(r"FROM\s+.*blobs/sha256-([0-9a-f]+)", line.strip())
        if m:
            weight_blob = m.group(1)
            break
    quant = None
    for line in _run(["ollama", "show", tag]).splitlines():
        if "quantization" in line.lower():
            parts = line.split()
            quant = parts[-1] if parts else None
            break
    sd = short_digests if short_digests is not None else _short_digests()
    return {
        "tag": tag,
        "manifest_digest": sd.get(tag),
        "weight_blob_sha256": weight_blob,
        "quantization": quant,
        "modelfile": modelfile.strip() or None,
    }


def host_info() -> dict:
    chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "chip": chip or platform.processor(),
    }


def git_commit() -> str:
    return _run(["git", "rev-parse", "HEAD"]).strip() or "unknown"


_DATASETS = {
    "mmlu": {"hf_id": "cais/mmlu", "split": "test"},
    "extraction": {"hf_id": "eriktks/conll2003", "split": "test"},
}

_ASSUMPTION = (
    "Causal interpretation assumes the arms share identical base weights, "
    "tokenizer, and chat template, differing only in quantization. The per-arm "
    "weight_blob_sha256 differ (expected for distinct quantizations); identical "
    "stop tokens and modelfile templates across arms support the assumption. "
    "Dataset revisions are not pinned to a commit hash (a known reproducibility gap)."
)


def update_manifest(
    path: Path,
    task: str,
    n: int,
    arm_tags: dict[str, str],
    arms: list[str],
    seed: int,
    sampling: dict,
) -> dict:
    """Create or update the manifest for one run, merging the task's n.

    Running mmlu then extraction yields a single manifest covering both tasks.
    Model/host/version fields are refreshed each call.
    """
    path = Path(path)
    manifest: dict = {}
    if path.exists():
        try:
            manifest = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            manifest = {}

    sd = _short_digests()
    manifest.update({
        "schema": "experiment_manifest/1",
        "ollama_version": ollama_version(),
        "host": host_info(),
        "git_commit": git_commit(),
        "seed": seed,
        "sampling": sampling,
        "datasets": _DATASETS,
        "arms": [model_provenance(arm_tags[a], sd) for a in arms],
        "identifying_assumption": _ASSUMPTION,
    })
    tasks = manifest.get("tasks", {})
    tasks[task] = {"n_requested": n}
    manifest["tasks"] = tasks
    path.write_text(json.dumps(manifest, indent=2))
    return manifest
