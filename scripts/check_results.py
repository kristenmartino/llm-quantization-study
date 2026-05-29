"""End-to-end guard: the committed numbers must reproduce from the committed data.

Unit tests cover the statistical primitives; this script covers the pipeline and
the artifacts a reader actually trusts. It runs in CI with NO model or Ollama —
the raw JSONL outputs are committed, so analyze.py can be replayed offline.

Checks:
  1. Schema/integrity of the committed raw outputs (required fields, status ok,
     no unexpected duplicate (arm, example_id) records).
  2. summary.json reproduces from the committed JSONL within tolerance
     (point estimates exact; bootstrap CIs deterministic under the fixed seed).

Exit non-zero on any drift so a stale summary or a silent pipeline change fails
the build. Usage: python scripts/check_results.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
ARMS = ["fp16", "q8_0", "q4_K_M"]
TASKS = ["mmlu", "extraction"]
REQUIRED_FIELDS = {"example_id", "arm", "task", "score", "output", "gold", "timing"}
# One genuine duplicate question in cais/mmlu high_school_mathematics is expected
# (same stem/answer, shuffled choices -> same question hash). Documented in README.
KNOWN_DUP_IDS = {"high_school_mathematics::d3f3d532"}
TOL = 1e-6

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)


def check_raw_integrity() -> None:
    for task in TASKS:
        for arm in ARMS:
            path = RESULTS / f"{task}_{arm}.jsonl"
            if not path.exists():
                failures.append(f"missing {path}")
                continue
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            ids: dict[str, int] = {}
            for r in rows:
                missing = REQUIRED_FIELDS - r.keys()
                check(not missing, f"{path.name}: row missing fields {missing}")
                check(r.get("status", "ok") == "ok", f"{path.name}: non-ok status row {r.get('example_id')}")
                ids[r["example_id"]] = ids.get(r["example_id"], 0) + 1
            unexpected = {k for k, v in ids.items() if v > 1} - KNOWN_DUP_IDS
            check(not unexpected, f"{path.name}: unexpected duplicate ids {unexpected}")


def _num_close(a, b, path: str) -> None:
    if isinstance(a, bool) or isinstance(b, bool):
        check(a == b, f"{path}: {a} != {b}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        check(abs(a - b) <= TOL, f"{path}: {a} != {b} (>|{TOL}|)")
    elif isinstance(a, dict) and isinstance(b, dict):
        check(a.keys() == b.keys(), f"{path}: key mismatch {set(a) ^ set(b)}")
        for k in a.keys() & b.keys():
            _num_close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        check(len(a) == len(b), f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            _num_close(x, y, f"{path}[{i}]")
    else:
        check(a == b, f"{path}: {a!r} != {b!r}")


def check_summary_reproduces() -> None:
    committed_path = RESULTS / "summary.json"
    if not committed_path.exists():
        failures.append("results/summary.json missing")
        return
    committed = json.loads(committed_path.read_text())

    # Replay analyze.py into a scratch dir copy so we never clobber the committed
    # artifacts during the check.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_results = Path(tmp) / "results"
        tmp_results.mkdir()
        for task in TASKS:
            for arm in ARMS:
                src = RESULTS / f"{task}_{arm}.jsonl"
                if src.exists():
                    (tmp_results / src.name).write_text(src.read_text())
        proc = subprocess.run(
            [sys.executable, str(REPO / "analyze.py")],
            cwd=tmp, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            failures.append(f"analyze.py failed on replay:\n{proc.stderr[-2000:]}")
            return
        regen = json.loads((tmp_results / "summary.json").read_text())

    check(len(committed) == len(regen), "summary.json: task count changed")
    for c, r in zip(committed, regen):
        _num_close(c, r, f"summary[{c.get('task')}]")


def main() -> int:
    check_raw_integrity()
    check_summary_reproduces()
    if failures:
        print("FAIL — committed results do not validate:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("OK — raw outputs valid and summary.json reproduces from committed data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
