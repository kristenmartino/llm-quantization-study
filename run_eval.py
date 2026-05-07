"""Main eval runner: arms × tasks → JSONL.

For each (arm, task) pair, runs every example through Ollama with deterministic
sampling, scores the output, and writes one JSONL line per example to
results/{task}_{arm}.jsonl. Each line records the score, model output, gold
answer, and per-call timing — enough to reproduce stats and Pareto plots
downstream without re-running the model.

Usage:
    python run_eval.py --task mmlu --n 500
    python run_eval.py --task extraction --n 300 --arms fp16,q8_0,q4_K_M
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import ollama_client
import tasks


# Map short arm names to full Ollama model tags
ARM_TAGS = {
    "fp16": "llama3.1:8b-instruct-fp16",
    "q8_0": "llama3.1:8b-instruct-q8_0",
    "q4_K_M": "llama3.1:8b-instruct-q4_K_M",
}


def _run_one(
    task: tasks.Task, arm: str, ex: tasks.Example
) -> tuple[float, dict]:
    """Run a single example for one arm. Returns (score, jsonl_record)."""
    model_tag = ARM_TAGS[arm]
    try:
        result = ollama_client.generate(
            model=model_tag,
            prompt=ex.prompt,
            system=task.system_prompt,
            max_tokens=task.max_tokens,
        )
        score = task.score_fn(result.text, ex.gold)
    except Exception as exc:
        # Don't let one bad call kill a 500-item run; record it and move on.
        result = None
        score = 0.0
        print(f"[{arm}] example {ex.id} failed: {exc}")

    record = {
        "example_id": ex.id,
        "arm": arm,
        "task": task.name,
        "score": score,
        "output": result.text if result else "",
        "gold": ex.gold,
        "metadata": ex.metadata,
        "timing": {
            "eval_count": result.eval_count if result else 0,
            "eval_duration_ns": result.eval_duration_ns if result else 0,
            "total_duration_ns": result.total_duration_ns if result else 0,
            "tokens_per_second": result.tokens_per_second if result else 0.0,
        },
    }
    return score, record


def _existing_ids(path: Path) -> set[str]:
    """Read example_ids already recorded in a JSONL file (for resume)."""
    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line)["example_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def run_eval(
    task: tasks.Task,
    arms: list[str],
    results_dir: Path,
    schedule: str = "roundrobin",
    resume: bool = False,
) -> None:
    """Run task across all arms, streaming JSONL per arm.

    schedule="roundrobin": for each example, run each arm in turn before
        advancing — splits any monotonic thermal/daemon drift evenly across
        arms (the cost-vs-quality framing depends on tok/sec being unbiased).
    schedule="sequential": old behavior — run arm A fully, then B, then C.
    resume=True: skip an example only when ALL arms already have a record for
        that example_id; partial examples re-run all arms (cost ≤ k extra
        calls per crash).
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    out_paths = {arm: results_dir / f"{task.name}_{arm}.jsonl" for arm in arms}

    # Warmups for all arms before timed work — first-call latency on a freshly
    # loaded model can be many times steady-state, biasing whichever arm runs
    # first. Done sequentially before the example loop, not interleaved.
    for arm in arms:
        print(f"[{arm}] warming up {ARM_TAGS[arm]}...")
        ollama_client.warmup(ARM_TAGS[arm])

    # Resume: drop examples already complete across all arms.
    examples = task.examples
    if resume:
        existing = {arm: _existing_ids(out_paths[arm]) for arm in arms}
        to_run = [
            ex for ex in examples
            if not all(ex.id in existing[arm] for arm in arms)
        ]
        skipped = len(examples) - len(to_run)
        if skipped:
            print(f"[resume] skipping {skipped} examples already complete in all arms")
        examples = to_run

    file_mode = "a" if resume else "w"
    handles = {arm: out_paths[arm].open(file_mode) for arm in arms}
    running_sum = {arm: 0.0 for arm in arms}
    counts = {arm: 0 for arm in arms}
    start = time.time()

    try:
        if schedule == "roundrobin":
            for i, ex in enumerate(examples):
                for arm in arms:
                    score, record = _run_one(task, arm, ex)
                    handles[arm].write(json.dumps(record) + "\n")
                    handles[arm].flush()
                    running_sum[arm] += score
                    counts[arm] += 1

                if (i + 1) % 25 == 0:
                    elapsed = time.time() - start
                    means = "  ".join(
                        f"{a}={running_sum[a]/max(1, counts[a]):.3f}" for a in arms
                    )
                    print(f"[rr] {i + 1}/{len(examples)}  {means}  elapsed={elapsed:.0f}s")

        elif schedule == "sequential":
            for arm in arms:
                print(f"\n[{arm}] running {len(examples)} examples -> {out_paths[arm]}")
                arm_start = time.time()
                for i, ex in enumerate(examples):
                    score, record = _run_one(task, arm, ex)
                    handles[arm].write(json.dumps(record) + "\n")
                    handles[arm].flush()
                    running_sum[arm] += score
                    counts[arm] += 1
                    if (i + 1) % 25 == 0:
                        m = running_sum[arm] / counts[arm]
                        e = time.time() - arm_start
                        print(f"[{arm}] {i + 1}/{len(examples)}  mean={m:.3f}  elapsed={e:.0f}s")
                m = running_sum[arm] / max(1, counts[arm])
                print(f"[{arm}] DONE  mean={m:.4f}  elapsed={time.time() - arm_start:.0f}s")

        else:
            raise SystemExit(f"unknown schedule: {schedule}")
    finally:
        for h in handles.values():
            h.close()

    elapsed = time.time() - start
    print(f"\nALL DONE  elapsed={elapsed:.0f}s")
    for arm in arms:
        m = running_sum[arm] / max(1, counts[arm])
        print(f"  {arm}: mean={m:.4f}  n={counts[arm]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantization eval runner")
    parser.add_argument("--task", required=True, choices=["mmlu", "extraction"])
    parser.add_argument("--n", type=int, default=500, help="number of examples")
    parser.add_argument(
        "--arms",
        default="fp16,q8_0,q4_K_M",
        help="comma-separated arm names",
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--schedule",
        default="roundrobin",
        choices=["roundrobin", "sequential"],
        help="roundrobin (default) interleaves arms per-example to debias tok/sec; "
             "sequential replicates old behavior (arm A fully, then B, then C).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="append to existing JSONL files; skip examples already complete in all arms.",
    )
    args = parser.parse_args()

    arms = [a.strip() for a in args.arms.split(",")]
    for arm in arms:
        if arm not in ARM_TAGS:
            raise SystemExit(f"unknown arm: {arm}; valid: {list(ARM_TAGS)}")

    if args.task == "mmlu":
        task = tasks.load_mmlu(n=args.n)
    else:
        task = tasks.load_extraction(n=args.n)

    results_dir = Path(args.results_dir)
    print(f"Loaded {len(task.examples)} examples for task={task.name}")
    print(f"Arms: {arms}  schedule: {args.schedule}  resume: {args.resume}")
    print(f"Writing to: {results_dir.resolve()}")

    run_eval(task, arms, results_dir, schedule=args.schedule, resume=args.resume)


if __name__ == "__main__":
    main()
