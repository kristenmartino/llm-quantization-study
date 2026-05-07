"""Read JSONL results, compute statistics, generate the Pareto plot.

Reads every results/{task}_{arm}.jsonl, computes per-arm intervals and pairwise
differences, and writes:
  - results/summary.json     — machine-readable summary for the writeup
  - results/summary.txt      — human-readable table
  - results/pareto.png       — the money-shot quality vs. cost plot

Pairwise differences exploit the paired (same-example, same-seed) design:
  - MMLU (binary):    McNemar's test for p-values; paired bootstrap for CIs.
                      Cluster-bootstrap on subjects for the overall task CI.
  - NER (continuous): paired bootstrap for both p-values and CIs.

p-values are Holm-adjusted within each task (3 pairwise tests per task).

Usage:
    python analyze.py
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt

from scoring import (
    DifferenceEstimate,
    bootstrap_interval,
    holm_adjust,
    mcnemar_test,
    paired_bootstrap_diff_ci,
    paired_bootstrap_pvalue,
    wilson_interval,
)


RESULTS_DIR = Path("results")
ARMS_ORDER = ["fp16", "q8_0", "q4_K_M"]   # used for consistent plot ordering
TASKS = ["mmlu", "extraction"]
SCHEMA_VERSION = 2


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _align_arms(
    arm_rows: dict[str, list[dict]],
) -> tuple[list[str], dict[str, list[float]], list[str]]:
    """Align rows across arms by example_id.

    Returns (example_ids, scores_per_arm, subjects). Drops any example_id not
    present in every arm; warns the count. The aligned order is stable: it
    follows the order example_ids first appear in the first arm.
    """
    arms = list(arm_rows.keys())
    if not arms:
        return [], {}, []

    by_id_per_arm: dict[str, dict[str, dict]] = {
        arm: {r["example_id"]: r for r in rows} for arm, rows in arm_rows.items()
    }
    first_arm_order = [r["example_id"] for r in arm_rows[arms[0]]]
    seen: set[str] = set()
    aligned_ids: list[str] = []
    dropped = 0
    for eid in first_arm_order:
        if eid in seen:
            continue
        seen.add(eid)
        if all(eid in by_id_per_arm[a] for a in arms):
            aligned_ids.append(eid)
        else:
            dropped += 1
    # Also count any IDs in other arms not in the first arm
    for arm in arms[1:]:
        for eid in by_id_per_arm[arm]:
            if eid not in seen:
                dropped += 1

    if dropped:
        print(f"  warning: dropped {dropped} example(s) not present in all arms")

    scores: dict[str, list[float]] = {arm: [] for arm in arms}
    subjects: list[str] = []
    for eid in aligned_ids:
        first_meta = by_id_per_arm[arms[0]][eid].get("metadata") or {}
        subjects.append(first_meta.get("subject", ""))
        for arm in arms:
            scores[arm].append(float(by_id_per_arm[arm][eid]["score"]))
    return aligned_ids, scores, subjects


def _pairwise_diffs(
    scores_per_arm: dict[str, list[float]],
    is_binary: bool,
    cluster_keys: list | None = None,
    n_bootstrap: int = 5000,
) -> list[DifferenceEstimate]:
    """Compute paired pairwise differences for one task (or one subject slice).

    Uses McNemar for binary p-values, paired bootstrap for continuous. CIs use
    paired bootstrap throughout (cluster-bootstrap if cluster_keys provided).
    Holm-adjusted p-values are populated within the returned family.
    """
    arms_present = list(scores_per_arm.keys())
    estimates: list[DifferenceEstimate] = []
    for i, arm_a in enumerate(arms_present):
        for arm_b in arms_present[i + 1 :]:
            sa = scores_per_arm[arm_a]
            sb = scores_per_arm[arm_b]
            de = paired_bootstrap_diff_ci(
                sa, sb, arm_a, arm_b,
                n_resamples=n_bootstrap,
                cluster_keys=cluster_keys,
            )
            if is_binary:
                _, p = mcnemar_test(sa, sb)
            else:
                p = paired_bootstrap_pvalue(
                    sa, sb, n_resamples=n_bootstrap,
                    cluster_keys=cluster_keys,
                )
            de.p_value = p
            estimates.append(de)

    # Holm adjustment within this family
    p_values = [de.p_value for de in estimates]
    if p_values:
        adjusted = holm_adjust(p_values)
        for de, p_adj in zip(estimates, adjusted):
            de.p_adj = p_adj

    return estimates


def analyze_task(task: str) -> dict:
    """Compute per-arm intervals + paired pairwise diffs (with Holm) for one task.

    For MMLU, also produces a per-subject breakdown and uses cluster-bootstrap
    on subjects for the overall pairwise CIs (subjects are the cluster unit;
    within-subject correlation is real, iid bootstrap underestimates variance).
    """
    is_binary = task == "mmlu"

    arm_rows: dict[str, list[dict]] = {}
    arm_speed: dict[str, float] = {}
    for arm in ARMS_ORDER:
        path = RESULTS_DIR / f"{task}_{arm}.jsonl"
        if not path.exists():
            print(f"  skipping {arm}: {path} not found")
            continue
        rows = load_jsonl(path)
        arm_rows[arm] = rows
        speeds = [
            r["timing"]["tokens_per_second"]
            for r in rows
            if r["timing"]["tokens_per_second"] > 0
        ]
        arm_speed[arm] = statistics.mean(speeds) if speeds else 0.0

    if not arm_rows:
        return {"task": task, "intervals": {}, "pairwise_diffs": [],
                "speeds_tok_per_sec": {}, "n_per_arm": {},
                "schema_version": SCHEMA_VERSION}

    example_ids, scores_per_arm, subjects = _align_arms(arm_rows)

    # Per-arm point estimates with CIs (per-arm marginals — independent of pairing)
    intervals = {}
    for arm, scores in scores_per_arm.items():
        ci_fn = wilson_interval if is_binary else bootstrap_interval
        intervals[arm] = asdict(ci_fn(scores))

    # Pairwise diffs: cluster-bootstrap on subjects for MMLU overall, plain paired for NER
    cluster_keys = subjects if (is_binary and any(subjects)) else None
    pair_estimates = _pairwise_diffs(
        scores_per_arm, is_binary=is_binary, cluster_keys=cluster_keys,
    )

    result: dict = {
        "task": task,
        "schema_version": SCHEMA_VERSION,
        "intervals": intervals,
        "pairwise_diffs": [asdict(d) for d in pair_estimates],
        "speeds_tok_per_sec": arm_speed,
        "n_per_arm": {arm: len(s) for arm, s in scores_per_arm.items()},
        "n_aligned": len(example_ids),
    }

    # Per-subject MMLU breakdown
    if is_binary and any(subjects):
        result["per_subject"] = _per_subject_breakdown(scores_per_arm, subjects, is_binary)

    return result


def _per_subject_breakdown(
    scores_per_arm: dict[str, list[float]],
    subjects: list[str],
    is_binary: bool,
) -> dict:
    """Slice MMLU results by subject; report per-arm intervals and pairwise diffs."""
    by_subject: dict[str, dict[str, list[float]]] = {}
    for i, subj in enumerate(subjects):
        if not subj:
            continue
        for arm, scores in scores_per_arm.items():
            by_subject.setdefault(subj, {}).setdefault(arm, []).append(scores[i])

    out: dict = {}
    for subj, arm_scores in sorted(by_subject.items()):
        intervals = {
            arm: asdict(wilson_interval(s) if is_binary else bootstrap_interval(s))
            for arm, s in arm_scores.items()
        }
        # No clustering within a single subject — plain paired bootstrap
        pair_estimates = _pairwise_diffs(arm_scores, is_binary=is_binary)
        out[subj] = {
            "n": len(next(iter(arm_scores.values()))),
            "intervals": intervals,
            "pairwise_diffs": [asdict(d) for d in pair_estimates],
        }
    return out


def format_summary(results: list[dict]) -> str:
    """Plain-text summary suitable for a terminal or pasting into the writeup."""
    lines = []
    for res in results:
        lines.append(f"\n=== {res['task'].upper()} ===")
        lines.append(f"n per arm: {res['n_per_arm']}  (aligned: {res.get('n_aligned', '?')})")
        lines.append("\nPer-arm performance (95% CI):")
        for arm, est in res["intervals"].items():
            lines.append(
                f"  {arm:8s}  {est['mean']:.4f}  "
                f"[{est['lower']:.4f}, {est['upper']:.4f}]  "
                f"speed={res['speeds_tok_per_sec'].get(arm, 0):.1f} tok/s"
            )
        lines.append("\nPaired pairwise differences (95% CI; * = Holm-adjusted p<0.05):")
        for d in res["pairwise_diffs"]:
            lines.append("  " + _format_diff(d))

        if "per_subject" in res:
            lines.append("\nPer-subject MMLU effects (sorted by Q4-FP16 Δ):")
            ranked = sorted(
                res["per_subject"].items(),
                key=lambda kv: _q4_minus_fp16(kv[1]),
            )
            for subj, sub in ranked:
                q4_fp = _q4_minus_fp16(sub)
                fp16_acc = sub["intervals"].get("fp16", {}).get("mean", float("nan"))
                q4_acc = sub["intervals"].get("q4_K_M", {}).get("mean", float("nan"))
                lines.append(
                    f"  {subj:32s}  fp16={fp16_acc:.3f}  q4={q4_acc:.3f}  Δ={q4_fp:+.3f}  (n={sub['n']})"
                )
    return "\n".join(lines)


def _format_diff(d: dict) -> str:
    p_adj = d.get("p_adj")
    sig = " *" if (p_adj is not None and p_adj < 0.05) else ""
    p_str = f"  p={d['p_value']:.4g} p_adj={p_adj:.4g}" if p_adj is not None else ""
    return (
        f"{d['arm_a']} - {d['arm_b']}: {d['diff']:+.4f}  "
        f"[{d['lower']:+.4f}, {d['upper']:+.4f}]{p_str}{sig}"
    )


def _q4_minus_fp16(per_subject_entry: dict) -> float:
    for d in per_subject_entry["pairwise_diffs"]:
        if d["arm_a"] == "fp16" and d["arm_b"] == "q4_K_M":
            return -d["diff"]  # report q4 - fp16
        if d["arm_a"] == "q4_K_M" and d["arm_b"] == "fp16":
            return d["diff"]
    return 0.0


def plot_pareto(results: list[dict], out_path: Path) -> None:
    """Quality-vs-cost frontier across arms, one panel per task."""
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        for arm in ARMS_ORDER:
            if arm not in res["intervals"]:
                continue
            quality = res["intervals"][arm]["mean"]
            speed = res["speeds_tok_per_sec"].get(arm, 0)
            ci_low = res["intervals"][arm]["lower"]
            ci_high = res["intervals"][arm]["upper"]
            ax.errorbar(
                speed,
                quality,
                yerr=[[quality - ci_low], [ci_high - quality]],
                fmt="o",
                markersize=10,
                capsize=5,
                label=arm,
            )
            ax.annotate(
                arm, (speed, quality),
                textcoords="offset points", xytext=(8, 8), fontsize=11,
            )
        ax.set_xlabel("Inference speed (tokens/sec)")
        ax.set_ylabel("Accuracy" if res["task"] == "mmlu" else "Span-F1 (CoNLL)")
        ax.set_title(f"{res['task']}: quality vs. cost")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    results = []
    for task in TASKS:
        if not any((RESULTS_DIR / f"{task}_{arm}.jsonl").exists() for arm in ARMS_ORDER):
            print(f"no result files found for task={task}; skipping")
            continue
        print(f"\nAnalyzing {task}...")
        results.append(analyze_task(task))

    if not results:
        print("No results to analyze. Run the eval first.")
        return

    # Write machine-readable summary
    with (RESULTS_DIR / "summary.json").open("w") as f:
        json.dump(results, f, indent=2)

    # Write human-readable summary
    summary_text = format_summary(results)
    with (RESULTS_DIR / "summary.txt").open("w") as f:
        f.write(summary_text)
    print(summary_text)

    # Plot
    plot_pareto(results, RESULTS_DIR / "pareto.png")


if __name__ == "__main__":
    main()
