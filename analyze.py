"""Read JSONL results, compute statistics, generate the Pareto plot.

Reads every results/{task}_{arm}.jsonl, computes per-arm intervals and pairwise
differences, and writes:
  - results/summary.json     — machine-readable summary for the writeup
  - results/summary.txt      — human-readable table
  - results/pareto.png       — the money-shot quality vs. cost plot

Pairwise differences exploit the paired (same-example, same-seed) design:
  - MMLU (binary):    McNemar's test for p-values; paired bootstrap for CIs.
                      Cluster-bootstrap on subjects for the overall task CI.
  - NER:              two estimands are reported.
                        * micro-F1 (canonical CoNLL): pool TP/FP/FN, then F1;
                          paired bootstrap over examples for CIs/p-values.
                        * macro-F1 (per-sentence): mean of per-example F1;
                          paired bootstrap on the per-example deltas.
                      micro is the primary/headline metric; macro is reported
                      alongside because the gap between them localizes the
                      failure mode (over-extraction on entity-free sentences).

p-values are Holm-adjusted within each task (3 pairwise tests per task).

Equivalence ("Q8_0 ≈ FP16") is a TOST result against a pre-declared ±1pp margin,
not the absence of a significant test — see `equivalence` in the summary.

Usage:
    python analyze.py
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: works in CI without a display
import matplotlib.pyplot as plt

import tasks
from scoring import (
    DifferenceEstimate,
    bootstrap_interval,
    holm_adjust,
    mcnemar_test,
    micro_f1_interval,
    paired_bootstrap_diff_ci,
    paired_bootstrap_pvalue,
    paired_micro_f1_diff_ci,
    paired_micro_f1_pvalue,
    tost_equivalence,
    wilson_interval,
)


RESULTS_DIR = Path("results")
ARMS_ORDER = ["fp16", "q8_0", "q4_K_M"]   # used for consistent plot ordering
TASKS = ["mmlu", "extraction"]
SCHEMA_VERSION = 3

# Pre-declared equivalence margins (set a priori, not after seeing the data):
# ±1 percentage point on both MMLU accuracy and NER F1.
EQUIVALENCE_MARGIN = 0.01


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _dedup_rows(rows: list[dict], arm: str, warn: bool = True) -> tuple[dict[str, dict], int]:
    """Map example_id -> row (last wins). Warns loudly if a file has duplicates.

    A raw file with more rows than unique example_ids means either a resume
    re-append or a source duplicate collapsed onto one id; either way the
    collapse should be visible, not silent.
    """
    by_id: dict[str, dict] = {}
    dup_ids: list[str] = []
    for r in rows:
        eid = r["example_id"]
        if eid in by_id:
            dup_ids.append(eid)
        by_id[eid] = r
    if dup_ids and warn:
        shown = ", ".join(sorted(set(dup_ids))[:5])
        print(f"  warning: {arm} has {len(dup_ids)} duplicate example_id record(s) "
              f"(collapsed last-wins): {shown}")
    return by_id, len(dup_ids)


def _align_arms(
    arm_rows: dict[str, list[dict]],
) -> tuple[list[str], dict[str, list[float]], list[str], dict[str, dict[str, dict]]]:
    """Align rows across arms by example_id.

    Returns (example_ids, scores_per_arm, subjects, by_id_per_arm). Drops any
    example_id not present in every arm; warns the count. Within-arm duplicate
    example_ids are detected and warned in `_dedup_rows`. The aligned order is
    stable: it follows the order example_ids first appear in the first arm.
    """
    arms = list(arm_rows.keys())
    if not arms:
        return [], {}, [], {}

    by_id_per_arm: dict[str, dict[str, dict]] = {}
    for arm, rows in arm_rows.items():
        by_id, _ = _dedup_rows(rows, arm)
        by_id_per_arm[arm] = by_id

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
    return aligned_ids, scores, subjects, by_id_per_arm


def _arm_speed(rows: list[dict]) -> float:
    """Mean tok/sec over deduplicated examples (one record per example_id)."""
    by_id, _ = _dedup_rows(rows, "speed", warn=False)
    speeds = [
        r["timing"]["tokens_per_second"]
        for r in by_id.values()
        if r["timing"]["tokens_per_second"] > 0
    ]
    return statistics.mean(speeds) if speeds else 0.0


def _pairwise_diffs(
    scores_per_arm: dict[str, list[float]],
    is_binary: bool,
    cluster_keys: list | None = None,
    n_bootstrap: int = 5000,
) -> list[DifferenceEstimate]:
    """Paired pairwise differences for one per-example metric (with Holm).

    McNemar for binary p-values, paired bootstrap for continuous. CIs use
    paired bootstrap throughout (cluster-bootstrap if cluster_keys provided).
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

    p_values = [de.p_value for de in estimates]
    if p_values:
        for de, p_adj in zip(estimates, holm_adjust(p_values)):
            de.p_adj = p_adj
    return estimates


def _micro_pairwise_diffs(
    counts_per_arm: dict[str, list[tuple[int, int, int]]],
    n_bootstrap: int = 5000,
) -> list[DifferenceEstimate]:
    """Paired pairwise corpus micro-F1 differences (with Holm) for NER."""
    arms_present = list(counts_per_arm.keys())
    estimates: list[DifferenceEstimate] = []
    for i, arm_a in enumerate(arms_present):
        for arm_b in arms_present[i + 1 :]:
            ca, cb = counts_per_arm[arm_a], counts_per_arm[arm_b]
            de = paired_micro_f1_diff_ci(ca, cb, arm_a, arm_b, n_resamples=n_bootstrap)
            de.p_value = paired_micro_f1_pvalue(ca, cb, n_resamples=n_bootstrap)
            estimates.append(de)
    p_values = [de.p_value for de in estimates]
    if p_values:
        for de, p_adj in zip(estimates, holm_adjust(p_values)):
            de.p_adj = p_adj
    return estimates


def _equivalence_tests(
    scores_per_arm: dict[str, list[float]],
    margin: float,
    cluster_keys: list | None = None,
) -> list[dict]:
    """TOST equivalence for every arm pair on a per-example metric."""
    arms_present = list(scores_per_arm.keys())
    out: list[dict] = []
    for i, arm_a in enumerate(arms_present):
        for arm_b in arms_present[i + 1 :]:
            res = tost_equivalence(
                scores_per_arm[arm_a], scores_per_arm[arm_b], margin,
                arm_a=arm_a, arm_b=arm_b, cluster_keys=cluster_keys,
            )
            out.append(asdict(res))
    return out


def analyze_task(task: str) -> dict:
    """Compute per-arm intervals + paired pairwise diffs (with Holm) for one task."""
    is_binary = task == "mmlu"

    arm_rows: dict[str, list[dict]] = {}
    arm_speed: dict[str, float] = {}
    for arm in ARMS_ORDER:
        path = RESULTS_DIR / f"{task}_{arm}.jsonl"
        if not path.exists():
            print(f"  skipping {arm}: {path} not found")
            continue
        rows = load_jsonl(path)
        # Drop infrastructure failures (status != "ok") so a daemon timeout is
        # never scored as a wrong answer. Rows predating the status field count
        # as "ok". This run has zero such rows; the guard is for future reruns.
        n_raw = len(rows)
        rows = [r for r in rows if r.get("status", "ok") == "ok"]
        if len(rows) != n_raw:
            print(f"  note: {arm} excluded {n_raw - len(rows)} runtime-error row(s)")
        arm_rows[arm] = rows
        arm_speed[arm] = _arm_speed(rows)

    if not arm_rows:
        return {"task": task, "intervals": {}, "pairwise_diffs": [],
                "speeds_tok_per_sec": {}, "n_per_arm": {},
                "schema_version": SCHEMA_VERSION}

    example_ids, scores_per_arm, subjects, by_id_per_arm = _align_arms(arm_rows)

    if is_binary:
        return _analyze_mmlu(task, scores_per_arm, subjects, arm_speed, example_ids)
    return _analyze_ner(task, scores_per_arm, by_id_per_arm, example_ids, arm_speed)


def _analyze_mmlu(task, scores_per_arm, subjects, arm_speed, example_ids) -> dict:
    intervals = {arm: asdict(wilson_interval(s)) for arm, s in scores_per_arm.items()}
    cluster_keys = subjects if any(subjects) else None
    pair_estimates = _pairwise_diffs(scores_per_arm, is_binary=True, cluster_keys=cluster_keys)
    result: dict = {
        "task": task,
        "schema_version": SCHEMA_VERSION,
        "primary_metric": "accuracy",
        "intervals": intervals,
        "pairwise_diffs": [asdict(d) for d in pair_estimates],
        "equivalence_margin": EQUIVALENCE_MARGIN,
        "equivalence": _equivalence_tests(scores_per_arm, EQUIVALENCE_MARGIN, cluster_keys),
        "speeds_tok_per_sec": arm_speed,
        "n_per_arm": {arm: len(s) for arm, s in scores_per_arm.items()},
        "n_aligned": len(example_ids),
    }
    if any(subjects):
        result["per_subject"] = _per_subject_breakdown(scores_per_arm, subjects, True)
    return result


def _analyze_ner(task, scores_per_arm, by_id_per_arm, example_ids, arm_speed) -> dict:
    arms = list(scores_per_arm.keys())
    # Per-example (tp, fp, fn) for the canonical micro-F1, aligned across arms.
    counts_per_arm: dict[str, list[tuple[int, int, int]]] = {arm: [] for arm in arms}
    pr_per_arm: dict[str, dict] = {}
    for arm in arms:
        for eid in example_ids:
            row = by_id_per_arm[arm][eid]
            c = tasks.score_extraction_counts(row["output"], row["gold"])
            counts_per_arm[arm].append((c["tp"], c["fp"], c["fn"]))
        tp = sum(c[0] for c in counts_per_arm[arm])
        fp = sum(c[1] for c in counts_per_arm[arm])
        fn = sum(c[2] for c in counts_per_arm[arm])
        pr_per_arm[arm] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": (tp / (tp + fp)) if (tp + fp) else 0.0,
            "recall": (tp / (tp + fn)) if (tp + fn) else 0.0,
        }

    # Primary = corpus micro-F1; secondary = per-sentence macro-F1.
    intervals_micro = {arm: asdict(micro_f1_interval(counts_per_arm[arm])) for arm in arms}
    intervals_macro = {arm: asdict(bootstrap_interval(scores_per_arm[arm])) for arm in arms}
    diffs_micro = _micro_pairwise_diffs(counts_per_arm)
    diffs_macro = _pairwise_diffs(scores_per_arm, is_binary=False)

    return {
        "task": task,
        "schema_version": SCHEMA_VERSION,
        "primary_metric": "micro_f1",
        "intervals": intervals_micro,                     # primary (plotted/headline)
        "pairwise_diffs": [asdict(d) for d in diffs_micro],
        "intervals_macro": intervals_macro,               # secondary (brittleness)
        "pairwise_diffs_macro": [asdict(d) for d in diffs_macro],
        "precision_recall": pr_per_arm,
        "equivalence_margin": EQUIVALENCE_MARGIN,
        # Formal TOST runs on the per-example (macro) deltas; micro equivalence
        # is asserted conservatively via the micro CI lying within the margin.
        "equivalence": _equivalence_tests(scores_per_arm, EQUIVALENCE_MARGIN),
        "equivalence_micro_within_margin": {
            f"{d.arm_a}-{d.arm_b}": bool(d.lower > -EQUIVALENCE_MARGIN and d.upper < EQUIVALENCE_MARGIN)
            for d in diffs_micro
        },
        "speeds_tok_per_sec": arm_speed,
        "n_per_arm": {arm: len(s) for arm, s in scores_per_arm.items()},
        "n_aligned": len(example_ids),
    }


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
        primary = res.get("primary_metric", "accuracy")
        label = {"accuracy": "accuracy", "micro_f1": "micro-F1 (canonical CoNLL)"}.get(primary, primary)
        lines.append(f"\nPer-arm {label} (95% CI):")
        for arm, est in res["intervals"].items():
            extra = ""
            pr = res.get("precision_recall", {}).get(arm)
            if pr:
                extra = f"  P={pr['precision']:.3f} R={pr['recall']:.3f}"
            lines.append(
                f"  {arm:8s}  {est['mean']:.4f}  "
                f"[{est['lower']:.4f}, {est['upper']:.4f}]  "
                f"speed={res['speeds_tok_per_sec'].get(arm, 0):.1f} tok/s{extra}"
            )
        lines.append(f"\nPaired pairwise differences — {label} (95% CI; * = Holm p<0.05):")
        for d in res["pairwise_diffs"]:
            lines.append("  " + _format_diff(d))

        if "intervals_macro" in res:
            lines.append("\nSecondary: per-sentence (macro) F1 (95% CI):")
            for arm, est in res["intervals_macro"].items():
                lines.append(f"  {arm:8s}  {est['mean']:.4f}  [{est['lower']:.4f}, {est['upper']:.4f}]")
            lines.append("\nPaired pairwise differences — macro F1 (95% CI; * = Holm p<0.05):")
            for d in res["pairwise_diffs_macro"]:
                lines.append("  " + _format_diff(d))

        if res.get("equivalence"):
            margin = res.get("equivalence_margin", EQUIVALENCE_MARGIN)
            lines.append(f"\nEquivalence (TOST, ±{margin:g} margin on per-example metric):")
            for e in res["equivalence"]:
                verdict = "EQUIVALENT" if e["equivalent"] else "not established"
                lines.append(
                    f"  {e['arm_a']} ≈ {e['arm_b']}: {verdict}  "
                    f"diff={e['diff']:+.4f} 90%CI[{e['lower']:+.4f},{e['upper']:+.4f}] "
                    f"p_TOST={e['p_tost']:.4g}"
                )

        if "per_subject" in res:
            lines.append("\nPer-subject MMLU effects (sorted by Q4-FP16 Δ):")
            ranked = sorted(res["per_subject"].items(), key=lambda kv: _q4_minus_fp16(kv[1]))
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
            return -d["diff"]
        if d["arm_a"] == "q4_K_M" and d["arm_b"] == "fp16":
            return d["diff"]
    return 0.0


def plot_pareto(results: list[dict], out_path: Path) -> None:
    """Quality-vs-cost frontier across arms, one panel per task (primary metric)."""
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
                speed, quality,
                yerr=[[quality - ci_low], [ci_high - quality]],
                fmt="o", markersize=10, capsize=5, label=arm,
            )
            ax.annotate(arm, (speed, quality),
                        textcoords="offset points", xytext=(8, 8), fontsize=11)
        ax.set_xlabel("Inference speed (tokens/sec)")
        if res["task"] == "mmlu":
            ax.set_ylabel("Accuracy")
        else:
            ax.set_ylabel("Span-F1 (micro, CoNLL)")
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

    with (RESULTS_DIR / "summary.json").open("w") as f:
        json.dump(results, f, indent=2)

    summary_text = format_summary(results)
    with (RESULTS_DIR / "summary.txt").open("w") as f:
        f.write(summary_text)
    print(summary_text)

    plot_pareto(results, RESULTS_DIR / "pareto.png")


if __name__ == "__main__":
    main()
