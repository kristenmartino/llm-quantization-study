"""Statistics for the eval: confidence intervals and pairwise differences.

CI flavors:
  - Wilson score interval for binary outcomes (MMLU accuracy)
  - Bootstrap percentile interval for continuous per-example outcomes (macro F1)
  - Corpus micro-F1 with bootstrap CI for NER (the canonical CoNLL metric;
    pools TP/FP/FN across examples rather than averaging per-example F1)

The "headline" numbers in the writeup are not point estimates — they're effect
sizes with CIs. "Q4 loses 3.2pp corpus micro-F1 on NER [95% CI: 0.6–5.8pp;
Holm-adjusted p=0.034] vs FP16" beats "Q4 = 0.617, FP16 = 0.649" every time,
and beats both by including the multiple-comparison-corrected p-value.

Two F1 estimands are reported for NER and they answer different questions:
  - micro (canonical): pooled entity-level F1 — the benchmark-standard number.
  - macro (per-sentence): mean of per-example F1 — weights every sentence
    equally, so it surfaces per-request brittleness (e.g. over-extraction on
    entity-free sentences) that micro dilutes across the corpus.

Equivalence (e.g. "Q8_0 ≈ FP16") is a TOST claim against a specified
practical-equivalence margin, not the absence of a significant difference.
`tost_equivalence` implements the two one-sided tests; a non-significant pairwise
test is NOT evidence of equivalence on its own.

Pairwise tests are paired (same examples scored across arms): McNemar for
binary outcomes, paired bootstrap for continuous. Holm-Bonferroni adjusts
the p-value family within each task.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class IntervalEstimate:
    """A point estimate with a confidence interval."""
    mean: float
    lower: float
    upper: float
    n: int

    def __str__(self) -> str:
        return f"{self.mean:.4f} [{self.lower:.4f}, {self.upper:.4f}] (n={self.n})"


@dataclass
class DifferenceEstimate:
    """Effect size: arm A minus arm B, with CI on the difference."""
    arm_a: str
    arm_b: str
    diff: float
    lower: float
    upper: float
    p_value: float | None = None
    p_adj: float | None = None
    lower_adj: float | None = None
    upper_adj: float | None = None
    n_pairs: int | None = None

    def __str__(self) -> str:
        # Significance marker keys off Holm-adjusted p when available,
        # else falls back to the unadjusted CI not crossing zero.
        if self.p_adj is not None:
            sig = self.p_adj < 0.05
        else:
            sig = (self.lower > 0) or (self.upper < 0)
        marker = " *" if sig else ""

        if self.p_adj is not None and self.lower_adj is not None:
            return (
                f"{self.arm_a} - {self.arm_b}: {self.diff:+.4f} "
                f"[{self.lower:+.4f}, {self.upper:+.4f}]  "
                f"adj=[{self.lower_adj:+.4f}, {self.upper_adj:+.4f}]  "
                f"p={self.p_value:.4g} p_adj={self.p_adj:.4g}{marker}"
            )
        return (
            f"{self.arm_a} - {self.arm_b}: {self.diff:+.4f} "
            f"[{self.lower:+.4f}, {self.upper:+.4f}]{marker}"
        )


# ---------------------------------------------------------------------------
# Wilson score interval (for binary outcomes like MMLU correct/incorrect)
# ---------------------------------------------------------------------------

_Z_95 = 1.96  # Two-sided z for 95% confidence — the only level the study uses


def wilson_interval(scores: list[float]) -> IntervalEstimate:
    """Wilson score 95% CI for a binary proportion.

    More accurate than the normal approximation, especially for proportions near
    0 or 1. Inputs should be 0/1 floats (or coerced equivalents). Confidence is
    fixed at 95% — the entire study uses one CI level, so the parameter was
    pure surface area.
    """
    n = len(scores)
    if n == 0:
        return IntervalEstimate(mean=0.0, lower=0.0, upper=0.0, n=0)

    p = sum(scores) / n
    z = _Z_95
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half_width = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom

    return IntervalEstimate(
        mean=p,
        lower=max(0.0, center - half_width),
        upper=min(1.0, center + half_width),
        n=n,
    )


# ---------------------------------------------------------------------------
# Bootstrap (for continuous outcomes like F1)
# ---------------------------------------------------------------------------

def bootstrap_interval(
    scores: list[float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> IntervalEstimate:
    """Bootstrap percentile CI for the mean of a continuous metric."""
    n = len(scores)
    if n == 0:
        return IntervalEstimate(mean=0.0, lower=0.0, upper=0.0, n=0)

    rng = np.random.default_rng(seed)
    arr = np.asarray(scores, dtype=float)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        means[i] = sample.mean()

    alpha = (1 - confidence) / 2
    lower = float(np.quantile(means, alpha))
    upper = float(np.quantile(means, 1 - alpha))
    return IntervalEstimate(mean=float(arr.mean()), lower=lower, upper=upper, n=n)


# ---------------------------------------------------------------------------
# Paired bootstrap (exploits same-example, same-seed alignment across arms)
# ---------------------------------------------------------------------------

def paired_bootstrap_resamples(
    scores_a: list[float],
    scores_b: list[float],
    n_resamples: int = 5000,
    seed: int = 42,
    cluster_keys: list | None = None,
) -> tuple[float, np.ndarray]:
    """Resample paired score deltas; return (observed_mean_diff, resampled_means).

    With `cluster_keys`, performs cluster-bootstrap: resample unique cluster IDs
    with replacement, then take all paired deltas from the chosen clusters.
    Used for MMLU overall CI where examples are clustered by subject and the
    iid bootstrap underestimates variance.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must be the same length (paired)")
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    diffs = a - b
    n = len(diffs)
    diff_obs = float(diffs.mean()) if n > 0 else 0.0
    rng = np.random.default_rng(seed)
    resamples = np.empty(n_resamples)

    if cluster_keys is not None:
        if len(cluster_keys) != n:
            raise ValueError("cluster_keys length must match scores")
        cluster_to_idx: dict = {}
        for i, key in enumerate(cluster_keys):
            cluster_to_idx.setdefault(key, []).append(i)
        cluster_ids = list(cluster_to_idx.keys())
        n_clusters = len(cluster_ids)
        for k in range(n_resamples):
            chosen = rng.integers(0, n_clusters, size=n_clusters)
            picked: list[int] = []
            for c in chosen:
                picked.extend(cluster_to_idx[cluster_ids[c]])
            resamples[k] = diffs[picked].mean()
    else:
        for k in range(n_resamples):
            idx = rng.integers(0, n, size=n)
            resamples[k] = diffs[idx].mean()

    return diff_obs, resamples


def paired_bootstrap_diff_ci(
    scores_a: list[float],
    scores_b: list[float],
    arm_a: str,
    arm_b: str,
    n_resamples: int = 5000,
    seed: int = 42,
    cluster_keys: list | None = None,
    confidence: float = 0.95,
) -> DifferenceEstimate:
    """Paired bootstrap CI on the mean per-example difference.

    Works identically for binary (MMLU) and continuous (NER F1) scores —
    the per-example delta is a real number either way. Pairing reduces variance
    vs. independent two-sample CIs when the same examples are scored by both
    arms (which is the design here).
    """
    diff_obs, resamples = paired_bootstrap_resamples(
        scores_a, scores_b, n_resamples=n_resamples,
        seed=seed, cluster_keys=cluster_keys,
    )
    alpha = (1 - confidence) / 2
    lower = float(np.quantile(resamples, alpha))
    upper = float(np.quantile(resamples, 1 - alpha))
    return DifferenceEstimate(
        arm_a=arm_a, arm_b=arm_b, diff=diff_obs,
        lower=lower, upper=upper, n_pairs=len(scores_a),
    )


def paired_bootstrap_pvalue(
    scores_a: list[float],
    scores_b: list[float],
    n_resamples: int = 10000,
    seed: int = 42,
    cluster_keys: list | None = None,
) -> float:
    """Two-sided bootstrap p-value for paired mean difference under H0: diff=0.

    Centers the bootstrap distribution at zero (subtracts observed mean) and
    measures the proportion of |centered draws| ≥ |observed|. Needed because
    Holm-Bonferroni operates on p-values, and percentile-CI exclusion of zero
    alone is not a p-value.
    """
    diff_obs, resamples = paired_bootstrap_resamples(
        scores_a, scores_b, n_resamples=n_resamples,
        seed=seed, cluster_keys=cluster_keys,
    )
    # Center under H0
    centered = resamples - diff_obs
    p = float(np.mean(np.abs(centered) >= abs(diff_obs)))
    return min(1.0, max(0.0, p))


# ---------------------------------------------------------------------------
# McNemar's test (paired binary outcomes, e.g. MMLU correct/incorrect)
# ---------------------------------------------------------------------------

def mcnemar_test(
    scores_a: list[float], scores_b: list[float]
) -> tuple[float, float]:
    """McNemar's test for paired binary scores.

    Uses exact two-sided binomial test when the discordant cell count is small
    (≤25), continuity-corrected χ² approximation otherwise. Discordant pairs
    are the only ones that carry signal — concordant pairs (both right or both
    wrong) cancel.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must be the same length (paired)")
    n01 = sum(1 for a, b in zip(scores_a, scores_b) if a > 0.5 and b <= 0.5)
    n10 = sum(1 for a, b in zip(scores_a, scores_b) if a <= 0.5 and b > 0.5)
    discordant = n01 + n10
    if discordant == 0:
        return 0.0, 1.0  # all pairs agree → no evidence of difference

    if discordant <= 25:
        from scipy.stats import binom
        k = min(n01, n10)
        # Two-sided p = 2 * P(X ≤ k | n=discordant, p=0.5), capped at 1
        p_value = float(min(1.0, 2 * binom.cdf(k, discordant, 0.5)))
        statistic = float(abs(n01 - n10))
        return statistic, p_value

    from scipy.stats import chi2
    statistic = (abs(n01 - n10) - 1) ** 2 / discordant  # Yates continuity-corrected
    p_value = float(1.0 - chi2.cdf(statistic, df=1))
    return float(statistic), p_value


# ---------------------------------------------------------------------------
# Holm-Bonferroni step-down adjustment
# ---------------------------------------------------------------------------

def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment of p-values.

    Returns adjusted p-values in the same order as the input. Uniformly more
    powerful than Bonferroni; doesn't require the BH PRDS assumption (which is
    violated for our paired tests within a task).
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * p_values[i])
        running_max = max(running_max, adj)
        adjusted[i] = running_max
    return adjusted


# ---------------------------------------------------------------------------
# Corpus micro-F1 (the canonical CoNLL NER metric: pool TP/FP/FN, then F1)
# ---------------------------------------------------------------------------
#
# Per-example (macro) F1 averages each sentence's F1 equally. The canonical
# CoNLL/conlleval score instead pools counts across the whole corpus and
# computes one F1 — entity-rich sentences carry proportionally more weight.
# The two diverge when errors concentrate in particular sentence types (e.g.
# over-extraction on entity-free sentences inflates macro relative to micro),
# so the study reports both. Inputs here are per-example (tp, fp, fn) triples.

def _micro_f1(tp: float, fp: float, fn: float) -> float:
    """Pooled F1 from summed counts. Returns 0.0 when there is nothing to score."""
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def _counts_array(counts) -> np.ndarray:
    arr = np.asarray(counts, dtype=float)
    if arr.size == 0:
        return arr.reshape(0, 3)
    return arr.reshape(-1, 3)


def micro_f1_interval(
    counts, n_resamples: int = 5000, seed: int = 42, confidence: float = 0.95,
) -> IntervalEstimate:
    """Corpus micro-F1 point estimate with a bootstrap percentile CI.

    `counts` is a sequence of (tp, fp, fn) per example. The bootstrap resamples
    examples with replacement and re-pools the counts each draw, so the CI
    reflects sentence-level sampling variability in the pooled metric.
    """
    arr = _counts_array(counts)
    n = len(arr)
    if n == 0:
        return IntervalEstimate(mean=0.0, lower=0.0, upper=0.0, n=0)
    point = _micro_f1(*arr.sum(axis=0))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    sums = arr[idx].sum(axis=1)  # (n_resamples, 3)
    denom = 2 * sums[:, 0] + sums[:, 1] + sums[:, 2]
    boots = np.where(denom > 0, 2 * sums[:, 0] / denom, 0.0)
    alpha = (1 - confidence) / 2
    return IntervalEstimate(
        mean=point,
        lower=float(np.quantile(boots, alpha)),
        upper=float(np.quantile(boots, 1 - alpha)),
        n=n,
    )


def paired_micro_f1_resamples(
    counts_a, counts_b, n_resamples: int = 5000, seed: int = 42,
) -> tuple[float, np.ndarray]:
    """Resample examples (paired) and return (observed micro-F1 diff, resamples).

    Same resampled example indices are applied to both arms, preserving the
    paired design; each draw re-pools counts and takes the micro-F1 difference.
    """
    a = _counts_array(counts_a)
    b = _counts_array(counts_b)
    if len(a) != len(b):
        raise ValueError("counts_a and counts_b must be the same length (paired)")
    n = len(a)
    obs = _micro_f1(*a.sum(axis=0)) - _micro_f1(*b.sum(axis=0)) if n else 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n)) if n else np.zeros((n_resamples, 0), int)

    def pooled(arr: np.ndarray) -> np.ndarray:
        s = arr[idx].sum(axis=1)
        d = 2 * s[:, 0] + s[:, 1] + s[:, 2]
        return np.where(d > 0, 2 * s[:, 0] / d, 0.0)

    res = pooled(a) - pooled(b) if n else np.zeros(n_resamples)
    return obs, res


def paired_micro_f1_diff_ci(
    counts_a, counts_b, arm_a: str, arm_b: str,
    n_resamples: int = 5000, seed: int = 42, confidence: float = 0.95,
) -> DifferenceEstimate:
    """Paired bootstrap CI on the corpus micro-F1 difference (arm_a − arm_b)."""
    obs, res = paired_micro_f1_resamples(counts_a, counts_b, n_resamples, seed)
    alpha = (1 - confidence) / 2
    return DifferenceEstimate(
        arm_a=arm_a, arm_b=arm_b, diff=float(obs),
        lower=float(np.quantile(res, alpha)),
        upper=float(np.quantile(res, 1 - alpha)),
        n_pairs=len(_counts_array(counts_a)),
    )


def paired_micro_f1_pvalue(
    counts_a, counts_b, n_resamples: int = 10000, seed: int = 42,
) -> float:
    """Two-sided bootstrap p-value for the paired micro-F1 difference (H0: diff=0)."""
    obs, res = paired_micro_f1_resamples(counts_a, counts_b, n_resamples, seed)
    centered = res - obs
    return float(min(1.0, max(0.0, np.mean(np.abs(centered) >= abs(obs)))))


# ---------------------------------------------------------------------------
# TOST equivalence (for "arm A is practically equivalent to arm B")
# ---------------------------------------------------------------------------

@dataclass
class EquivalenceResult:
    """Result of a two one-sided test for practical equivalence."""
    arm_a: str
    arm_b: str
    diff: float
    margin: float
    p_tost: float          # max of the two one-sided p-values
    lower: float           # (1 - 2*alpha) CI, the interval TOST checks
    upper: float
    equivalent: bool
    n_pairs: int | None = None

    def __str__(self) -> str:
        verdict = "EQUIVALENT" if self.equivalent else "not established"
        return (
            f"{self.arm_a} ≈ {self.arm_b}? {verdict} at ±{self.margin:g}: "
            f"diff={self.diff:+.4f}, {int(round((1 - 2 * 0.05) * 100))}% CI "
            f"[{self.lower:+.4f}, {self.upper:+.4f}], p_TOST={self.p_tost:.4g}"
        )


def tost_equivalence(
    scores_a: list[float],
    scores_b: list[float],
    margin: float,
    arm_a: str = "A",
    arm_b: str = "B",
    n_resamples: int = 10000,
    seed: int = 42,
    cluster_keys: list | None = None,
    alpha: float = 0.05,
) -> EquivalenceResult:
    """Two one-sided tests (Schuirmann) for equivalence of a paired mean diff.

    Equivalence at ±`margin` is declared when both one-sided nulls are rejected
    at `alpha` — equivalently, when the (1 − 2·alpha) bootstrap CI for the paired
    difference lies entirely inside (−margin, +margin). This is the correct claim
    for "A ≈ B"; a non-significant difference test does NOT establish equivalence.
    The two one-sided p-values are estimated from the bootstrap distribution
    re-centered at each margin boundary; `p_tost` is their max.
    """
    diff_obs, resamples = paired_bootstrap_resamples(
        scores_a, scores_b, n_resamples=n_resamples,
        seed=seed, cluster_keys=cluster_keys,
    )
    centered = resamples - diff_obs
    # H0_lower: diff <= -margin  (reject toward diff > -margin)
    p_lower = float(np.mean(centered >= diff_obs + margin))
    # H0_upper: diff >= +margin  (reject toward diff < +margin)
    p_upper = float(np.mean(centered <= diff_obs - margin))
    p_tost = float(min(1.0, max(p_lower, p_upper)))
    lo = float(np.quantile(resamples, alpha))
    hi = float(np.quantile(resamples, 1 - alpha))
    return EquivalenceResult(
        arm_a=arm_a, arm_b=arm_b, diff=float(diff_obs), margin=float(margin),
        p_tost=p_tost, lower=lo, upper=hi,
        equivalent=bool(lo > -margin and hi < margin),
        n_pairs=len(scores_a),
    )


