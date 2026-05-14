"""Statistics for the eval: confidence intervals and pairwise differences.

Two CI flavors:
  - Wilson score interval for binary outcomes (MMLU accuracy)
  - Bootstrap percentile interval for continuous outcomes (extraction F1)

The "headline" numbers in the writeup are not point estimates — they're effect
sizes with CIs. "Q4 loses 3.2pp accuracy [95% CI: 1.8-4.6pp] vs FP16" beats
"Q4 = 71.2%, FP16 = 74.4%" every time.
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


