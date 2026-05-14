"""Unit tests for scoring.py.

Covers the statistical primitives the study relies on:
- Wilson and bootstrap CIs for per-arm marginals.
- Paired bootstrap CI and p-value for pairwise differences (with cluster option).
- McNemar's test for paired binary outcomes (exact + chi-square branches).
- Holm-Bonferroni step-down correction (preserves input order).

These are the same checks I ran ad-hoc during the build; promoting them to
committed tests so any reviewer can re-verify the math without rerunning the
~6h experiment.
"""
from __future__ import annotations

import numpy as np
import pytest

from scoring import (
    DifferenceEstimate,
    bootstrap_interval,
    holm_adjust,
    mcnemar_test,
    paired_bootstrap_diff_ci,
    paired_bootstrap_pvalue,
    paired_bootstrap_resamples,
    wilson_interval,
)


class TestWilsonInterval:
    def test_empty_input_returns_zero_interval(self):
        ci = wilson_interval([])
        assert ci.n == 0 and ci.mean == 0.0

    def test_all_correct_pulls_lower_below_one(self):
        """At p=1 the Wilson interval has finite width — unlike the Wald CI which collapses."""
        ci = wilson_interval([1.0] * 100)
        assert ci.mean == 1.0
        assert ci.lower < 1.0
        assert ci.upper == pytest.approx(1.0, abs=1e-9)

    def test_half_correct_brackets_one_half(self):
        ci = wilson_interval([1.0] * 50 + [0.0] * 50)
        assert ci.mean == pytest.approx(0.5)
        assert ci.lower < 0.5 < ci.upper


class TestBootstrapInterval:
    def test_empty_input_returns_zero_interval(self):
        ci = bootstrap_interval([])
        assert ci.n == 0

    def test_constant_input_collapses_ci(self):
        ci = bootstrap_interval([0.42] * 50, n_resamples=500, seed=42)
        assert ci.mean == pytest.approx(0.42)
        assert ci.lower == pytest.approx(0.42)
        assert ci.upper == pytest.approx(0.42)

    def test_seed_determinism(self):
        a = bootstrap_interval([0.1, 0.2, 0.3, 0.5, 0.9], n_resamples=500, seed=7)
        b = bootstrap_interval([0.1, 0.2, 0.3, 0.5, 0.9], n_resamples=500, seed=7)
        assert (a.lower, a.upper) == (b.lower, b.upper)


class TestHolmAdjust:
    def test_empty_input(self):
        assert holm_adjust([]) == []

    def test_single_test_unchanged(self):
        """With m=1 there's nothing to correct."""
        assert holm_adjust([0.04]) == [0.04]

    def test_canonical_sorted_input(self):
        """[0.01, 0.02, 0.03] → [0.03, 0.04, 0.04] (monotonic max enforced)."""
        got = holm_adjust([0.01, 0.02, 0.03])
        assert got == pytest.approx([0.03, 0.04, 0.04])

    def test_preserves_input_order_under_unsorted_input(self):
        """Adjusted p-values map back to input position, not sort rank."""
        got = holm_adjust([0.03, 0.01, 0.02])
        # Sorted: 0.01 (orig idx 1), 0.02 (orig idx 2), 0.03 (orig idx 0)
        # rank 0 → 3*0.01 = 0.03 → out[1] = 0.03
        # rank 1 → 2*0.02 = 0.04 → out[2] = 0.04
        # rank 2 → 1*0.03 = 0.03, running max = 0.04 → out[0] = 0.04
        assert got == pytest.approx([0.04, 0.03, 0.04])

    def test_caps_at_one(self):
        """A test with p > 1/m would otherwise give adjusted p > 1; we cap at 1.0."""
        got = holm_adjust([0.5, 0.6, 0.9])
        assert all(p <= 1.0 for p in got)


class TestMcnemarTest:
    def test_no_discordants_returns_p_one(self):
        stat, p = mcnemar_test([1.0, 1.0, 0.0], [1.0, 1.0, 0.0])
        assert stat == 0.0
        assert p == 1.0

    def test_exact_binomial_small_discordants(self):
        """n01=10, n10=0 (discordant ≤ 25) → exact two-sided p = 2 * 0.5^10."""
        scores_a = [1.0] * 10 + [1.0] * 30 + [0.0] * 30
        scores_b = [0.0] * 10 + [1.0] * 30 + [0.0] * 30
        _, p = mcnemar_test(scores_a, scores_b)
        assert p == pytest.approx(2 * 0.5**10, abs=1e-6)

    def test_chi_square_large_discordants(self):
        """n01=40, n10=10 (discordant=50 > 25) → Yates: (|40-10|-1)^2 / 50 = 16.82."""
        scores_a = [1.0] * 40 + [0.0] * 10 + [1.0] * 100 + [0.0] * 100
        scores_b = [0.0] * 40 + [1.0] * 10 + [1.0] * 100 + [0.0] * 100
        stat, p = mcnemar_test(scores_a, scores_b)
        assert stat == pytest.approx(16.82, abs=0.1)
        assert p < 0.001

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            mcnemar_test([1.0, 0.0], [1.0, 0.0, 1.0])


class TestPairedBootstrap:
    def test_identical_inputs_zero_diff(self):
        scores = [0.5] * 100
        diff, resamples = paired_bootstrap_resamples(
            scores, scores, n_resamples=500, seed=42
        )
        assert diff == pytest.approx(0.0)
        assert np.allclose(resamples, 0.0)

    def test_constant_diff_collapses_resamples(self):
        diff, resamples = paired_bootstrap_resamples(
            [0.7] * 100, [0.5] * 100, n_resamples=500, seed=42
        )
        assert diff == pytest.approx(0.2)
        assert np.allclose(resamples, 0.2)

    def test_diff_ci_excludes_zero_for_real_effect(self):
        de = paired_bootstrap_diff_ci(
            [0.7] * 100, [0.5] * 100, "a", "b",
            n_resamples=500, seed=42,
        )
        assert de.diff == pytest.approx(0.2)
        assert de.lower > 0
        assert de.n_pairs == 100

    def test_pvalue_small_for_constant_diff(self):
        p = paired_bootstrap_pvalue(
            [0.7] * 100, [0.5] * 100, n_resamples=500, seed=42
        )
        assert p < 0.01

    def test_pvalue_one_for_identical(self):
        p = paired_bootstrap_pvalue(
            [0.5] * 100, [0.5] * 100, n_resamples=500, seed=42
        )
        assert p == pytest.approx(1.0)

    def test_cluster_bootstrap_preserves_observed_diff(self):
        """Cluster-bootstrap (resample subjects, take all examples within) should
        still recover the observed mean diff."""
        a = [0.7] * 50 + [0.6] * 50
        b = [0.5] * 50 + [0.4] * 50
        keys = ["math"] * 50 + ["law"] * 50
        de = paired_bootstrap_diff_ci(
            a, b, "a", "b", n_resamples=500, seed=42, cluster_keys=keys,
        )
        assert de.diff == pytest.approx(0.2)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            paired_bootstrap_resamples([1.0, 0.0], [1.0, 0.0, 1.0])

    def test_cluster_keys_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            paired_bootstrap_resamples(
                [1.0, 0.0], [1.0, 0.0], cluster_keys=["a", "b", "c"]
            )


class TestDifferenceEstimate:
    def test_default_optional_fields_are_none(self):
        de = DifferenceEstimate(arm_a="a", arm_b="b", diff=0.1, lower=0.05, upper=0.15)
        assert de.p_value is None
        assert de.p_adj is None
        assert de.lower_adj is None
        assert de.upper_adj is None
        assert de.n_pairs is None

    def test_str_shows_adjusted_block_when_present(self):
        de = DifferenceEstimate(
            arm_a="a", arm_b="b", diff=0.1, lower=0.05, upper=0.15,
            p_value=0.01, p_adj=0.03, lower_adj=0.02, upper_adj=0.18, n_pairs=200,
        )
        s = str(de)
        assert "p_adj=0.03" in s
        assert "*" in s  # significant under adjusted α=0.05

    def test_str_falls_back_to_ci_marker_without_p_adj(self):
        # No p_adj: significance keys off CI not crossing zero
        sig = DifferenceEstimate(arm_a="a", arm_b="b", diff=0.1, lower=0.05, upper=0.15)
        assert "*" in str(sig)

        not_sig = DifferenceEstimate(arm_a="a", arm_b="b", diff=0.01, lower=-0.05, upper=0.07)
        assert "*" not in str(not_sig)
