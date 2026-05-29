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
    EquivalenceResult,
    bootstrap_interval,
    holm_adjust,
    mcnemar_test,
    micro_f1_interval,
    paired_bootstrap_diff_ci,
    paired_bootstrap_pvalue,
    paired_bootstrap_resamples,
    paired_micro_f1_diff_ci,
    paired_micro_f1_pvalue,
    tost_equivalence,
    wilson_interval,
    _micro_f1,
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


class TestMicroF1:
    """Corpus micro-F1: pooled TP/FP/FN, and its difference from macro F1."""

    def test_micro_f1_pooled_formula(self):
        # 2*TP / (2*TP + FP + FN)
        assert _micro_f1(10, 0, 0) == 1.0
        assert _micro_f1(0, 5, 5) == 0.0
        assert _micro_f1(5, 5, 0) == pytest.approx(2 * 5 / (2 * 5 + 5))
        assert _micro_f1(0, 0, 0) == 0.0  # nothing to score

    def test_micro_differs_from_macro_on_skewed_counts(self):
        # One entity-rich sentence scored perfectly, two empty sentences where the
        # model hallucinated one spurious entity each. Macro (mean per-sentence F1)
        # is dragged down hard by the two zeros; micro dilutes the 2 FP across the
        # large TP pool. This is the exact mechanism behind the study's 3.2 vs 5.0pp.
        counts = [(10, 0, 0), (0, 1, 0), (0, 1, 0)]  # (tp, fp, fn) per sentence
        micro = micro_f1_interval(counts, n_resamples=200).mean
        macro = (1.0 + 0.0 + 0.0) / 3
        assert micro > 0.9            # pooled: 20/22
        assert macro == pytest.approx(1 / 3)
        assert micro - macro > 0.5    # they genuinely diverge

    def test_micro_interval_brackets_point_and_is_ordered(self):
        counts = [(3, 1, 1)] * 50
        est = micro_f1_interval(counts, n_resamples=500)
        assert est.lower <= est.mean <= est.upper
        assert est.n == 50

    def test_paired_micro_identical_arms_zero_diff(self):
        counts = [(2, 1, 0), (0, 0, 1), (3, 0, 2)] * 20
        de = paired_micro_f1_diff_ci(counts, counts, "a", "b", n_resamples=500)
        assert de.diff == pytest.approx(0.0, abs=1e-9)
        assert de.lower <= 0 <= de.upper
        assert paired_micro_f1_pvalue(counts, counts, n_resamples=500) > 0.5

    def test_paired_micro_detects_real_gap(self):
        good = [(4, 0, 0)] * 60
        bad = [(2, 2, 2)] * 60
        de = paired_micro_f1_diff_ci(good, bad, "good", "bad", n_resamples=1000)
        assert de.diff > 0.3
        assert de.lower > 0  # CI excludes zero
        assert paired_micro_f1_pvalue(good, bad, n_resamples=1000) < 0.05

    def test_paired_micro_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            paired_micro_f1_diff_ci([(1, 0, 0)], [(1, 0, 0), (0, 1, 0)], "a", "b")


class TestTostEquivalence:
    """Two one-sided tests: equivalence is a positive claim, not non-significance."""

    def test_identical_arms_are_equivalent(self):
        scores = [1.0, 0.0, 1.0, 1.0, 0.0] * 40
        res = tost_equivalence(scores, scores, margin=0.05, arm_a="a", arm_b="b")
        assert isinstance(res, EquivalenceResult)
        assert res.equivalent is True
        assert res.p_tost < 0.05
        assert res.lower <= 0 <= res.upper

    def test_large_difference_is_not_equivalent(self):
        a = [1.0] * 100
        b = [0.0] * 100
        res = tost_equivalence(a, b, margin=0.05)
        assert res.equivalent is False
        assert res.p_tost > 0.05

    def test_tiny_difference_within_margin_is_equivalent(self):
        # ~0.2pp difference, well inside a ±1pp margin -> equivalent
        a = [1.0] * 500 + [0.0] * 500
        b = [1.0] * 499 + [0.0] * 501
        res = tost_equivalence(a, b, margin=0.01)
        assert res.equivalent is True

    def test_margin_too_tight_fails_equivalence(self):
        # Same small difference, but a margin tighter than the CI -> not established
        a = [1.0] * 500 + [0.0] * 500
        b = [1.0] * 480 + [0.0] * 520
        res = tost_equivalence(a, b, margin=0.005)
        assert res.equivalent is False

    def test_str_reports_verdict(self):
        scores = [0.6, 0.6, 0.6] * 50
        res = tost_equivalence(scores, scores, margin=0.05, arm_a="q8", arm_b="fp16")
        assert "q8" in str(res) and "fp16" in str(res)
