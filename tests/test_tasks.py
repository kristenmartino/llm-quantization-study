"""Unit tests for tasks.py — BIO span extraction, NER scoring, JSON parsing.

The NER scorer is the load-bearing function for the extraction half of the
study; getting it wrong would shift every NER number in the writeup. These
tests pin the contracts: span-level F1 with type-match, liberal alias parser
for quantization-drifted keys, tolerant JSON parsing of fenced output.
"""
from __future__ import annotations

import pytest

from tasks import (
    _extract_spans_from_bio,
    _safe_parse_json_array,
    score_extraction,
)


class TestExtractSpansFromBio:
    """CoNLL BIO indices: 0=O, 1=B-PER, 2=I-PER, 3=B-ORG, 4=I-ORG,
    5=B-LOC, 6=I-LOC, 7=B-MISC, 8=I-MISC."""

    def test_single_token_entities_separated_by_O(self):
        tokens = ["EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", "."]
        tags = [3, 0, 7, 0, 0, 0, 7, 0, 0]
        assert _extract_spans_from_bio(tokens, tags) == [
            ("EU", "ORG"), ("German", "MISC"), ("British", "MISC"),
        ]

    def test_multi_token_entity_joins_with_space(self):
        """B-LOC I-LOC I-LOC = single 'New York City' span."""
        tokens = ["Born", "in", "New", "York", "City", "."]
        tags = [0, 0, 5, 6, 6, 0]
        assert _extract_spans_from_bio(tokens, tags) == [("New York City", "LOC")]

    def test_adjacent_entities_of_different_types_split(self):
        """B-PER I-PER B-ORG with no intervening O → two separate spans."""
        tokens = ["Bill", "Gates", "Microsoft"]
        tags = [1, 2, 3]
        assert _extract_spans_from_bio(tokens, tags) == [
            ("Bill Gates", "PER"), ("Microsoft", "ORG"),
        ]

    def test_empty_input(self):
        assert _extract_spans_from_bio([], []) == []

    def test_all_O_tags(self):
        assert _extract_spans_from_bio(["the", "cat", "sat"], [0, 0, 0]) == []


class TestScoreExtraction:
    GOLD = [("EU", "ORG"), ("German", "MISC")]

    def test_perfect_match(self):
        out = '[{"text":"EU","type":"ORG"},{"text":"German","type":"MISC"}]'
        assert score_extraction(out, self.GOLD) == 1.0

    def test_type_mismatch_yields_half_f1(self):
        """1 TP + 1 FP + 1 FN → F1 = 2*1 / (2+1+1) = 0.5."""
        out = '[{"text":"EU","type":"LOC"},{"text":"German","type":"MISC"}]'
        assert score_extraction(out, self.GOLD) == pytest.approx(0.5)

    def test_empty_pred_with_nonempty_gold_scores_zero(self):
        assert score_extraction("[]", self.GOLD) == 0.0

    def test_both_empty_scores_one(self):
        """No entities expected, none predicted — perfect agreement."""
        assert score_extraction("[]", []) == 1.0

    def test_malformed_json_scores_zero(self):
        assert score_extraction("not json at all", self.GOLD) == 0.0

    def test_liberal_alias_parser_accepts_entity_label_keys(self):
        """Quantized arms drift on key naming — parser accepts text|entity|span
        and type|label|tag. The drift is itself a finding worth reporting."""
        out = '[{"entity":"EU","label":"ORG"},{"span":"German","tag":"MISC"}]'
        assert score_extraction(out, self.GOLD) == 1.0

    def test_markdown_fence_stripped(self):
        out = '```json\n[{"text":"EU","type":"ORG"},{"text":"German","type":"MISC"}]\n```'
        assert score_extraction(out, self.GOLD) == 1.0

    def test_case_normalization_on_span_and_type(self):
        """Spans lowercased + stripped, types uppered, before set comparison."""
        out = '[{"text":"eu","type":"org"},{"text":"GERMAN","type":"MISC"}]'
        assert score_extraction(out, self.GOLD) == 1.0


class TestSafeParseJsonArray:
    def test_clean_array(self):
        assert _safe_parse_json_array("[1, 2, 3]") == [1, 2, 3]

    def test_array_embedded_in_prose(self):
        """Real model outputs sometimes wrap arrays in narration — be tolerant."""
        assert _safe_parse_json_array("Here you go: [1, 2, 3] hope that helps") == [1, 2, 3]

    def test_fenced_array(self):
        assert _safe_parse_json_array("```json\n[1,2,3]\n```") == [1, 2, 3]

    def test_no_bracket_returns_none(self):
        assert _safe_parse_json_array("just prose") is None

    def test_malformed_returns_none(self):
        assert _safe_parse_json_array("[1, 2,") is None

    def test_object_not_array_returns_none(self):
        """Helper only accepts arrays; an object that parses validly is still rejected."""
        assert _safe_parse_json_array('{"key": "value"}') is None
