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
    _mmlu_row_to_example,
    _safe_parse_json_array,
    score_extraction,
    score_extraction_counts,
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


class TestScoreExtractionCounts:
    """Counts feed the canonical corpus micro-F1; they differ from macro on malformed."""

    def test_perfect_match_counts(self):
        gold = [("Alice", "PER"), ("Google", "ORG")]
        out = '[{"text": "Alice", "type": "PER"}, {"text": "Google", "type": "ORG"}]'
        c = score_extraction_counts(out, gold)
        assert (c["tp"], c["fp"], c["fn"]) == (2, 0, 0)
        assert c["parse_status"] == "ok"

    def test_over_extraction_is_false_positive(self):
        # Gold empty, model hallucinates one entity: the empty-sentence failure mode.
        c = score_extraction_counts('[{"text": "Reuters", "type": "ORG"}]', [])
        assert (c["tp"], c["fp"], c["fn"]) == (0, 1, 0)

    def test_malformed_counts_gold_as_false_negatives(self):
        # Malformed JSON predicts no spans: gold -> FN, no FP (conlleval pooling),
        # which is WHY micro and macro diverge (macro scores this 0; micro pools).
        gold = [("Alice", "PER")]
        c = score_extraction_counts("not json at all", gold)
        assert (c["tp"], c["fp"], c["fn"]) == (0, 0, 1)
        assert c["parse_status"] == "malformed"
        # macro scores the same malformed output as a flat 0.0
        assert score_extraction("not json at all", gold) == 0.0

    def test_partial_overlap(self):
        gold = [("Alice", "PER"), ("Bob", "PER")]
        out = '[{"text": "Alice", "type": "PER"}, {"text": "Carol", "type": "PER"}]'
        c = score_extraction_counts(out, gold)
        assert (c["tp"], c["fp"], c["fn"]) == (1, 1, 1)


class TestMmluExampleId:
    """example_id must be row-unique so two distinct source rows never collapse."""

    def _row(self, question, choices, answer):
        return {"question": question, "choices": choices, "answer": answer}

    def test_id_includes_row_index_and_metadata(self):
        ex = _mmlu_row_to_example(self._row("2+2?", ["3", "4", "5", "6"], 1), "algebra", 7)
        assert ex.id.startswith("algebra::7::")
        assert ex.metadata["row_index"] == 7
        assert ex.metadata["subject"] == "algebra"
        assert "content_hash" in ex.metadata
        assert ex.gold == "B"

    def test_same_question_different_rows_get_distinct_ids(self):
        # The real cais/mmlu failure: identical question text at two row indices.
        # Question-only hashing collapsed them; row_index keeps them distinct.
        r = self._row("What is 6 times 12?", ["60", "72", "84", "66"], 1)
        a = _mmlu_row_to_example(r, "high_school_mathematics", 5)
        b = _mmlu_row_to_example(r, "high_school_mathematics", 48)
        assert a.id != b.id
        # ...but the content_hash matches, enabling explicit duplicate detection
        assert a.metadata["content_hash"] == b.metadata["content_hash"]

    def test_shuffled_choices_change_content_hash(self):
        r1 = self._row("Q?", ["a", "b", "c", "d"], 0)
        r2 = self._row("Q?", ["b", "a", "c", "d"], 1)  # same answer value, shuffled
        h1 = _mmlu_row_to_example(r1, "s", 0).metadata["content_hash"]
        h2 = _mmlu_row_to_example(r2, "s", 1).metadata["content_hash"]
        assert h1 != h2
