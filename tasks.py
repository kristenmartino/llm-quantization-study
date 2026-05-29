"""Task definitions for the quantization eval.

Two tasks ship by default:

  - MMLU subset: knowledge/reasoning, 4-way multiple choice, mechanically scored.
    Stratified across subjects to avoid topic concentration.

  - CoNLL-2003 NER: extract named entities (PER, ORG, LOC, MISC) from news
    sentences. Span-level exact match with type agreement. `score_extraction`
    returns per-example (macro) F1; `score_extraction_counts` returns the
    (tp, fp, fn) needed for the canonical corpus-level micro-F1 computed in
    analyze.py. Real-world structured extraction; closer to production use cases.

Each task returns a list of `Example` records and provides a `score` function
that maps (model_output, gold) -> a float (or 0/1 for binary tasks).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Callable

# datasets is the HuggingFace library; install via `pip install datasets`
try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover
    load_dataset = None  # type: ignore


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

@dataclass
class Example:
    """A single eval item."""
    id: str
    prompt: str
    gold: Any                 # type depends on task — str for MMLU, list[(span, type)] for NER
    metadata: dict            # subject, source, etc. — for stratified analysis later


@dataclass
class Task:
    """A complete task spec the runner can execute."""
    name: str
    examples: list[Example]
    system_prompt: str
    max_tokens: int
    score_fn: Callable[[str, Any], float]


# ---------------------------------------------------------------------------
# MMLU
# ---------------------------------------------------------------------------

# A curated subset that spans STEM, humanities, social science, and applied
# professional subjects. Drop or extend as needed — keeping these constant
# across arms is what matters.
MMLU_SUBJECTS = [
    "high_school_mathematics",
    "college_computer_science",
    "professional_medicine",
    "professional_law",
    "moral_scenarios",
    "high_school_us_history",
    "econometrics",
    "machine_learning",
    "miscellaneous",
    "abstract_algebra",
]

MMLU_SYSTEM = (
    "You are answering a multiple-choice question. Respond with exactly one "
    "letter: A, B, C, or D. Do not include any other text."
)

MMLU_TEMPLATE = """Question: {question}

A) {a}
B) {b}
C) {c}
D) {d}

Answer:"""


def load_mmlu(n: int = 500, seed: int = 42) -> Task:
    """Load a stratified MMLU subset of size `n`.

    Stratification: equal counts per subject (per_subject = n // len(SUBJECTS)).
    If `n` doesn't divide evenly into len(MMLU_SUBJECTS), the remainder is
    distributed round-robin across subjects (not piled onto the first one), so
    representation stays balanced for any `n`.

    Each example carries a row-unique id (`{subject}::{row_index}::{qhash}`) plus
    a `content_hash` over (question, choices, answer). The row index guarantees
    two distinct source rows can never collapse to one id; the content_hash lets
    analyze.py detect and report genuine duplicate items explicitly rather than
    silently merging them on a question-text hash collision. (cais/mmlu's
    high_school_mathematics test split contains one true duplicate question.)
    """
    if load_dataset is None:
        raise RuntimeError("Install `datasets`: pip install datasets")

    rng = random.Random(seed)
    per_subject = n // len(MMLU_SUBJECTS)
    remainder = n - (per_subject * len(MMLU_SUBJECTS))

    examples: list[Example] = []
    leftovers: dict[str, list[tuple[int, dict]]] = {}
    for subject in MMLU_SUBJECTS:
        ds = load_dataset("cais/mmlu", subject, split="test")
        rows = list(enumerate(ds))  # keep the original split row index
        rng.shuffle(rows)
        for row_index, row in rows[:per_subject]:
            examples.append(_mmlu_row_to_example(row, subject, row_index))
        leftovers[subject] = rows[per_subject:]

    # Distribute any remainder round-robin across subjects' leftover pools.
    while remainder > 0:
        progress = False
        for subject in MMLU_SUBJECTS:
            if remainder == 0:
                break
            pool = leftovers[subject]
            if pool:
                row_index, row = pool.pop()
                examples.append(_mmlu_row_to_example(row, subject, row_index))
                remainder -= 1
                progress = True
        if not progress:
            break  # all subject pools exhausted (n exceeds available rows)

    rng.shuffle(examples)

    return Task(
        name="mmlu",
        examples=examples,
        system_prompt=MMLU_SYSTEM,
        max_tokens=8,
        score_fn=score_mmlu,
    )


def _mmlu_row_to_example(row: dict, subject: str, row_index: int) -> Example:
    choices = row["choices"]
    answer_idx = row["answer"]  # 0-3
    answer_letter = "ABCD"[answer_idx]
    prompt = MMLU_TEMPLATE.format(
        question=row["question"],
        a=choices[0],
        b=choices[1],
        c=choices[2],
        d=choices[3],
    )
    qhash = hashlib.sha1(row["question"].encode("utf-8")).hexdigest()[:8]
    # content_hash fingerprints the full item (question + choices + answer) so
    # genuine duplicates can be detected explicitly; the id keys on row_index so
    # a question-text hash collision can never silently drop a distinct row.
    content_src = row["question"] + "||" + "|".join(map(str, choices)) + "||" + str(answer_idx)
    content_hash = hashlib.sha1(content_src.encode("utf-8")).hexdigest()[:12]
    return Example(
        id=f"{subject}::{row_index}::{qhash}",
        prompt=prompt,
        gold=answer_letter,
        metadata={"subject": subject, "row_index": row_index, "content_hash": content_hash},
    )


def score_mmlu(output: str, gold: str) -> float:
    """Parse model output for a letter A-D. Returns 1.0 if it matches gold."""
    # Strip whitespace and look for the first A/B/C/D character
    cleaned = output.strip().upper()
    for ch in cleaned:
        if ch in "ABCD":
            return 1.0 if ch == gold else 0.0
    return 0.0  # No valid answer found = wrong


# ---------------------------------------------------------------------------
# CoNLL-2003 NER
# ---------------------------------------------------------------------------

# CoNLL-2003 BIO tag indices: 0=O, 1=B-PER, 2=I-PER, 3=B-ORG, 4=I-ORG,
# 5=B-LOC, 6=I-LOC, 7=B-MISC, 8=I-MISC
_CONLL_TAG_TYPES = {1: "PER", 2: "PER", 3: "ORG", 4: "ORG",
                    5: "LOC", 6: "LOC", 7: "MISC", 8: "MISC"}
_CONLL_BEGIN_TAGS = {1, 3, 5, 7}  # B-* tags

EXTRACTION_SYSTEM = (
    "You extract named entities from text. Respond with ONLY a JSON array. "
    "Each element must be an object with two keys: \"text\" (the exact entity "
    "span as it appears in the input) and \"type\" (one of: PER, ORG, LOC, "
    "MISC). No markdown, no commentary, no code fences. If no entities are "
    "present, return []."
)

EXTRACTION_TEMPLATE = """Extract every named entity in the sentence below. Use these types only:
  PER  - person name
  ORG  - organization (company, agency, team)
  LOC  - geographic location (city, country, region)
  MISC - other named entity (nationality, event, work title)

Sentence: {sentence}

JSON:"""


def load_extraction(n: int = 300, seed: int = 42) -> Task:
    """Load CoNLL-2003 NER eval examples from HuggingFace.

    Samples `n` examples from the test split using a deterministic
    `random.Random(seed).sample(...)` over a sorted index list — bypasses HF's
    shuffle for bit-exact reproducibility across `datasets` versions. Gold is a
    list of (span_text, type) pairs reconstructed from BIO tags.
    """
    if load_dataset is None:
        raise RuntimeError("Install `datasets`: pip install datasets")

    ds = load_dataset("eriktks/conll2003", split="test", trust_remote_code=True)
    indices = sorted(range(len(ds)))
    rng = random.Random(seed)
    chosen = rng.sample(indices, min(n, len(indices)))

    examples: list[Example] = []
    for idx in chosen:
        row = ds[idx]
        tokens: list[str] = row["tokens"]
        tags: list[int] = row["ner_tags"]
        sentence = " ".join(tokens)
        gold_spans = _extract_spans_from_bio(tokens, tags)
        examples.append(
            Example(
                id=f"conll2003::{idx}",
                prompt=EXTRACTION_TEMPLATE.format(sentence=sentence),
                gold=gold_spans,
                metadata={"source": "conll2003", "row_index": idx},
            )
        )

    return Task(
        name="extraction",
        examples=examples,
        system_prompt=EXTRACTION_SYSTEM,
        max_tokens=512,
        score_fn=score_extraction,
    )


def _extract_spans_from_bio(tokens: list[str], tags: list[int]) -> list[tuple[str, str]]:
    """Convert BIO-tagged tokens to a list of (span_text, type) tuples."""
    spans: list[tuple[str, str]] = []
    cur_tokens: list[str] = []
    cur_type: str | None = None
    for tok, tag in zip(tokens, tags):
        ent_type = _CONLL_TAG_TYPES.get(tag)
        is_begin = tag in _CONLL_BEGIN_TAGS
        if is_begin or (ent_type is not None and ent_type != cur_type):
            # Flush any open span before starting a new one
            if cur_tokens and cur_type is not None:
                spans.append((" ".join(cur_tokens), cur_type))
            cur_tokens = [tok]
            cur_type = ent_type
        elif ent_type is not None:
            # Continuation of current span
            cur_tokens.append(tok)
        else:
            # O tag — flush
            if cur_tokens and cur_type is not None:
                spans.append((" ".join(cur_tokens), cur_type))
            cur_tokens = []
            cur_type = None
    if cur_tokens and cur_type is not None:
        spans.append((" ".join(cur_tokens), cur_type))
    return spans


def _span_sets(output: str, gold: list) -> tuple[set, set, bool]:
    """Parse output into (gold_set, pred_set, parse_ok) of (span_lower, TYPE) pairs."""
    pred = _safe_parse_json_array(output)
    parse_ok = pred is not None
    pred_spans = _normalize_pred_spans(pred) if parse_ok else []
    gold_set = set((str(span).lower().strip(), str(typ).upper())
                   for span, typ in gold)
    pred_set = set((str(span).lower().strip(), str(typ).upper())
                   for span, typ in pred_spans)
    return gold_set, pred_set, parse_ok


def score_extraction(output: str, gold: list) -> float:
    """Per-example (macro) span-level F1 with type-match required.

    Per-example: 2*TP / (2*TP + FP + FN). TP = exact (span_text, type) match.
    Malformed JSON output → F1 of 0 (a real failure mode worth measuring, not
    an error to skip). This is the *per-sentence* estimand; analyze.py averages
    it (macro F1). The canonical corpus-level micro-F1 is computed separately
    from `score_extraction_counts` by pooling TP/FP/FN across the corpus.
    """
    gold_set, pred_set, parse_ok = _span_sets(output, gold)
    if not parse_ok:
        return 0.0
    if not gold_set and not pred_set:
        return 1.0  # both empty — perfect agreement
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    if tp == 0:
        return 0.0
    return 2 * tp / (2 * tp + fp + fn)


def score_extraction_counts(output: str, gold: list) -> dict:
    """Per-example (tp, fp, fn) plus parse_status, for corpus-level micro-F1.

    Unlike `score_extraction` (which returns a per-sentence F1 and scores any
    malformed JSON as 0), this returns raw counts: a malformed output predicts
    no spans, so its gold entities become false negatives and it contributes no
    false positives — the standard conlleval treatment when pooling across the
    corpus. analyze.py sums these and computes one micro-F1 (the benchmark
    metric); the macro path keeps the parse-failure-is-zero convention.
    """
    gold_set, pred_set, parse_ok = _span_sets(output, gold)
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return {"tp": tp, "fp": fp, "fn": fn,
            "parse_status": "ok" if parse_ok else "malformed"}


def _normalize_pred_spans(pred: list) -> list[tuple[str, str]]:
    """Coerce model-emitted entity list into [(span_text, type), ...].

    Liberal in accepted key names ('text'|'entity'|'span', 'type'|'label'|'tag')
    because quantized arms drift on key naming — that drift is itself a finding
    to report, not something to crash on.
    """
    out: list[tuple[str, str]] = []
    for item in pred:
        if not isinstance(item, dict):
            continue
        span = item.get("text") or item.get("entity") or item.get("span")
        typ = item.get("type") or item.get("label") or item.get("tag")
        if span is None or typ is None:
            continue
        out.append((str(span), str(typ)))
    return out


def _safe_parse_json_array(text: str) -> list | None:
    """Try to parse a JSON array, tolerating common LLM output quirks.

    Strips markdown code fences and finds the first '[' / last ']' before
    parsing. Returns None on failure.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text[3:]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, list) else None
    except json.JSONDecodeError:
        return None
