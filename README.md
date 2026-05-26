# Quantization's Causal Effect on LLM Output Quality

**A controlled experiment on Llama 3.1 8B Instruct: FP16 vs Q8 vs Q4**

Kristen Martino · [kristenmartino.ai](https://kristenmartino.ai) · [Repo](https://github.com/kristenmartino/llm-quantization-study)

[![tests](https://github.com/kristenmartino/llm-quantization-study/actions/workflows/test.yml/badge.svg)](https://github.com/kristenmartino/llm-quantization-study/actions/workflows/test.yml)

---

## TL;DR

**Ship Q8_0 as the default for Llama 3.1 8B Instruct.** It's statistically indistinguishable from FP16 on both tasks — MMLU accuracy (Δ=−0.2pp, 95% CI: −0.8 to +0.4) and CoNLL-2003 NER F1 (Δ=+0.0003, 95% CI: −0.006 to +0.007) — at ~1.8× the throughput and roughly half the memory footprint. The strongest evidence in this study is for Q8_0 ≈ FP16, not for any quantization-vs-quantization degradation.

**Reach for Q4_K_M only when memory or latency is binding *and* the workload doesn't include structured information extraction.** Q4_K_M loses 5.0pp F1 on NER (95% CI: 2.1–8.1pp; Holm-adjusted p=0.003) for an additional ~40% throughput (2.5× FP16 total) and an ~3× smaller memory footprint. The MMLU regression (1.6pp) does not reach significance. The failure mode: Q4 emits well-formed JSON at a slightly *higher* parse rate than FP16, but selects the wrong entities — the brittleness is semantic, not syntactic.

## Why this matters for PMs

Model quantization is a routine production decision: ship at FP16 and pay for the GPU memory, or ship at Q4 and absorb some quality loss for cheaper, faster inference. Most teams pick by feel. This study quantifies the cost/quality tradeoff on a single model family using the same prompts and sampling, so the only thing varying is precision. The output is a decision framework, not a benchmark leaderboard.

---

## Method

**Treatment.** Quantization level — three arms: FP16 (control), Q8_0, Q4_K_M. All three pulled from Ollama: `llama3.1:8b-instruct-{fp16|q8_0|q4_K_M}`.

**Tasks.**
- **MMLU (knowledge/reasoning).** Stratified subset of 500 questions across 10 subjects (STEM, professional, humanities). Mechanically scored: parse first A–D character from output, compare to gold.
- **Named entity recognition (CoNLL-2003).** 300 sentences from the test split → JSON array of (entity, type) pairs. Span-level F1 with type-match (canonical CoNLL metric); malformed JSON scored as 0. Liberal alias parser for `text|entity|span` and `type|label|tag` keys.

**Design.** Examples paired across arms: same items, same prompts, same seed (42). Arms run round-robin per example to debias inference timing against thermal/daemon drift; warmups for all arms precede the timed loop.

**Sampling.** Temperature = 0, fixed seed (42), identical prompts and system messages, max_tokens task-appropriate. Held constant across arms.

**Statistics.**
- MMLU: Wilson 95% CI per arm on accuracy. Paired pairwise differences via paired bootstrap, with cluster-bootstrap on subjects for the overall CI (within-subject correlation is real). p-values from McNemar's test.
- NER: Bootstrap percentile 95% CI per arm on span-F1; paired pairwise differences via paired bootstrap; p-values via paired bootstrap (two-sided, centered under H0).
- Multiple-comparison correction: Holm-Bonferroni applied per task across the 3 pairwise tests.

**Power.** At n=499 aligned per arm on MMLU, detects ≥3.8pp accuracy differences at 80% power, α=0.05 (paired diff SD = 0.30 observed). At n=300 per arm on NER, detects span-F1 differences ≥4.3pp (paired diff SD = 0.27 observed). Both post-hoc verified on the actual data. (1 MMLU example was dropped from alignment because the SHA1-hashed `example_id` correctly identified a true duplicate in `cais/mmlu`'s `high_school_mathematics` test split — the same word problem appears at rows 5 and 48 with shuffled answer choices but the same correct answer. Known MMLU data-hygiene artifact, not a harness bug.)

**Hardware.** MacBook Pro (Apple M4 Max, 14-core CPU / 32-core GPU, 36 GB unified memory), Ollama 0.22.1, llama.cpp Metal backend.

---

## Results

### Headline numbers

| Arm     | MMLU accuracy [95% CI]    | NER F1 (CoNLL) [95% CI]   | tok/sec (MMLU / NER) | Memory¹  |
|---------|----------------------------|----------------------------|----------------------|----------|
| FP16    | 0.585 [0.541, 0.628]       | 0.614 [0.570, 0.662]       | 40.9 / 20.7          | ~16.0 GB |
| Q8_0    | 0.587 [0.544, 0.630]       | 0.614 [0.568, 0.661]       | 72.7 / 38.8          | ~8.5 GB  |
| Q4_K_M  | 0.569 [0.525, 0.612]       | 0.564 [0.518, 0.611]       | 102.9 / 51.3         | ~4.9 GB  |

¹ Model footprint as reported by Ollama (`ollama show`). On Apple Silicon this is unified memory, not discrete VRAM; runtime usage adds KV cache and activation overhead. On a 36 GB M4 Max, FP16 is the largest model that comfortably runs alongside other workloads — the memory column matters as much as the throughput column.

### Pairwise effect sizes (paired CIs; Holm-adjusted within task)

| Comparison        | MMLU Δ accuracy [95% CI; p_adj]    | NER Δ F1 [95% CI; p_adj]                |
|-------------------|--------------------------------------|-------------------------------------------|
| FP16 − Q8_0       | −0.002 [−0.008, +0.004]; p_adj=1.00  | +0.000 [−0.006, +0.007]; p_adj=0.92       |
| FP16 − Q4_K_M     | +0.016 [−0.006, +0.036]; p_adj=0.70  | **+0.050 [+0.021, +0.081]; p_adj=0.003*** |
| Q8_0 − Q4_K_M     | +0.018 [−0.002, +0.038]; p_adj=0.70  | **+0.050 [+0.020, +0.081]; p_adj=0.003*** |

A CI that doesn't cross zero is a statistically detectable effect at the 95% level; significance markers (`*`) use the Holm-adjusted p-value within each task's 3-test family at α=0.05. **Significant comparisons:** FP16 − Q4_K_M and Q8_0 − Q4_K_M on NER (both p_adj=0.003). All MMLU pairs are NS after Holm. The FP16 − Q8_0 contrasts are essentially zero on both tasks — the strongest evidence in the study is for Q8_0 ≈ FP16.

### Quality vs. cost frontier

![Pareto frontier: quality vs. inference speed across arms](results/pareto.png)

Q8_0 sits cleanly on the Pareto frontier — same accuracy as FP16, ~1.8× the throughput. Q4_K_M earns an additional ~40% throughput (2.5× FP16) but pays for it on NER. The failure mode is semantic, not syntactic: Q4 actually has the highest JSON parse rate of the three arms (99.7%, vs. 99.0% FP16 and 98.7% Q8_0), but it produces 95 well-formed-but-zero-F1 outputs vs. 76 for FP16 — Q4 emits valid JSON arrays that name the wrong entities. The brittleness shows up in entity selection, not in structural compliance.

### Where the effect concentrates

The Q4 vs. FP16 gap on MMLU is concentrated in `miscellaneous` (−6.0pp) and `professional_medicine` (−6.0pp), with smaller regressions in `moral_scenarios`, `abstract_algebra`, and `machine_learning` (each −4.0pp). Several subjects move the other way — `professional_law` (+4.0pp), `college_computer_science` (+2.0pp), `high_school_us_history` (+2.0pp) — within the noise band for n=50/subject. None of the per-subject pairwise diffs survives Holm correction within the per-subject family of 3 tests, so this is suggestive heterogeneity worth flagging, not replicable evidence of subject-specific quantization sensitivity. The pattern is consistent with the published intuition that quantization hits broad-knowledge recall harder than narrow reasoning, but a properly powered subject-level study would need ~3× the per-subject sample size.

*Calibration against the literature.* Published Llama-family Q4 quantization evaluations (GPTQ, AWQ, QLoRA papers; the Ollama / llama.cpp release notes) typically report low-single-digit percentage-point degradation on standard benchmarks. The 5pp NER drop sits at the higher end of that band, consistent with structured-output tasks stressing quantization harder than free-form generation or multiple-choice classification. The MMLU result (1.6pp NS) is firmly within the published range.

---

## A decision framework

Based on these results, here's the call I'd make as a PM choosing a quantization level for a feature:

**Pick Q8_0 (the default):**
- Same measured quality as FP16 on both tasks, at ~1.8× throughput. The strongest evidence in the study is for Q8_0 ≈ FP16.
- Right call when you need a single model serving heterogeneous workloads.
- Right call when reasoning depth matters (multi-step, legal, medical) — MMLU shows Q8 holds.

**Pick Q4_K_M only when:**
- Throughput is the binding constraint and a measurable F1 hit on structured extraction is acceptable downstream (human review, retry logic, low-stakes outputs).
- Memory is the binding constraint: Q4_K_M's ~5 GB footprint vs. FP16's ~16 GB matters on memory-limited machines (laptops, edge, or multi-tenant serving).

> Avoid Q4_K_M for structured information extraction. The 5pp NER gap is significant, and Q4 emits well-formed JSON with the wrong entities — harder to detect downstream than malformed output.

**Stay at FP16 when:**
- Accuracy is the binding constraint (regulated outputs, safety-critical, eval ground truth).
- The throughput delta from Q8 (~1.8×) doesn't move unit economics — inference cost isn't the constraint.

A non-obvious finding: the published intuition that "Q4 is fine for pattern-matching, hurts on reasoning" doesn't hold here. NER is a structured pattern-extraction task, and it's where Q4 visibly degrades. MMLU (more reasoning-heavy) shows a numerically smaller Q4 regression that doesn't reach significance. The mechanism worth pointing at: Q4 is sensitive to *output structure fidelity*, not problem difficulty.

---

## Limitations

- **Single model family.** Results may not generalize to other architectures (Mistral, Qwen, Gemma) or to the smaller/larger Llama 3.1 variants. Quantization-friendliness varies.
- **English only.** No multilingual eval; quantization effects on non-English tokens can differ.
- **Deterministic sampling.** Temperature=0 measures the model, not the user-facing distribution. Robustness check at temperature=0.7 would strengthen claims about production behavior.
- **No fine-tuned variants.** Domain-tuned models may behave differently under quantization than the base instruct.
- **MMLU is not the world.** It correlates with reasoning quality but doesn't measure tool use, code generation, or long-context behavior.
- **Sample size on NER.** n=300 is modest for span-F1 estimation; CIs are wider than I'd want for a production decision on this task.

---

## Reproducibility

Code, prompts, per-example raw outputs, and the analysis script are in this repo. The harness is ~1,000 LOC of Python: a deterministic Ollama wrapper, two task loaders (MMLU from HuggingFace `cais/mmlu`, NER from `eriktks/conll2003`), paired bootstrap + Wilson CIs + McNemar + Holm in pure scipy, a round-robin runner, and an analyze step that emits the Pareto plot. The `results/` directory ships with the n=2,400 raw outputs backing every number above.

```
ollama_client.py       # Thin wrapper around Ollama with deterministic sampling
tasks.py               # Task loaders (MMLU subset, CoNLL-2003 NER)
scoring.py             # Wilson + bootstrap CIs, paired bootstrap, McNemar, Holm
run_eval.py            # Main runner: arms × tasks → JSONL (round-robin schedule)
analyze.py             # Paired diffs, Holm-adjusted p-values, per-subject breakdown, Pareto plot
tests/                 # pytest suite for scoring.py and tasks.py (45 tests)
results/               # Per-run JSONL outputs, summary.{json,txt}, pareto.png, run logs
requirements.txt       # Runtime deps
requirements-dev.txt   # Adds pytest for the test suite
pyproject.toml         # pytest config
LICENSE                # MIT
README.md              # This document — the study writeup
```

To run the tests:

```bash
pip install -r requirements-dev.txt
pytest
```

**Prerequisites:** Python 3.9+ and [Ollama](https://ollama.com) installed locally. The `--task extraction` CLI choice runs the CoNLL-2003 NER eval — the argument name predates the swap from a synthetic structured-extraction task to NER and is retained for filename compatibility with `results/extraction_*.jsonl`.

To reproduce:

```bash
git clone https://github.com/kristenmartino/llm-quantization-study
cd llm-quantization-study
pip install -r requirements.txt
ollama pull llama3.1:8b-instruct-{fp16,q8_0,q4_K_M}
python run_eval.py --task mmlu --n 500
python run_eval.py --task extraction --n 300   # CoNLL-2003 NER
python analyze.py
```

Total LLM runtime: ~5.8 hours of generation across 3 arms × (500 MMLU + 300 NER) examples, mostly unattended. MMLU runs at ~3h22m wall, NER at ~2h35m.

### Data sources

- **MMLU** (`cais/mmlu`) — MIT license.
- **CoNLL-2003** (`eriktks/conll2003`) — Reuters/RCV1 Data Use Agreement; research use only. Acknowledgment: Tjong Kim Sang & De Meulder (CoNLL 2003).
