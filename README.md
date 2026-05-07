# Quantization's Causal Effect on LLM Output Quality

A controlled experiment measuring how quantization precision (FP16 → Q8 → Q4) affects task performance on Llama 3.1 8B Instruct, served locally via Ollama.

## What this is

A portfolio study examining the cost/quality tradeoff of quantization as a PM decision. Same model weights, same prompts, same sampling — only precision varies. Results reported with effect sizes and confidence intervals, paired with measured inference cost per arm.

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pull models (one-time, ~30GB total)
ollama pull llama3.1:8b-instruct-fp16
ollama pull llama3.1:8b-instruct-q8_0
ollama pull llama3.1:8b-instruct-q4_K_M

# 3. Run evals (writes JSONL results to ./results/)
python run_eval.py --task mmlu --n 500
python run_eval.py --task extraction --n 300

# 4. Analyze and generate plots
python analyze.py
```

## Repo structure

```
ollama_client.py    # Thin wrapper around Ollama with deterministic sampling
tasks.py            # Task loaders (MMLU subset, CoNLL-2003 NER)
scoring.py          # Wilson + bootstrap CIs, paired bootstrap, McNemar, Holm
run_eval.py         # Main runner: arms × tasks → JSONL (round-robin schedule)
analyze.py          # Paired diffs, Holm-adjusted p-values, per-subject breakdown, Pareto plot
results/            # Per-run JSONL outputs (gitignored)
one_pager.md        # The deliverable
```

## Method summary

- **Model:** `llama3.1:8b-instruct` at three precision arms: FP16, Q8_0, Q4_K_M
- **Tasks:** MMLU subset (n=500, 10 subjects stratified, multiple choice) + CoNLL-2003 NER (n=300 sentences from test split, span-F1 with type-match)
- **Design:** Examples paired across arms; arms run round-robin per example so tok/sec measurements aren't biased by daemon warmth or thermals
- **Sampling:** temperature=0.0, fixed seed (42), identical prompts across arms
- **Stats:** Wilson 95% CIs (MMLU) and bootstrap 95% CIs (NER) per arm; paired bootstrap diff CIs; McNemar (MMLU) and paired bootstrap (NER) p-values; Holm-Bonferroni adjustment per task; cluster-bootstrap on subjects for MMLU overall CI; per-subject MMLU breakdown
- **Cost:** tokens/sec and VRAM measured on DGX Spark per arm

## Power

At n=500 / arm on MMLU, detects ≥4pp accuracy differences at 80% power, α=0.05 (two-proportion z-test). Published quantization studies typically report 1–8pp differences between FP16 and Q4.
