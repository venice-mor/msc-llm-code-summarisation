# Replication Package: Evaluating the Accuracy and Usefulness of LLMs for Code Summarisation

This repository contains the replication package for the MSc thesis:

**"Evaluating the Accuracy and Usefulness of LLMs for Code Summarisation in Open-Source Projects"**

*Venetia Moraki, MSc in Digital Transformation, Athens University of Economics and Business, 2025.*

*Supervisor: Dr. Maria Kechagia*

## Overview

This study evaluates eight Large Language Models (LLMs) on method-level code summarisation for 50 Python functions drawn from open-source repositories. The evaluation uses a multi-dimensional framework combining lexical metrics (BLEU, ROUGE, METEOR, Jaccard), semantic metrics (BERTScore, SentenceBERT), the code-aware SIDE metric, and an LLM-as-a-Judge qualitative assessment (GPT-4o-mini).

All models were evaluated under a unified zero-shot protocol — same functions, same prompt, same input format — so that observed differences reflect model capabilities rather than experimental variation.

## Models Evaluated

| Model | Category |
|-------|----------|
| OpenAI GPT-5 | Closed |
| Anthropic Claude Sonnet 4.5 | Closed |
| Google Gemini 2.5 Pro | Closed |
| Perplexity Sonar Large (Llama 3.1 70B) | Closed |
| DeepSeek-V3.2-Exp | Open |
| xAI Grok 4 | Closed |
| Mistral Codestral 25.08 | Code-Specialized |
| Meta AI Llama 4 | Open-Weight |

## Repository Structure

```
├── scripts/                          # Evaluation scripts
│   ├── compute_bleu_llms.py          # BLEU score computation
│   ├── compute_rouge_llms.py         # ROUGE-1/2/L computation
│   ├── compute_meteor_llms.py        # METEOR computation
│   ├── compute_jaccard_llms.py       # Jaccard similarity
│   ├── compute_bertscore_llms.py     # BERTScore (P/R/F1)
│   ├── compute_sentencebert_llms.py  # SentenceBERT (cosine sim + Euclidean)
│   ├── compute_side_all_llms.py      # SIDE metric (all models)
│   └── judge_evaluation.py           # LLM-as-a-Judge (GPT-4o-mini)
│
├── data/                             # Datasets and results
│   ├── results_fixed.xlsx            # Functions, LLM summaries, SIDE scores (per-model sheets)
│   ├── metrics_fixed.xlsx            # Consolidated metrics (400 rows × 17 columns)
│   └── LLM-as-a-Judge.xlsx          # Judge scores: Correctness, Coverage, Conciseness, Context
│
├── LICENSE
└── README.md
```

## Dataset

The dataset consists of 50 Python functions evaluated across 8 LLMs, producing 400 generated summaries.

**Sources:**
- [CodeXGLUE Code-to-Text](https://huggingface.co/datasets/google/code_x_glue_ct_code_to_text) — 38 functions
- [CodeSearchNet-Python](https://huggingface.co/datasets/Nan-Do/code-search-net-python) — 12 functions

The complete dataset with all summaries and scores is also available as a [Google Sheets file](https://docs.google.com/spreadsheets/d/1Q7aeSHiT7ZrAWuMziQ0ekx2fCNqKk789_YxuItgGyV4/edit?pli=1&gid=1403618722#gid=1403618722).

## Requirements

```
Python >= 3.13
pandas
nltk
bert-score
rouge-score
sentence-transformers
transformers
torch
openai (for Judge evaluation only)
```

Install dependencies:
```bash
pip install pandas nltk bert-score rouge-score sentence-transformers transformers torch openai
```

## Usage

### Metric Computation

Each metric script reads an Excel file with reference and generated summaries, computes the metric, and writes the results back:

```bash
python scripts/compute_bleu_llms.py
python scripts/compute_rouge_llms.py
python scripts/compute_meteor_llms.py
python scripts/compute_jaccard_llms.py
python scripts/compute_bertscore_llms.py
python scripts/compute_sentencebert_llms.py
python scripts/compute_side_all_llms.py
```

### SIDE Metric

The SIDE metric requires the pretrained checkpoint from [Mastropaolo et al. (ICSE 2024)](https://github.com/antonio-mastropaolo/code-summarization-metric). Download the checkpoint and place it in `SIDE/baseline/103080/` relative to the script.

### LLM-as-a-Judge

```bash
export OPENAI_API_KEY="your-key-here"
python scripts/judge_evaluation.py
```

Requires an OpenAI API key. Uses GPT-4o-mini to score summaries on Correctness, Coverage, Conciseness, and Context (1–5 Likert scale).

**Input:** `evaluation_input.csv` with columns: ID, Model, Reference, Summary

**Output:** `evaluation_results.csv` with columns: ID, Model, Correctness, Coverage, Conciseness, Context

## Citation

If you use this replication package, please cite:

```bibtex
@mastersthesis{moraki2025llm,
  author  = {Moraki, Venetia},
  title   = {Evaluating the Accuracy and Usefulness of LLMs
             for Code Summarisation in Open-Source Projects},
  school  = {Athens University of Economics and Business},
  year    = {2025},
  type    = {MSc Thesis}
}
```

## License

This project is licensed under the MIT License.
