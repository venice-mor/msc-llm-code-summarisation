"""
judge_evaluation.py
-------------------
LLM-as-a-Judge evaluation script using GPT-4o-mini.
Scores LLM-generated code summaries on four dimensions (1-5 Likert scale):
  - Correctness: factual accuracy vs the reference
  - Coverage: inclusion of essential aspects
  - Conciseness: clarity and brevity
  - Context: usefulness without reading the code

Usage:
    export OPENAI_API_KEY="your-key-here"
    python judge_evaluation.py

Input:  evaluation_input.csv  (columns: ID, Model, Reference, Summary)
Output: evaluation_results.csv (columns: ID, Model, Correctness, Coverage, Conciseness, Context)
"""

import os, json, time, pandas as pd
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ---- Choose provider ----
PROVIDER = "openai"
OPENAI_MODEL = "gpt-4o-mini"

# ---- Files ----
INPUT_CSV = "evaluation_input.csv"
OUTPUT_CSV = "evaluation_results.csv"
RESUME_CSV = "evaluation_progress_tmp.csv"

# ---- Rubric (consistent, short, unambiguous) ----
RUBRIC = """
You are an expert code-summary evaluator. Rate the GENERATED summary
against the REFERENCE summary on four 1-5 Likert scales.
Use integers only.

Definitions:
- Correctness: factual accuracy vs the reference (no hallucinations or contradictions).
- Coverage: inclusion of essential aspects (purpose, key behavior, inputs/outputs, errors).
- Conciseness: clarity and brevity (avoid fluff, redundancy, and raw links).
- Context: usefulness without needing to read the code (role/purpose is evident).

Guidelines:
- Prefer 3 when mixed (partly right, partly wrong / missing).
- Penalize explicit contradictions or speculation (Correctness<=2).
- Penalize missing core aspects (Coverage<=3).
- Penalize verbosity or meandering (Conciseness<=3).
- Penalize overly code-dependent phrasing (Context<=3).
Return ONLY a single CSV line:
Correctness,Coverage,Conciseness,Context
"""

# ---- Prompt template ----
def build_messages(reference: str, generated: str):
    sys = "You are precise, deterministic, and consistent. Answer with integers 1..5 only."
    user = f"""REFERENCE:
\"\"\"{reference.strip()[:4000]}\"\"\"

GENERATED:
\"\"\"{generated.strip()[:4000]}\"\"\"

{RUBRIC}
"""
    return sys, user


# ====== Providers ======
class ProviderError(Exception):
    pass

def _openai_client():
    from openai import OpenAI
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ProviderError("Missing OPENAI_API_KEY environment variable")
    return OpenAI(api_key=key)

@retry(reraise=True, stop=stop_after_attempt(5),
       wait=wait_exponential(multiplier=1, min=2, max=30),
       retry=retry_if_exception_type((ProviderError, Exception)))
def judge_once(reference: str, generated: str):
    client = _openai_client()
    sys, user = build_messages(reference, generated)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        max_tokens=50,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user}
        ],
    )
    text = resp.choices[0].message.content.strip()

    # Parse "c,cov,con,ctx"
    line = text.splitlines()[0]
    parts = [p.strip() for p in line.replace(" ", "").split(",")]
    if len(parts) != 4 or not all(p.isdigit() and 1 <= int(p) <= 5 for p in parts):
        import re
        nums = re.findall(r"\b[1-5]\b", text)
        if len(nums) >= 4:
            parts = nums[:4]
        else:
            raise ProviderError(f"Unparseable judge output: {text!r}")
    c, cov, conc, ctx = map(int, parts[:4])
    return c, cov, conc, ctx


# ====== Runner with resume ======
def load_input(path=INPUT_CSV):
    df = pd.read_csv(path)
    ren = {col: col.strip() for col in df.columns}
    df.rename(columns=ren, inplace=True)
    required = ["ID", "Model", "Reference", "Summary"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    return df

def maybe_resume(df: pd.DataFrame):
    if os.path.exists(RESUME_CSV):
        prev = pd.read_csv(RESUME_CSV)
        key_cols = ["ID", "Model"]
        df = df.merge(
            prev[key_cols + ["Correctness", "Coverage", "Conciseness", "Context"]],
            on=key_cols, how="left", suffixes=("", "_old")
        )
        return df
    else:
        for col in ["Correctness", "Coverage", "Conciseness", "Context"]:
            if col not in df.columns:
                df[col] = pd.NA
        return df

def save_progress(df):
    cols = ["ID", "Model", "Reference", "Summary",
            "Correctness", "Coverage", "Conciseness", "Context"]
    df[cols].to_csv(RESUME_CSV, index=False)

def main():
    df = load_input()
    df = maybe_resume(df)

    total = len(df)
    print(f"Rows: {total} | Provider: {PROVIDER} | Model: {OPENAI_MODEL}")

    for i, row in df.iterrows():
        if pd.notna(row.get("Correctness")):
            continue  # already done (resume)

        try:
            c, cov, conc, ctx = judge_once(str(row["Reference"]), str(row["Summary"]))
        except Exception as e:
            print(f"[{i}] ERROR: {e}")
            c, cov, conc, ctx = (pd.NA, pd.NA, pd.NA, pd.NA)

        df.at[i, "Correctness"] = c
        df.at[i, "Coverage"] = cov
        df.at[i, "Conciseness"] = conc
        df.at[i, "Context"] = ctx

        if (i + 1) % 10 == 0:
            save_progress(df)
            print(f"  ..saved progress at {i + 1}/{total}")

        time.sleep(0.2)  # polite pacing

    # Final save
    out_cols = ["ID", "Model", "Correctness", "Coverage", "Conciseness", "Context"]
    df[out_cols].to_csv(OUTPUT_CSV, index=False)
    df.to_csv(RESUME_CSV, index=False)
    print(f"Done. Wrote: {OUTPUT_CSV} (and full progress: {RESUME_CSV})")

if __name__ == "__main__":
    main()
