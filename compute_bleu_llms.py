import re
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

# --- CLEANING FUNCTION ---
def clean_summary(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove citation markers [1], [2], [10]
    text = re.sub(r'\[\d+\]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- CONFIG ---
START_EXCEL_ROW = 3
END_EXCEL_ROW = 401
xlsx_file = "LLMS.xlsx"

# Load FIRST sheet
df = pd.read_excel(xlsx_file, sheet_name=0)

# --- MAKE SURE COLUMN F EXISTS ---
while df.shape[1] <= 5:
    df[f"extra_col_{df.shape[1]}"] = None

df.columns = list(df.columns[:5]) + ["BLEU_C_E"] + list(df.columns[6:])

# --- BLEU FUNCTION ---
def compute_sentence_bleu(reference_text, candidate_text):
    if not isinstance(reference_text, str) or not isinstance(candidate_text, str):
        return 0.0
    ref_tokens = word_tokenize(reference_text)
    cand_tokens = word_tokenize(candidate_text)
    if len(ref_tokens) == 0 or len(cand_tokens) == 0:
        return 0.0
    smoothing = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], cand_tokens,
                         weights=(0.25, 0.25, 0.25, 0.25),
                         smoothing_function=smoothing)

# --- MAIN LOOP ---
for excel_row in range(START_EXCEL_ROW, END_EXCEL_ROW + 1):
    df_index = excel_row - 2
    if df_index >= len(df):
        break

    ref_text = clean_summary(df.iloc[df_index, 2])   # column C
    cand_text = clean_summary(df.iloc[df_index, 4])  # column E

    bleu = compute_sentence_bleu(ref_text, cand_text)
    df.iloc[df_index, 5] = bleu

# --- SAVE OUTPUT ---
output_file = "LLMS_bleu.xlsx"
df.to_excel(output_file, index=False)

print("BLEU completed and saved to", output_file)
