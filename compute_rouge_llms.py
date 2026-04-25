import re
import pandas as pd
from rouge_score import rouge_scorer

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

# --- ENSURE COLUMNS G, H, I EXIST ---
# G -> index 6, H -> index 7, I -> index 8
while df.shape[1] <= 8:
    df[f"extra_col_{df.shape[1]}"] = None

rouge1_col_name = "ROUGE_1_F1"
rouge2_col_name = "ROUGE_2_F1"
rougeL_col_name = "ROUGE_L_F1"

cols = list(df.columns)
cols[6] = rouge1_col_name
cols[7] = rouge2_col_name
cols[8] = rougeL_col_name
df.columns = cols

# --- ROUGE SCORER ---
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# --- MAIN LOOP ---
for excel_row in range(START_EXCEL_ROW, END_EXCEL_ROW + 1):
    df_index = excel_row - 2
    if df_index >= len(df):
        break

    # column C = index 2, column E = index 4
    ref_text = clean_summary(df.iloc[df_index, 2])
    cand_text = clean_summary(df.iloc[df_index, 4])

    if not ref_text or not cand_text:
        r1_f = r2_f = rL_f = 0.0
    else:
        scores = scorer.score(ref_text, cand_text)
        r1_f = scores['rouge1'].fmeasure
        r2_f = scores['rouge2'].fmeasure
        rL_f = scores['rougeL'].fmeasure

    df.iloc[df_index, 6] = r1_f
    df.iloc[df_index, 7] = r2_f
    df.iloc[df_index, 8] = rL_f

# --- SAVE OUTPUT ---
output_file = "LLMS_rouge.xlsx"
df.to_excel(output_file, index=False)

print("ROUGE completed and saved to", output_file)
