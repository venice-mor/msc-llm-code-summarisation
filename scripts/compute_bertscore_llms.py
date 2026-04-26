import re
import pandas as pd
from bert_score import score as bert_score

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

# Ensure columns L (11), M (12), N (13) exist
while df.shape[1] <= 13:
    df[f"extra_col_{df.shape[1]}"] = None

cols = list(df.columns)
cols[11] = "BERT_P"
cols[12] = "BERT_R"
cols[13] = "BERT_F1"
df.columns = cols

# --- MAIN LOOP ---
for excel_row in range(START_EXCEL_ROW, END_EXCEL_ROW + 1):
    df_index = excel_row - 2
    if df_index >= len(df):
        break

    ref_text = clean_summary(df.iloc[df_index, 2])  # col C
    cand_text = clean_summary(df.iloc[df_index, 4]) # col E

    if not ref_text or not cand_text:
        P = R = F1 = 0.0
    else:
        P, R, F1 = bert_score([cand_text], [ref_text], lang="en", model_type="roberta-large")
        P = float(P[0])
        R = float(R[0])
        F1 = float(F1[0])

    df.iloc[df_index, 11] = P
    df.iloc[df_index, 12] = R
    df.iloc[df_index, 13] = F1

# --- SAVE OUTPUT ---
output_file = "LLMS_bertscore.xlsx"
df.to_excel(output_file, index=False)

print("BERTScore completed and saved to", output_file)
