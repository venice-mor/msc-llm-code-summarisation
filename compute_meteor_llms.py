import re
import pandas as pd
from nltk.translate.meteor_score import meteor_score

# --- CLEANING FUNCTION (ίδια με BLEU/ROUGE) ---
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

# --- ENSURE COLUMN J (index 9) EXISTS ---
while df.shape[1] <= 9:
    df[f"extra_col_{df.shape[1]}"] = None

cols = list(df.columns)
cols[9] = "METEOR"
df.columns = cols

# --- MAIN LOOP ---
for excel_row in range(START_EXCEL_ROW, END_EXCEL_ROW + 1):
    df_index = excel_row - 2
    if df_index >= len(df):
        break

    # column C = index 2, column E = index 4
    ref_text = clean_summary(df.iloc[df_index, 2])
    cand_text = clean_summary(df.iloc[df_index, 4])

    if not ref_text or not cand_text:
        score = 0.0
    else:
        ref_tokens = ref_text.split()
        cand_tokens = cand_text.split()
        score = meteor_score([ref_tokens], cand_tokens)

    df.iloc[df_index, 9] = score

# --- SAVE OUTPUT ---
output_file = "LLMS_meteor.xlsx"
df.to_excel(output_file, index=False)

print("METEOR completed and saved to", output_file)
