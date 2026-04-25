import re
import pandas as pd

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

# Ensure column K (index 10) exists
while df.shape[1] <= 10:
    df[f"extra_col_{df.shape[1]}"] = None

cols = list(df.columns)
cols[10] = "JACCARD"
df.columns = cols

# --- JACCARD FUNCTION ---
def jaccard_similarity(set1, set2):
    if len(set1) == 0 and len(set2) == 0:
        return 0.0
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if len(union) == 0:
        return 0.0
    return len(intersection) / len(union)

# --- MAIN LOOP ---
for excel_row in range(START_EXCEL_ROW, END_EXCEL_ROW + 1):
    df_index = excel_row - 2
    if df_index >= len(df):
        break

    ref_text = clean_summary(df.iloc[df_index, 2])
    cand_text = clean_summary(df.iloc[df_index, 4])

    ref_set = set(ref_text.lower().split())
    cand_set = set(cand_text.lower().split())

    score = jaccard_similarity(ref_set, cand_set)

    df.iloc[df_index, 10] = score

# --- SAVE OUTPUT ---
output_file = "LLMS_jaccard.xlsx"
df.to_excel(output_file, index=False)

print("Jaccard completed and saved to", output_file)
