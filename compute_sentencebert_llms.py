import re
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# --- CLEANING FUNCTION ---
def clean_summary(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- CONFIG ---
START_EXCEL_ROW = 3
END_EXCEL_ROW = 401
xlsx_file = "LLMS.xlsx"

# Load FIRST sheet
df = pd.read_excel(xlsx_file, sheet_name=0)

# Ensure columns O, P exist (index 14, 15)
while df.shape[1] <= 15:
    df[f"extra_col_{df.shape[1]}"] = None

cols = list(df.columns)
cols[14] = "SBERT_CS"   # cosine similarity
cols[15] = "SBERT_ED"   # Euclidean distance
df.columns = cols

# Load lightweight but accurate SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- MAIN LOOP ---
for excel_row in range(START_EXCEL_ROW, END_EXCEL_ROW + 1):
    df_index = excel_row - 2
    if df_index >= len(df):
        break

    ref_text = clean_summary(df.iloc[df_index, 2])
    cand_text = clean_summary(df.iloc[df_index, 4])

    if not ref_text or not cand_text:
        cs = 0.0
        ed = 999.0
    else:
        emb_ref = model.encode(ref_text, convert_to_tensor=True)
        emb_cand = model.encode(cand_text, convert_to_tensor=True)

        cs = util.cos_sim(emb_ref, emb_cand).item()
        ed = util.pytorch_cos_sim(emb_ref, emb_cand)  # This is cosine; for EUCLIDEAN, compute manually
        ed = (emb_ref - emb_cand).norm().item()       # Euclidean distance

    df.iloc[df_index, 14] = cs
    df.iloc[df_index, 15] = ed

# --- SAVE OUTPUT ---
output_file = "LLMS_sentencebert.xlsx"
df.to_excel(output_file, index=False)

print("SentenceBERT CS/ED completed and saved to", output_file)
