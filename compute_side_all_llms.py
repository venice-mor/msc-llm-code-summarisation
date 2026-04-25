import re
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util

# -------------------------------------------------------
# ΡΥΘΜΙΣΕΙΣ
# -------------------------------------------------------
EXCEL_FILE = "Book1.xlsx"
SHEET_NAME = 0
START_EXCEL_ROW = 2
END_EXCEL_ROW = 401

# SIDE model path
CHECKPOINT_FOLDER = "SIDE/baseline/103080"  # Download checkpoint from Mastropaolo et al. replication package
DEVICE = "cpu"

# -------------------------------------------------------
# ΣΤΗΛΕΣ LLMs (index βάση του Book1.xlsx)
# -------------------------------------------------------
# Format:  (summary_column_index, output_column_index, output_column_name)

LLM_COLUMNS = [
    (1, 9,  "SIDE_ChatGPT5"),       # B → J
    (2, 10, "SIDE_Sonnet"),         # C → K
    (3, 11, "SIDE_Gemini"),         # D → L
    (4, 12, "SIDE_Perplexity"),     # E → M
    (5, 13, "SIDE_DeepSeek"),       # F → N
    (6, 14, "SIDE_Grok"),           # G → O
    (7, 15, "SIDE_Mistral"),        # H → P
    (8, 16, "SIDE_MetaLlama")       # I → Q
]

CODE_COL_INDEX = 0   # στήλη A → Functions

# -------------------------------------------------------
# CLEANING
# -------------------------------------------------------
def clean_summary(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

# -------------------------------------------------------
# LOAD SIDE MODEL
# -------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_FOLDER)
model = AutoModel.from_pretrained(CHECKPOINT_FOLDER).to(DEVICE)

# -------------------------------------------------------
# LOAD EXCEL
# -------------------------------------------------------
df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)

# -------------------------------------------------------
# ΒΕΒΑΙΩΣΗ ΟΤΙ ΥΠΑΡΧΟΥΝ ΟΙ ΣΤΗΛΕΣ ΕΞΟΔΟΥ
# -------------------------------------------------------
max_required_index = max([out_idx for _, out_idx, _ in LLM_COLUMNS])

while df.shape[1] <= max_required_index:
    df[f"col_{df.shape[1]}"] = None

cols = list(df.columns)
for _, out_idx, out_name in LLM_COLUMNS:
    cols[out_idx] = out_name
df.columns = cols

# -------------------------------------------------------
# MAIN LOOP ΓΙΑ ΟΛΑ ΤΑ LLMs
# -------------------------------------------------------
for excel_row in range(START_EXCEL_ROW, END_EXCEL_ROW + 1):
    df_index = excel_row - 2
    if df_index >= len(df):
        break

    code_text = df.iat[df_index, CODE_COL_INDEX]
    code_str = str(code_text) if isinstance(code_text, str) else ""

    if not code_str.strip():
        # αν δεν υπάρχει κώδικας, γεμίζει όλα τα SIDE της σειράς με 0
        for _, out_idx, _ in LLM_COLUMNS:
            df.iat[df_index, out_idx] = 0.0
        continue

    # υπολογίζουμε SIDE για κάθε LLM
    for summ_idx, out_idx, out_name in LLM_COLUMNS:

        summ_text = df.iat[df_index, summ_idx]
        summ_str = clean_summary(summ_text)

        if not summ_str.strip():
            sim = 0.0
        else:
            inputs = tokenizer([code_str, summ_str], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                output = model(**inputs)

            embeddings = mean_pooling(output, inputs["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)

            sim = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

        df.iat[df_index, out_idx] = sim

# -------------------------------------------------------
# SAVE
# -------------------------------------------------------
df.to_excel(EXCEL_FILE, index=False)
print("DONE! SIDE scores saved for ALL LLMs in Book1.xlsx")
