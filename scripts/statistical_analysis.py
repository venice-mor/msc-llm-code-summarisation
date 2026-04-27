"""
statistical_analysis.py
-----------------------
Reproduces all statistical tests reported in the thesis:
  1. Kruskal-Wallis H-test (Table 3) — overall model differences per metric
  2. Wilcoxon signed-rank tests with Bonferroni correction (Table 4) — pairwise SIDE comparisons
  3. Spearman rank correlations (Table 5) — agreement between metric families

Input:  metrics_fixed.xlsx (400 rows: 50 functions × 8 models, with all metric scores)
        LLM-as-a-Judge.xlsx (400 rows: Judge scores per summary)
Output: Prints all tables to console; saves results to statistical_results.xlsx

Usage:
    python statistical_analysis.py

Requirements:
    pip install pandas scipy openpyxl
"""

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_excel("data/metrics_fixed.xlsx")
judge = pd.read_excel("data/LLM-as-a-Judge.xlsx")

# Normalize model names (fix double spaces, trailing whitespace)
df['Model'] = df['Model'].str.replace(r'\s+', ' ', regex=True).str.strip()
judge['Model'] = judge['Model'].str.replace(r'\s+', ' ', regex=True).str.strip()

# Merge Judge scores into metrics
# First compute Judge average (only Valid rows)
judge_valid = judge[judge['Valid'] == 'Valid'].copy()
judge_valid['Judge_Avg'] = judge_valid[['Correctness', 'Coverage', 'Conciseness', 'Context']].mean(axis=1)
judge_avg = judge_valid.groupby(['ID', 'Model'])['Judge_Avg'].mean().reset_index()

df = df.merge(judge_avg, on=['ID', 'Model'], how='left')

models = df['Model'].unique().tolist()
print(f"Models: {len(models)}")
print(f"Total rows: {len(df)}")
print(f"Rows with Judge scores: {df['Judge_Avg'].notna().sum()}")

# ============================================================
# 2. TABLE 2: Mean ± SD per model per metric
# ============================================================
print("\n" + "=" * 70)
print("TABLE 2: Mean evaluation scores (± SD) per model")
print("=" * 70)

metrics_cols = ['BLEU', 'ROUGE1', 'METEOR', 'BERT_F1', 'SBERT_CS', 'SIDE']

for model in models:
    subset = df[df['Model'] == model]
    vals = []
    for col in metrics_cols:
        mean = subset[col].mean()
        std = subset[col].std()
        vals.append(f"{mean:.3f}±{std:.3f}")
    # Judge
    judge_subset = subset['Judge_Avg'].dropna()
    if len(judge_subset) > 0:
        vals.append(f"{judge_subset.mean():.2f}")
    else:
        vals.append("N/A")
    print(f"  {model:45s} {' | '.join(vals)}")

# ============================================================
# 3. TABLE 3: Kruskal-Wallis H-test
# ============================================================
print("\n" + "=" * 70)
print("TABLE 3: Kruskal-Wallis H-test (df=7, N=50 per group)")
print("=" * 70)
print(f"  {'Metric':<15s} {'H':>10s} {'p-value':>12s} {'Sig.':>6s}")
print("  " + "-" * 45)

test_metrics = ['SIDE', 'BERT_F1', 'BLEU', 'METEOR', 'SBERT_CS']

for metric in test_metrics:
    groups = [df[df['Model'] == m][metric].values for m in models]
    h_stat, p_val = stats.kruskal(*groups)
    
    if p_val < 0.001:
        sig = "***"
    elif p_val < 0.01:
        sig = "**"
    elif p_val < 0.05:
        sig = "*"
    else:
        sig = "n.s."
    
    print(f"  {metric:<15s} {h_stat:>10.2f} {p_val:>12.4f} {sig:>6s}")

# ============================================================
# 4. TABLE 4: Pairwise Wilcoxon signed-rank tests on SIDE
# ============================================================
print("\n" + "=" * 70)
print("TABLE 4: Pairwise Wilcoxon signed-rank tests on SIDE scores")
print("         Bonferroni-corrected α = 0.05/28 = 0.0018")
print("=" * 70)

n_comparisons = len(list(combinations(models, 2)))
alpha_corrected = 0.05 / n_comparisons

print(f"  Number of pairwise comparisons: {n_comparisons}")
print(f"  Corrected alpha: {alpha_corrected:.4f}")
print()
print(f"  {'Comparison':<55s} {'ΔMean':>7s} {'W':>8s} {'p-value':>10s} {'r':>6s} {'Sig.':>5s}")
print("  " + "-" * 93)

# Prepare paired data (each model has scores for the same 50 functions)
model_side = {}
for model in models:
    subset = df[df['Model'] == model].sort_values('ID')
    model_side[model] = subset['SIDE'].values

results_wilcoxon = []
for m1, m2 in combinations(models, 2):
    scores1 = model_side[m1]
    scores2 = model_side[m2]
    diff_mean = scores1.mean() - scores2.mean()
    
    try:
        w_stat, p_val = stats.wilcoxon(scores1, scores2)
        n = len(scores1)
        # Effect size r = |Z| / sqrt(N)
        # Approximate Z from W
        z_val = stats.norm.ppf(p_val / 2)
        r_effect = abs(z_val) / np.sqrt(n)
    except Exception:
        w_stat, p_val, r_effect = np.nan, np.nan, np.nan
    
    if p_val < 0.001:
        sig = "***"
    elif p_val < alpha_corrected:
        sig = "**"
    elif p_val < 0.05:
        sig = "*"
    else:
        sig = "n.s."
    
    # Shorten model names for display
    short1 = m1[:20]
    short2 = m2[:20]
    comparison = f"{short1} vs {short2}"
    
    results_wilcoxon.append({
        'comparison': comparison,
        'delta': diff_mean,
        'W': w_stat,
        'p': p_val,
        'r': r_effect,
        'sig': sig
    })
    
    print(f"  {comparison:<55s} {diff_mean:>+.3f} {w_stat:>8.0f} {p_val:>10.4f} {r_effect:>6.3f} {sig:>5s}")

# ============================================================
# 5. TABLE 5: Spearman rank correlation matrix
# ============================================================
print("\n" + "=" * 70)
print("TABLE 5: Spearman rank correlation matrix (N=400)")
print("=" * 70)

corr_cols = ['BLEU', 'ROUGE1', 'METEOR', 'BERT_F1', 'SBERT_CS', 'SIDE', 'Judge_Avg']

# Only use rows that have Judge scores for the full correlation
df_corr = df.dropna(subset=['Judge_Avg'])
print(f"  Using {len(df_corr)} rows with complete Judge scores")
print()

# Header
header = f"  {'':>12s}"
for col in corr_cols:
    short = col.replace('_Avg', '').replace('_', ' ')[:8]
    header += f" {short:>10s}"
print(header)
print("  " + "-" * (12 + 11 * len(corr_cols)))

for col1 in corr_cols:
    row_str = f"  {col1:>12s}"
    for col2 in corr_cols:
        if col1 == col2:
            row_str += f" {'—':>10s}"
        else:
            rho, p_val = stats.spearmanr(df_corr[col1], df_corr[col2])
            if p_val < 0.001:
                sig = "***"
            elif p_val < 0.01:
                sig = "**"
            elif p_val < 0.05:
                sig = "*"
            else:
                sig = ""
            row_str += f" {rho:>+.3f}{sig:<3s}"
    print(row_str)

# ============================================================
# 6. TABLE 6: Model rankings per dimension
# ============================================================
print("\n" + "=" * 70)
print("TABLE 6: Model rankings per evaluation dimension (1=best)")
print("=" * 70)

rank_metrics = {
    'BLEU': True,      # higher is better
    'BERT_F1': True,   # higher is better
    'SIDE': True,      # higher is better
    'Judge_Avg': True   # higher is better
}

means = df.groupby('Model').agg({
    'BLEU': 'mean',
    'BERT_F1': 'mean', 
    'SIDE': 'mean',
    'Judge_Avg': 'mean'
}).reset_index()

print(f"  {'Model':<45s} {'BLEU':>5s} {'BERT':>5s} {'SIDE':>5s} {'Judge':>5s} {'Gap':>5s}")
print("  " + "-" * 70)

for _, row in means.iterrows():
    bleu_rank = means['BLEU'].rank(ascending=False)[means['Model'] == row['Model']].values[0]
    bert_rank = means['BERT_F1'].rank(ascending=False)[means['Model'] == row['Model']].values[0]
    side_rank = means['SIDE'].rank(ascending=False)[means['Model'] == row['Model']].values[0]
    judge_rank = means['Judge_Avg'].rank(ascending=False)[means['Model'] == row['Model']].values[0]
    
    ranks = [bleu_rank, bert_rank, side_rank, judge_rank]
    gap = int(max(ranks) - min(ranks))
    
    print(f"  {row['Model']:<45s} {int(bleu_rank):>5d} {int(bert_rank):>5d} {int(side_rank):>5d} {int(judge_rank):>5d} {gap:>5d}")

# ============================================================
# 7. SAVE RESULTS
# ============================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

with pd.ExcelWriter("statistical_results.xlsx") as writer:
    # Table 2
    summary = df.groupby('Model')[metrics_cols + ['Judge_Avg']].agg(['mean', 'std'])
    summary.to_excel(writer, sheet_name='Table2_MeanSD')
    
    # Table 4
    wilcoxon_df = pd.DataFrame(results_wilcoxon)
    wilcoxon_df.to_excel(writer, sheet_name='Table4_Wilcoxon', index=False)
    
    # Table 5
    corr_matrix = df_corr[corr_cols].corr(method='spearman')
    corr_matrix.to_excel(writer, sheet_name='Table5_Spearman')
    
    # Table 6
    means.to_excel(writer, sheet_name='Table6_Rankings', index=False)

print("  Saved: statistical_results.xlsx")
print("\nDONE!")
