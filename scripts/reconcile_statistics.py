"""Phase 1: Reconcile statistical significance across strict and mapped category accuracy."""

import json
from pathlib import Path
import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent

test_file = REPO_ROOT / "ml/data/splits_final/test.jsonl"
baseline_file = REPO_ROOT / "experiments/final_run/baseline_predictions.jsonl"
finetuned_file = REPO_ROOT / "experiments/final_run/finetuned_predictions.jsonl"

test_data = [json.loads(line) for line in open(test_file, "r", encoding="utf-8")]
baseline_preds = [json.loads(line) for line in open(baseline_file, "r", encoding="utf-8")]
finetuned_preds = [json.loads(line) for line in open(finetuned_file, "r", encoding="utf-8")]

assert len(baseline_preds) == len(finetuned_preds) == 162

CANONICAL_CATEGORIES = [
    "Application",
    "Database",
    "Infrastructure",
    "Kubernetes / Container Orchestration",
    "Network",
]

def normalize_text(text: str) -> str:
    import re
    text = text.lower()
    text = re.sub(r"[_\-/]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def map_category_string(cat_str: str) -> str:
    norm = normalize_text(cat_str)
    for c in CANONICAL_CATEGORIES:
        c_norm = normalize_text(c)
        if norm == c_norm or norm in c_norm or c_norm in norm:
            return c
    return "Application"

# 1. Strict direct matching
strict_base = []
strict_ft = []

for b, f in zip(baseline_preds, finetuned_preds):
    gt = b["ground_truth"]["category"]
    pred_b = b["prediction_report"].get("incident_category", "")
    pred_f = f["prediction_report"].get("incident_category", "")
    
    norm_gt = normalize_text(str(gt))
    norm_b = normalize_text(str(pred_b))
    norm_f = normalize_text(str(pred_f))
    
    match_b = bool(norm_b and norm_gt and (norm_b == norm_gt or norm_b in norm_gt or norm_gt in norm_b))
    match_f = bool(norm_f and norm_gt and (norm_f == norm_gt or norm_f in norm_gt or norm_gt in norm_f))
    
    strict_base.append(1 if match_b else 0)
    strict_ft.append(1 if match_f else 0)

# 2. Mapped fallback matching
mapped_base = []
mapped_ft = []

for b, f in zip(baseline_preds, finetuned_preds):
    gt = map_category_string(b["ground_truth"]["category"])
    pred_b = map_category_string(b["prediction_report"].get("incident_category", ""))
    pred_f = map_category_string(f["prediction_report"].get("incident_category", ""))
    
    mapped_base.append(1 if pred_b == gt else 0)
    mapped_ft.append(1 if pred_f == gt else 0)

def analyze_contingency(name, base_arr, ft_arr):
    print(f"\n{'='*70}\nCONTINGENCY & MCNEMAR ANALYSIS: {name.upper()}\n{'='*70}")
    n = len(base_arr)
    # Contingency table:
    # Both correct (1, 1): n11
    # Baseline correct, FT incorrect (1, 0): n10 (worsened)
    # Baseline incorrect, FT correct (0, 1): n01 (improved)
    # Both incorrect (0, 0): n00
    n11 = sum(1 for b, f in zip(base_arr, ft_arr) if b == 1 and f == 1)
    n10 = sum(1 for b, f in zip(base_arr, ft_arr) if b == 1 and f == 0) # worse
    n01 = sum(1 for b, f in zip(base_arr, ft_arr) if b == 0 and f == 1) # better
    n00 = sum(1 for b, f in zip(base_arr, ft_arr) if b == 0 and f == 0)
    
    print(f"Total samples: {n}")
    print(f"Baseline accuracy: {sum(base_arr)}/{n} = {sum(base_arr)/n:.4f} ({sum(base_arr)/n*100:.2f}%)")
    print(f"Fine-Tuned accuracy: {sum(ft_arr)}/{n} = {sum(ft_arr)/n:.4f} ({sum(ft_arr)/n*100:.2f}%)")
    delta_pp = (sum(ft_arr) - sum(base_arr)) / n * 100
    rel_pct = (sum(ft_arr) - sum(base_arr)) / sum(base_arr) * 100
    print(f"Absolute delta: {delta_pp:+.2f} pp, Relative delta: {rel_pct:+.2f}%")
    
    print("\n2x2 Contingency Table (Rows: Baseline, Columns: Fine-Tuned):")
    print(f"                   FT Correct (1)    FT Incorrect (0)    Total Baseline")
    print(f"Base Correct (1)      n11 = {n11:3d}           n10 = {n10:3d} (worse)      {n11+n10:3d}")
    print(f"Base Incorrect (0)    n01 = {n01:3d} (better)  n00 = {n00:3d}              {n01+n00:3d}")
    print(f"Total Fine-Tuned      {n11+n01:3d}              {n10+n00:3d}              {n:3d}")
    
    b_disc = n01 # cases improved
    c_disc = n10 # cases worsened
    print(f"\nDiscordant pairs: b (improved) = {b_disc}, c (worsened) = {c_disc}, total discordant = {b_disc + c_disc}")
    
    # 1. Exact two-sided McNemar (binomial test)
    res_exact = stats.binomtest(k=b_disc, n=b_disc + c_disc, p=0.5, alternative="two-sided")
    p_exact = res_exact.pvalue
    print(f"1. Exact Two-Sided McNemar (scipy.stats.binomtest): p = {p_exact:.6f} (or {p_exact:.4f})")
    
    # 2. Asymptotic McNemar without continuity correction
    chi2_raw = ((b_disc - c_disc) ** 2) / (b_disc + c_disc)
    p_asymp_raw = 1.0 - stats.chi2.cdf(chi2_raw, df=1)
    print(f"2. Asymptotic McNemar without continuity correction: chi2 = {chi2_raw:.4f}, p = {p_asymp_raw:.6f} (or {p_asymp_raw:.4f})")
    
    # 3. Asymptotic McNemar with Edwards continuity correction
    chi2_corr = ((abs(b_disc - c_disc) - 1.0) ** 2) / (b_disc + c_disc)
    p_asymp_corr = 1.0 - stats.chi2.cdf(chi2_corr, df=1)
    print(f"3. Asymptotic McNemar WITH continuity correction: chi2 = {chi2_corr:.4f}, p = {p_asymp_corr:.6f} (or {p_asymp_corr:.4f})")
    
    # 4. Bootstrap Confidence Intervals
    for seed in [42, 2026]:
        for n_iter in [1000, 10000]:
            rng = np.random.RandomState(seed)
            diffs = []
            for _ in range(n_iter):
                idx = rng.randint(0, n, size=n)
                b_m = np.mean([base_arr[i] for i in idx])
                f_m = np.mean([ft_arr[i] for i in idx])
                diffs.append(f_m - b_m)
            diffs = np.sort(diffs)
            ci_low = float(np.percentile(diffs, 2.5))
            ci_high = float(np.percentile(diffs, 97.5))
            print(f"Bootstrap CI (seed={seed}, B={n_iter:5d}): [{ci_low*100:+.2f} pp, {ci_high*100:+.2f} pp] (raw: [{ci_low:.4f}, {ci_high:.4f}])")

analyze_contingency("Strict Category Matching", strict_base, strict_ft)
analyze_contingency("Mapped Fallback Matching", mapped_base, mapped_ft)
