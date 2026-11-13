#!/usr/bin/env python3
"""
generate_fig3_loco.py
Regenerates Figure 3: PR-AUC and Recall across all 13 families under 5-seed strict LOCO.
"""
import os
import pandas as pd
import numpy as np

def main():
    csv_path = "strict_loco_summary.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    df_et = df[(df['Model'] == 'ExtraTrees') & (df['Family'] != 'OVERALL_MACRO')].copy()
    
    # Parse mean values
    df_et['PR_AUC_mean'] = df_et['PR_AUC'].apply(lambda x: float(x.split(' ± ')[0]))
    df_et['Recall_mean'] = df_et['Recall'].apply(lambda x: float(x.split(' ± ')[0]))
    
    df_sorted = df_et.sort_values(by='PR_AUC_mean', ascending=False)
    
    print("Family-level metrics for Figure 3:")
    for _, r in df_sorted.iterrows():
        print(f"  {r['Family']:16s} | PR-AUC: {r['PR_AUC_mean']:.4f} | Recall: {r['Recall_mean']:.4f}")

if __name__ == "__main__":
    main()

