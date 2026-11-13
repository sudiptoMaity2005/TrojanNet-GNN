#!/usr/bin/env python3
"""
generate_confusion_matrix.py
Generates the publication-grade Confusion Matrix for the proposed Hybrid GNN + Extra Trees model
across all 108 benchmark circuits and 5 seeds (1,489,029 gates per seed, total 7,445,145 node evaluations).
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def main():
    raw_csv = "results/strict_loco/strict_loco_all_6classifiers_raw.csv"
    if not os.path.exists(raw_csv):
        print(f"File not found: {raw_csv}")
        return

    with open(raw_csv, "r") as f:
        rows = list(csv.DictReader(f))

    et_rows = [r for r in rows if r["Model"] == "ExtraTrees"]

    by_seed = defaultdict(lambda: {"TP": 0, "FP": 0, "TN": 0, "FN": 0})
    for r in et_rows:
        s = r["Seed"]
        by_seed[s]["TP"] += int(r["TP"])
        by_seed[s]["FP"] += int(r["FP"])
        by_seed[s]["TN"] += int(r["TN"])
        by_seed[s]["FN"] += int(r["FN"])

    # Average per seed
    avg_tp = np.mean([d["TP"] for d in by_seed.values()])
    avg_fp = np.mean([d["FP"] for d in by_seed.values()])
    avg_tn = np.mean([d["TN"] for d in by_seed.values()])
    avg_fn = np.mean([d["FN"] for d in by_seed.values()])

    std_tp = np.std([d["TP"] for d in by_seed.values()])
    std_fp = np.std([d["FP"] for d in by_seed.values()])
    std_tn = np.std([d["TN"] for d in by_seed.values()])
    std_fn = np.std([d["FN"] for d in by_seed.values()])

    total_gates = avg_tp + avg_fp + avg_tn + avg_fn
    total_trojans = avg_tp + avg_fn
    total_benign = avg_tn + avg_fp

    # Rates
    tpr = (avg_tp / total_trojans) * 100 # Recall
    fnr = (avg_fn / total_trojans) * 100 # Miss rate
    tnr = (avg_tn / total_benign) * 100  # Specificity
    fpr = (avg_fp / total_benign) * 100  # False alarm rate

    print("=== PROPOSED MODEL (HYBRID GNN + EXTRA TREES) CONFUSION MATRIX ===")
    print(f"Total Nodes per Run: {int(total_gates):,}")
    print(f"True Negatives  (TN): {avg_tn:12.1f} ± {std_tn:8.1f} ({tnr:6.2f}%)")
    print(f"False Positives (FP): {avg_fp:12.1f} ± {std_fp:8.1f} ({fpr:6.2f}%)")
    print(f"False Negatives (FN): {avg_fn:12.1f} ± {std_fn:8.1f} ({fnr:6.2f}%)")
    print(f"True Positives  (TP): {avg_tp:12.1f} ± {std_tp:8.1f} ({tpr:6.2f}%)")

    # Save CSV
    out_csv = "results/confusion_matrix_proposed_model.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Count_Mean", "Count_Std", "Rate_Pct"])
        writer.writerow(["True Negatives (TN)", f"{avg_tn:.1f}", f"{std_tn:.1f}", f"{tnr:.2f}%"])
        writer.writerow(["False Positives (FP)", f"{avg_fp:.1f}", f"{std_fp:.1f}", f"{fpr:.2f}%"])
        writer.writerow(["False Negatives (FN)", f"{avg_fn:.1f}", f"{std_fn:.1f}", f"{fnr:.2f}%"])
        writer.writerow(["True Positives (TP)", f"{avg_tp:.1f}", f"{std_tp:.1f}", f"{tpr:.2f}%"])
        writer.writerow(["Total Benign", f"{total_benign:.1f}", "-", "100.0%"])
        writer.writerow(["Total Trojans", f"{total_trojans:.1f}", "-", "100.0%"])
        writer.writerow(["Total Gates", f"{total_gates:.1f}", "-", "100.0%"])
    print(f"Saved: {out_csv}")

    # Plot Confusion Matrix Heatmap
    plt.figure(figsize=(7, 6), dpi=300)
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    cm_counts = np.array([[avg_tn, avg_fp], [avg_fn, avg_tp]])
    cm_rates = np.array([[tnr, fpr], [fnr, tpr]])

    annot = np.empty((2, 2), dtype=object)
    annot[0, 0] = f"True Negative\n{int(round(avg_tn)):,}\n({tnr:.2f}%)"
    annot[0, 1] = f"False Positive\n{int(round(avg_fp)):,}\n({fpr:.2f}%)"
    annot[1, 0] = f"False Negative\n{int(round(avg_fn)):,}\n({fnr:.2f}%)"
    annot[1, 1] = f"True Positive\n{int(round(avg_tp)):,}\n({tpr:.2f}%)"

    # Log-normalized color mapping due to 0.1% extreme class imbalance
    log_cm = np.log10(cm_counts + 1)
    ax = sns.heatmap(log_cm, annot=annot, fmt="", cmap="Blues", cbar=False,
                     xticklabels=["Predicted Benign (0)", "Predicted Trojan (1)"],
                     yticklabels=["Actual Benign (0)", "Actual Trojan (1)"],
                     linewidths=1.5, linecolor="black", annot_kws={"size": 12, "weight": "bold"})

    plt.title("Confusion Matrix: Proposed Hybrid GNN + Extra Trees\n(Strict Cross-Family LOCO, Reference-Free $f_2 \\equiv 0.0$, 5 Seeds Avg.)",
              fontsize=12, pad=14, weight="bold")
    plt.xlabel("Predicted Class", fontsize=11, labelpad=8, weight="bold")
    plt.ylabel("Ground Truth Class", fontsize=11, labelpad=8, weight="bold")
    plt.tight_layout()

    out_pdf = "results/confusion_matrix_proposed_model.pdf"
    out_png = "results/confusion_matrix_proposed_model.png"
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.savefig(out_png, format="png", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")

if __name__ == "__main__":
    main()

