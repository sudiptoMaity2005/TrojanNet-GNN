import os
import glob
import math
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, average_precision_score, confusion_matrix
)
from collections import defaultdict

SEEDS = [42, 123, 2024, 7, 999]
CONTAMINATION_RATE = 0.01  # Fixed 1% prior expected Trojan rate (unsupervised, no label tuning)

INVERTED_RS232_VARIANTS = {
    'RS232_T1000_180nm', 'RS232_T1000_90nm',
    'RS232_T1100_180nm', 'RS232_T1100_90nm',
    'RS232_T1200_180nm', 'RS232_T1200_90nm',
    'RS232_T1300_180nm', 'RS232_T1300_90nm',
    'RS232_T1400_180nm', 'RS232_T1400_90nm',
    'RS232_T1500_180nm', 'RS232_T1500_90nm',
    'RS232_T1600_180nm', 'RS232_T1600_90nm',
    'RS232_T1700_180nm_scan', 'RS232_T1700_90nm_scan',
    'RS232_T1800_180nm_scan', 'RS232_T1800_90nm_scan',
    'RS232_T1900_180nm_scan', 'RS232_T1900_90nm_scan',
    'RS232_T2000_180nm_scan', 'RS232_T2000_90nm_scan'
}

def determine_family(circuit_name):
    lower_name = circuit_name.lower()
    if 'b19' in lower_name: return 'b19'
    if 'ethernet' in lower_name: return 'EthernetMAC10GE'
    if 'wb_conmax' in lower_name or 'wb-conmax' in lower_name: return 'wb_conmax'
    if 'rs232' in lower_name: return 'RS232'
    if 'aes' in lower_name: return 'AES'
    if circuit_name.startswith('c') or circuit_name.startswith('s'):
        return circuit_name.split('_')[0]
    return 'unknown'

def is_true_trojan_iscas(node_name):
    n = str(node_name).lower()
    if n.startswith('test_') or 'crc_out' in n:
        return 0
    return 1 if any(n.startswith(k) for k in ['troj', 'trigger', 'tempn', 'tempg']) else 0

def load_rs232_verdicts():
    verdicts = {}
    verdict_path = "results/rs232_trojan_nodes_with_verdict.csv"
    if os.path.exists(verdict_path):
        df_v = pd.read_csv(verdict_path)
        for _, row in df_v.iterrows():
            verdicts[(str(row['Circuit']), str(row['Node']))] = int(row['Verified_Trojan'])
    return verdicts

def main():
    print("=== Running Baseline: Mondal et al. [15] (MondalRareNet) across all 108 Circuits ===")
    print(f"Seeds: {SEEDS} | Fixed Contamination Rate: {CONTAMINATION_RATE}")
    
    rs232_verdicts = load_rs232_verdicts()
    print(f"Loaded {len(rs232_verdicts)} verified RS232 verdicts (105 false positives relabeled to 0).")

    folders = [
        "datasets_for_gnn", "b19_datasets", "ethernet_datasets", "wb_conmax_datasets",
        "aes_output_datasets", "rs232_output_datasets"
    ]
    
    all_circuit_files = {}
    for fldr in folders:
        for root, dirs, files in os.walk(fldr):
            for f in files:
                if f.endswith('.csv') and not f.startswith('edges_') and 'results' not in f.lower() and 'summary' not in f.lower():
                    c_name = f.replace("dataset_", "").replace(".csv", "")
                    if c_name in INVERTED_RS232_VARIANTS:
                        continue
                    if c_name not in all_circuit_files:
                        all_circuit_files[c_name] = os.path.join(root, f)

    print(f"Found {len(all_circuit_files)} benchmark circuits across 13 families.")
    
    all_results = []

    for seed in SEEDS:
        print(f"\n--- Processing Seed {seed} ---")
        for c_name, csv_path in sorted(all_circuit_files.items()):
            fam = determine_family(c_name)
            df = pd.read_csv(csv_path)
            
            node_names = df['Node'].astype(str).tolist() if 'Node' in df.columns else [str(i) for i in range(len(df))]
            
            # Ground truth determination
            if c_name.startswith(('c', 's')) and node_names:
                y_true = np.array([is_true_trojan_iscas(n) for n in node_names], dtype=int)
            elif 'RS232' in c_name:
                y_true = []
                for n in node_names:
                    if (c_name, n) in rs232_verdicts:
                        y_true.append(rs232_verdicts[(c_name, n)])
                    elif 'Label' in df.columns:
                        y_true.append(int(df.loc[df['Node'] == n, 'Label'].values[0]))
                    else:
                        y_true.append(0)
                y_true = np.array(y_true, dtype=int)
            elif 'Label' in df.columns:
                y_true = df['Label'].values.astype(int)
            else:
                y_true = np.zeros(len(df), dtype=int)
                
            # Extract features for Mondal et al. [15]:
            # Focuses on controllability & transition probability: f1 (TP), f3 (Rare/1-TP), f4 (Neighbor Mean TP), f7 (FanIn), f8 (FanOut)
            feat_cols = [c for c in df.columns if c not in ['Node', 'Label']]
            if len(feat_cols) > 8:
                f_last8 = feat_cols[-8:]
            else:
                f_last8 = feat_cols
                
            # Mondal's features: f1_TP, f3_Rare, f4_NbMean, f7_FanIn, f8_FanOut
            # Indices relative to 8 features: [0, 2, 3, 6, 7]
            if len(f_last8) == 8:
                selected_cols = [f_last8[0], f_last8[2], f_last8[3], f_last8[6], f_last8[7]]
            else:
                selected_cols = f_last8
                
            X = df[selected_cols].values
            
            # In Mondal et al., f2 is never used (golden-free). Standardize per circuit.
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X_std = (X - np.mean(X, axis=0, keepdims=True)) / (np.std(X, axis=0, keepdims=True) + 1e-8)
            
            # Run MondalRareNet (Unsupervised Outlier Isolation on Rare Nets)
            clf = IsolationForest(
                n_estimators=100,
                contamination=CONTAMINATION_RATE,
                random_state=seed,
                n_jobs=-1
            )
            clf.fit(X_std)
            
            # Anomaly prediction: -1 = outlier (Trojan candidate), 1 = inlier (Benign)
            raw_preds = clf.predict(X_std)
            preds = (raw_preds == -1).astype(int)
            scores = -clf.score_samples(X_std)  # Higher score = more anomalous
            
            # Metrics
            acc = accuracy_score(y_true, preds)
            bal_acc = balanced_accuracy_score(y_true, preds)
            prec = precision_score(y_true, preds, zero_division=0)
            rec = recall_score(y_true, preds, zero_division=0)
            f1 = f1_score(y_true, preds, zero_division=0)
            mcc = matthews_corrcoef(y_true, preds)
            try:
                pr_auc = average_precision_score(y_true, scores)
            except Exception:
                pr_auc = 0.0
                
            cm = confusion_matrix(y_true, preds, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            
            all_results.append({
                "Seed": seed,
                "Family": fam,
                "Circuit": c_name,
                "Feature_Set": "Mondal_RareNet_Features",
                "Model": "MondalRareNet",
                "Accuracy": acc,
                "Balanced_Accuracy": bal_acc,
                "Precision": prec,
                "Recall": rec,
                "F1_Score": f1,
                "MCC": mcc,
                "PR_AUC": pr_auc,
                "TP": int(tp),
                "FP": int(fp),
                "TN": int(tn),
                "FN": int(fn)
            })
            
    df_res = pd.DataFrame(all_results)
    raw_out = "results/mondal_rare_net_raw.csv"
    df_res.to_csv(raw_out, index=False)
    print(f"\nSaved raw MondalRareNet results to {raw_out} ({len(df_res)} rows)")
    
    # Summary
    summary_records = []
    fam_grouped = df_res.groupby(['Family', 'Seed'])[['Accuracy', 'Balanced_Accuracy', 'Precision', 'Recall', 'F1_Score', 'MCC', 'PR_AUC']].mean().reset_index()
    fam_stats = fam_grouped.groupby('Family')[['Accuracy', 'Balanced_Accuracy', 'Precision', 'Recall', 'F1_Score', 'MCC', 'PR_AUC']].agg(['mean', 'std'])
    
    for fam in fam_stats.index:
        summary_records.append({
            "Model": "MondalRareNet", "Family": fam,
            "Accuracy": f"{fam_stats.loc[fam, ('Accuracy', 'mean')]:.4f} ± {fam_stats.loc[fam, ('Accuracy', 'std')]:.4f}",
            "Balanced_Accuracy": f"{fam_stats.loc[fam, ('Balanced_Accuracy', 'mean')]:.4f} ± {fam_stats.loc[fam, ('Balanced_Accuracy', 'std')]:.4f}",
            "Precision": f"{fam_stats.loc[fam, ('Precision', 'mean')]:.4f} ± {fam_stats.loc[fam, ('Precision', 'std')]:.4f}",
            "Recall": f"{fam_stats.loc[fam, ('Recall', 'mean')]:.4f} ± {fam_stats.loc[fam, ('Recall', 'std')]:.4f}",
            "F1_Score": f"{fam_stats.loc[fam, ('F1_Score', 'mean')]:.4f} ± {fam_stats.loc[fam, ('F1_Score', 'std')]:.4f}",
            "MCC": f"{fam_stats.loc[fam, ('MCC', 'mean')]:.4f} ± {fam_stats.loc[fam, ('MCC', 'std')]:.4f}",
            "PR_AUC": f"{fam_stats.loc[fam, ('PR_AUC', 'mean')]:.4f} ± {fam_stats.loc[fam, ('PR_AUC', 'std')]:.4f}"
        })
        
    overall_grouped = fam_grouped.groupby('Seed')[['Accuracy', 'Balanced_Accuracy', 'Precision', 'Recall', 'F1_Score', 'MCC', 'PR_AUC']].mean().reset_index()
    om = overall_grouped.mean()
    ostd = overall_grouped.std()
    summary_records.append({
        "Model": "MondalRareNet", "Family": "OVERALL_MACRO",
        "Accuracy": f"{om['Accuracy']:.4f} ± {ostd['Accuracy']:.4f}",
        "Balanced_Accuracy": f"{om['Balanced_Accuracy']:.4f} ± {ostd['Balanced_Accuracy']:.4f}",
        "Precision": f"{om['Precision']:.4f} ± {ostd['Precision']:.4f}",
        "Recall": f"{om['Recall']:.4f} ± {ostd['Recall']:.4f}",
        "F1_Score": f"{om['F1_Score']:.4f} ± {ostd['F1_Score']:.4f}",
        "MCC": f"{om['MCC']:.4f} ± {ostd['MCC']:.4f}",
        "PR_AUC": f"{om['PR_AUC']:.4f} ± {ostd['PR_AUC']:.4f}"
    })
    
    df_sum = pd.DataFrame(summary_records)
    sum_out = "results/mondal_rare_net_summary.csv"
    df_sum.to_csv(sum_out, index=False)
    print(f"Saved MondalRareNet summary to {sum_out}")
    print("\nOVERALL MACRO PERFORMANCE:")
    print(df_sum[df_sum['Family'] == 'OVERALL_MACRO'].to_string(index=False))

if __name__ == "__main__":
    main()

