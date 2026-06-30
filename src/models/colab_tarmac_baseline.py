import os
import sys
import glob
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==============================================================================
# 1. Dataset Loading
# ==============================================================================

def load_tarmac_dataset(csv_path):
    df = pd.read_csv(csv_path)
    # The first feature is f1_TP (Trigger Probability from TARMAC)
    tp_scores = df['f1_TP'].values
    labels = df['Label'].values
    num_nodes = len(labels)
    return tp_scores, labels, num_nodes

def find_best_tarmac_threshold(train_data_list):
    """
    Original TARMAC strictly uses Trigger Probabilities (TP).
    This function finds the mathematically optimal TP threshold 
    on the training set to maximize F1-score.
    """
    if not train_data_list:
        return 0.5
        
    all_tp = []
    all_labels = []
    for tp_scores, labels, _ in train_data_list:
        all_tp.extend(tp_scores)
        all_labels.extend(labels)
        
    all_tp = np.array(all_tp)
    all_labels = np.array(all_labels)
    
    best_f1 = -1
    best_thresh = 0.5
    
    # Search 100 possible thresholds between min and max TP
    thresholds = np.linspace(all_tp.min(), all_tp.max(), 100)
    for t in thresholds:
        preds = (all_tp >= t).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            
    return best_thresh

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    base_dir = "datasets_for_gnn"
    sizes = ["small", "medium", "Large"]
    
    if not os.path.exists(base_dir):
        print("ERROR: Could not find datasets_for_gnn folder! Please unzip it in Colab.")
        sys.exit()

    print("Loading Datasets for TARMAC Baseline Evaluation...")
    
    circuit_families = {}
    
    for size in sizes:
        # Load Train
        train_nodes = glob.glob(f"{base_dir}/{size}/train/nodes/*.csv")
        for n_csv in train_nodes:
            basename = os.path.basename(n_csv).replace("dataset_", "").replace(".csv", "")
            circ_name = basename.split('_')[0]
            if circ_name not in circuit_families:
                circuit_families[circ_name] = {'size': size, 'train': [], 'test': []}
            circuit_families[circ_name]['train'].append(load_tarmac_dataset(n_csv))
                
        # Load Test
        test_nodes = glob.glob(f"{base_dir}/{size}/test/nodes/*.csv")
        for n_csv in test_nodes:
            basename = os.path.basename(n_csv).replace("dataset_", "").replace(".csv", "")
            circ_name = basename.split('_')[0]
            if circ_name not in circuit_families:
                circuit_families[circ_name] = {'size': size, 'train': [], 'test': []}
            circuit_families[circ_name]['test'].append(load_tarmac_dataset(n_csv))

    print("\n===========================================================================")
    print("BASELINE EVALUATION: ORIGINAL TARMAC (No Graph Topology)")
    print("===========================================================================")
    print("| Circuit Size | Circuit Name | Optimal TP Threshold | Precision | Recall | F1-Score |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    csv_data = []

    for circ_name, data_dict in circuit_families.items():
        train_data = data_dict['train']
        test_data = data_dict['test']
        size = data_dict['size']
        
        if len(train_data) == 0 or len(test_data) == 0:
            continue
            
        # Find the mathematically best threshold for this circuit using TARMAC probs
        best_thresh = find_best_tarmac_threshold(train_data)
        
        stats = []
        for tp_scores, labels, num_nodes in test_data:
            
            # Predict using ONLY the thresholded Trigger Probability (Original TARMAC)
            preds = (tp_scores >= best_thresh).astype(int)
            
            prec = precision_score(labels, preds, zero_division=0)
            rec = recall_score(labels, preds, zero_division=0)
            f1 = f1_score(labels, preds, zero_division=0)
            
            stats.append((prec, rec, f1))
            
        # Average metrics
        avg_prec = np.mean([s[0] for s in stats])
        avg_rec = np.mean([s[1] for s in stats])
        avg_f1 = np.mean([s[2] for s in stats])
        
        display_size = "Small" if size == "small" else "Medium" if size == "medium" else "Large"
        
        print(f"| **{display_size}** | **{circ_name}** | {best_thresh:.4f} | {avg_prec*100:.1f}% | {avg_rec*100:.1f}% | {avg_f1:.2f} |")
        
        csv_data.append({
            "Circuit Size": display_size,
            "Circuit Name": circ_name,
            "Optimal TP Threshold": best_thresh,
            "Precision (%)": avg_prec * 100,
            "Recall (%)": avg_rec * 100,
            "F1-Score": avg_f1
        })

    # Save to CSV
    df_results = pd.DataFrame(csv_data)
    csv_path = "baseline_tarmac_results.csv"
    df_results.to_csv(csv_path, index=False)
    
    print("\n[INFO] Copy these Baseline F1-Scores and compare them directly against your GNN Table.")
    print(f"[INFO] Baseline evaluation results saved to {csv_path}!")
