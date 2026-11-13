import os
import glob
import pandas as pd
import numpy as np
import scipy.stats as stats
import torch

def calculate_topological_metrics(edge_index, num_nodes):
    # edge_index is shape [2, E], where [0] is src, [1] is dst
    if edge_index.shape[1] == 0:
        return 0.0, 0.0
        
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    
    # Calculate out-degrees (Fan-out)
    out_degrees = np.bincount(src, minlength=num_nodes)
    fanout_variance = np.var(out_degrees)
    
    # Calculate in-degrees (Fan-in)
    in_degrees = np.bincount(dst, minlength=num_nodes)
    fanin_variance = np.var(in_degrees)
    
    return fanin_variance, fanout_variance

def main():
    print("Priority 2: Topology Measurement & Correlation")
    
    # 1. Load F1 scores from LOCO results
    # Checking possible results files
    results_file = "trittc_hard_threshold_tuned (2).csv"
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found.")
        return
        
    df_results = pd.read_csv(results_file)
    # Filter for Random Forest and Extra Trees
    df_rf = df_results[df_results['Model'] == 'RandomForest'].copy()
    df_et = df_results[df_results['Model'] == 'ExtraTrees'].copy()
    
    print(f"Loaded {len(df_rf)} variants for Random Forest.")
    
    # 2. Extract topological metrics for each variant
    variants = df_rf['Circuit'].unique()
    
    variant_metrics = []
    
    # Search for edge csv files
    csv_files = []
    for d in ['datasets_for_gnn', 'other_trit_tc_datasets', 'rest_trit_tc_datasets']:
        csv_files.extend(glob.glob(f"{d}/**/edges_*.csv", recursive=True))
        
    csv_dict = {os.path.basename(f).replace('edges_', '').replace('.csv', ''): f for f in csv_files}
    
    for v in variants:
        if v in csv_dict:
            try:
                df_edges = pd.read_csv(csv_dict[v])
                
                # out-degrees (Fan-out)
                out_degrees = df_edges.groupby('src').size()
                # in-degrees (Fan-in)
                in_degrees = df_edges.groupby('dst').size()
                
                fo_var = np.var(out_degrees.values)
                fi_var = np.var(in_degrees.values)
                
                variant_metrics.append({
                    'Circuit': v,
                    'FanIn_Variance': fi_var,
                    'FanOut_Variance': fo_var
                })
            except Exception as e:
                print(f"Failed to process {v}: {e}")
        else:
            print(f"Warning: edges file for {v} not found.")
            
    df_metrics = pd.DataFrame(variant_metrics)
    
    if len(df_metrics) == 0:
        print("No metrics collected. Exiting.")
        return
        
    # Merge metrics with F1 scores
    df_rf = df_rf.merge(df_metrics, on='Circuit', how='inner')
    df_et = df_et.merge(df_metrics, on='Circuit', how='inner')
    
    # 3. Calculate correlations
    print("\n--- Pearson Correlation Results ---")
    
    # Random Forest
    rf_f1 = df_rf['F1_Score'].values
    rf_fi = df_rf['FanIn_Variance'].values
    rf_fo = df_rf['FanOut_Variance'].values
    
    r_fi_rf, p_fi_rf = stats.pearsonr(rf_fi, rf_f1)
    r_fo_rf, p_fo_rf = stats.pearsonr(rf_fo, rf_f1)
    
    print(f"Random Forest F1 vs. Fan-In Variance : r = {r_fi_rf:.4f}, p-value = {p_fi_rf:.4g}")
    print(f"Random Forest F1 vs. Fan-Out Variance: r = {r_fo_rf:.4f}, p-value = {p_fo_rf:.4g}")
    
    # Extra Trees
    et_f1 = df_et['F1_Score'].values
    et_fi = df_et['FanIn_Variance'].values
    et_fo = df_et['FanOut_Variance'].values
    
    r_fi_et, p_fi_et = stats.pearsonr(et_fi, et_f1)
    r_fo_et, p_fo_et = stats.pearsonr(et_fo, et_f1)
    
    print(f"Extra Trees F1 vs. Fan-In Variance   : r = {r_fi_et:.4f}, p-value = {p_fi_et:.4g}")
    print(f"Extra Trees F1 vs. Fan-Out Variance  : r = {r_fo_et:.4f}, p-value = {p_fo_et:.4g}")
    
    # Save the merged metrics
    df_rf.to_csv("topology_correlation_rf.csv", index=False)
    print("\nSaved topology metrics and F1 scores to topology_correlation_rf.csv")

if __name__ == "__main__":
    main()
