import os
import pandas as pd
import numpy as np

def check_aes_datasets():
    node_dir = "/Users/sudiptomaity/projects/TrojanNet-GNN/aes_output_datasets"
    edge_dir = "/Users/sudiptomaity/projects/TrojanNet-GNN/aes_edges_datasets"
    
    node_files = [f for f in os.listdir(node_dir) if f.endswith('.csv')]
    
    total_errors = 0
    total_files = len(node_files)
    
    print("==================================================")
    print(f"STARTING ULTIMATE ANOMALY CHECK ACROSS {total_files} AES DATASETS")
    print("==================================================")
    
    for file in sorted(node_files):
        node_path = os.path.join(node_dir, file)
        edge_filename = file.replace("dataset_AES", "edges_AES")
        edge_path = os.path.join(edge_dir, edge_filename)
        
        errors = 0
        df = pd.read_csv(node_path)
        
        # 1. Null / NaN Check
        if df.isnull().values.any():
            print(f"[FAIL] {file}: Contains NaNs or Nulls")
            errors += 1
            
        # 2. Duplicate Node Check
        if df['Node'].duplicated().any():
            print(f"[FAIL] {file}: Contains duplicate node names")
            errors += 1
            
        # 3. Label Value Check
        if not df['Label'].isin([0, 1]).all():
            print(f"[FAIL] {file}: Label contains values other than 0 and 1")
            errors += 1
            
        # 4. Feature Bounds Check [0, 1]
        for col in ['f1_TP', 'f2_TPDiff', 'f3_Rare', 'f4_NbMean']:
            if (df[col] < 0).any() or (df[col] > 1).any():
                print(f"[FAIL] {file}: {col} out of bounds [0, 1]")
                errors += 1
                
        # 5. Math Consistency Check (f3 = 1 - f1)
        f3_diff = np.abs(df['f3_Rare'] - (1.0 - df['f1_TP']))
        if (f3_diff > 1e-5).any():
            print(f"[FAIL] {file}: Math violation f3_Rare != 1 - f1_TP")
            errors += 1
            
        # 6. Edge Integrity Check
        if not os.path.exists(edge_path):
            print(f"[FAIL] {file}: Missing corresponding edge file {edge_filename}")
            errors += 1
        else:
            df_edges = pd.read_csv(edge_path)
            node_names = set(df['Node'].values)
            edge_srcs = set(df_edges['src'].values)
            edge_dsts = set(df_edges['dst'].values)
            
            # Check if edges contain nodes NOT in the features list
            unknown_srcs = edge_srcs - node_names
            unknown_dsts = edge_dsts - node_names
            if unknown_srcs or unknown_dsts:
                print(f"[FAIL] {file}: Edge list contains nodes that do not exist in the features dataset! ({len(unknown_srcs)} src, {len(unknown_dsts)} dst)")
                errors += 1
                
            # Basic sanity check on edges
            if len(df_edges) == 0 and len(df) > 10:
                print(f"[FAIL] {file}: Edge list is empty for a graph with {len(df)} nodes!")
                errors += 1
                
        total_errors += errors

    print("==================================================")
    if total_errors == 0:
        print("ALL DATASETS PASSED EVERY MATHEMATICAL AND STRUCTURAL CHECK.")
        print("STATUS: PRISTINE (0 ERRORS)")
    else:
        print(f"DETECTED {total_errors} TOTAL ERRORS.")
    print("==================================================")

if __name__ == "__main__":
    check_aes_datasets()
