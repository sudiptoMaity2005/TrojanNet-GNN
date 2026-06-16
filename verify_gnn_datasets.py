import os
import pandas as pd
import glob
import numpy as np

def verify_datasets():
    base_dir = "/Users/sudiptomaity/projects/TrojanNet-GNN/datasets_for_gnn"
    node_files = glob.glob(f"{base_dir}/*/*/nodes/*.csv")
    
    total_files = len(node_files)
    total_errors = 0
    
    print("==================================================")
    print(f"STARTING STRICT VALIDATION ACROSS {total_files} DATASETS")
    print("==================================================")
    
    required_cols = ['Node', 'f1_TP', 'f2_TPDiff', 'f3_Rare', 'f4_NbMean', 'f5_NbDist', 'f6_LZ', 'f7_FanIn', 'f8_FanOut', 'Label']
    
    for n_csv in sorted(node_files):
        # Determine paths
        basename = os.path.basename(n_csv)
        e_csv = n_csv.replace('/nodes/', '/edges/').replace('dataset_', 'edges_')
        circuit_name = basename.replace('.csv', '')
        
        errors = 0
        try:
            df = pd.read_csv(n_csv)
            
            # 1. Null Check
            if df.isnull().values.any():
                print(f"[ERROR] {circuit_name}: Contains NaNs or Nulls")
                errors += 1
                
            # 2. Column Check
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                print(f"[ERROR] {circuit_name}: Missing columns {missing_cols}")
                errors += 1
                
            # 3. Label Value Check
            if not df['Label'].isin([0, 1]).all():
                print(f"[ERROR] {circuit_name}: Labels contain values other than 0 or 1")
                errors += 1
                
            # 4. Duplicate Nodes
            if df['Node'].duplicated().any():
                print(f"[ERROR] {circuit_name}: Contains duplicate node names")
                errors += 1
                
            # 5. Check Edge File Exists
            if not os.path.exists(e_csv):
                print(f"[ERROR] {circuit_name}: Missing corresponding edge CSV -> {e_csv}")
                errors += 1
            else:
                df_e = pd.read_csv(e_csv)
                if 'src' not in df_e.columns or 'dst' not in df_e.columns:
                    print(f"[ERROR] {circuit_name}: Edge CSV missing 'src' or 'dst' columns")
                    errors += 1
                else:
                    # Optional: Check if edges reference valid nodes
                    valid_nodes = set(df['Node'].astype(str))
                    invalid_src = df_e[~df_e['src'].astype(str).isin(valid_nodes)]
                    invalid_dst = df_e[~df_e['dst'].astype(str).isin(valid_nodes)]
                    
                    if not invalid_src.empty or not invalid_dst.empty:
                        print(f"[ERROR] {circuit_name}: Edges reference nodes not in Node CSV!")
                        errors += 1
            
            if errors == 0:
                pass # print(f"[PASS] {circuit_name}")
                
        except Exception as e:
            print(f"[ERROR] Exception reading {circuit_name}: {e}")
            errors += 1
            
        total_errors += errors
        
    print("==================================================")
    if total_errors == 0:
        print(f"SUCCESS: ALL {total_files} DATASETS PASSED WITH 0 ERRORS!")
    else:
        print(f"FAILED: FOUND {total_errors} TOTAL ERRORS!")
    print("==================================================")

if __name__ == "__main__":
    verify_datasets()
