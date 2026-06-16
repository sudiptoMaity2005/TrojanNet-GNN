import os
import glob
import pandas as pd

def fix_labels():
    base_dir = "/Users/sudiptomaity/projects/TrojanNet-GNN/datasets_for_gnn"
    node_files = glob.glob(f"{base_dir}/*/*/nodes/*.csv")
    
    total_files = len(node_files)
    fixed = 0
    total_positives = 0
    
    trojan_keywords = ["troj", "tj", "trigger", "tempn", "tsc", "lfsr", "am_transmission", "t_"]
    
    for n_csv in node_files:
        df = pd.read_csv(n_csv)
        original_sum = df['Label'].sum()
        
        # Recalculate labels based on node names
        def is_trojan(node_name):
            node_str = str(node_name).lower()
            return int(any(k in node_str for k in trojan_keywords))
            
        df['Label'] = df['Node'].apply(is_trojan)
        new_sum = df['Label'].sum()
        
        total_positives += new_sum
        
        if new_sum != original_sum:
            df.to_csv(n_csv, index=False)
            fixed += 1
            print(f"Fixed {os.path.basename(n_csv)}: {original_sum} -> {new_sum} Trojans")

    print("==================================================")
    print(f"LABEL FIX COMPLETE. Fixed {fixed}/{total_files} files.")
    print(f"Total Positive Labels across all datasets: {total_positives}")
    print("==================================================")

if __name__ == "__main__":
    fix_labels()
