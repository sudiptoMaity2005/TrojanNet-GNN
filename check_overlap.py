import os
import pandas as pd

def check_overlap():
    folder = "/Users/sudiptomaity/projects/TrojanNet-GNN/aes_output_datasets"
    
    df_norm = pd.read_csv(os.path.join(folder, "dataset_AES_normal.csv"))
    norm_nodes = set(df_norm['Node'])
    
    print(f"Normal nodes: {len(norm_nodes)}")
    
    for t in ['T100', 'T2600']:
        df_t = pd.read_csv(os.path.join(folder, f"dataset_AES_{t}.csv"))
        t_nodes = df_t['Node']
        
        # Test exact match
        exact_matches = sum(1 for n in t_nodes if n in norm_nodes)
        
        # Test stripped match
        stripped_matches = sum(1 for n in t_nodes if (n[4:] if n.startswith("AES.") else n) in norm_nodes)
        
        print(f"--- {t} ---")
        print(f"Total nodes: {len(t_nodes)}")
        print(f"Exact matches: {exact_matches}")
        print(f"Stripped matches: {stripped_matches}")

if __name__ == "__main__":
    check_overlap()
