import os
import pandas as pd

def check_missing():
    folder = "/Users/sudiptomaity/projects/TrojanNet-GNN/aes_output_datasets"
    
    df_norm = pd.read_csv(os.path.join(folder, "dataset_AES_normal.csv"))
    norm_nodes = set(df_norm['Node'])
    
    df_t = pd.read_csv(os.path.join(folder, "dataset_AES_T100.csv"))
    t_nodes = set(n[4:] if n.startswith("AES.") else n for n in df_t['Node'])
    
    missing = norm_nodes - t_nodes
    print(f"Missing {len(missing)} nodes.")
    print("Examples:")
    for m in list(missing)[:20]:
        print(m)

if __name__ == "__main__":
    check_missing()
