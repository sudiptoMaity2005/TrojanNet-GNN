import torch
import pandas as pd
import numpy as np
from torch_geometric.utils import to_undirected

def generate_custom_features(pt_file_path, ht_csv_path, clean_csv_path, output_pt_path):
    """
    Computes the 8 custom mathematical features for each node in the Graph
    and replaces the existing node features (data.x) with this new feature vector.
    """
    print(f"Loading Graph Data from: {pt_file_path}...")
    data = torch.load(pt_file_path, weights_only=False)
    num_nodes = data.num_nodes
    
    # 1. Load the TP CSVs into dictionaries mapping Node Name -> TP
    df_ht = pd.read_csv(ht_csv_path)
    df_norm = pd.read_csv(clean_csv_path)
    
    tp_ht = dict(zip(df_ht['Node'], df_ht['TP']))
    tp_norm = dict(zip(df_norm['Node'], df_norm['TP']))
    
    # We assume the order of nodes in df_ht perfectly matches the node indexing in the .pt graph.
    # Our extraction pipeline guarantees this.
    node_names = df_ht['Node'].tolist()
    
    # Initialize the new Feature Matrix: Shape = [num_nodes, 8]
    x_new = torch.zeros((num_nodes, 8), dtype=torch.float)
    
    # Basic Features
    tp_tensor = torch.zeros(num_nodes)
    tp_diff_tensor = torch.zeros(num_nodes)
    
    for idx, node in enumerate(node_names):
        val_ht = tp_ht.get(node, 0.0)
        # If the Trojan node doesn't exist in the Normal circuit, TP_Normal is 0.
        val_norm = tp_norm.get(node, 0.0) 
        
        tp_tensor[idx] = val_ht
        tp_diff_tensor[idx] = abs(val_ht - val_norm)
        
    # Apply Mathematical Formulas
    x_new[:, 0] = tp_tensor                          # f1 = TP_i
    x_new[:, 1] = tp_diff_tensor                     # f2 = |TP_i^HT - TP_i^Normal|
    x_new[:, 2] = 1.0 - tp_tensor                    # f3 = 1 - TP_i
    
    # Topological Neighborhood Features
    edge_index = data.edge_index
    # Convert graph to undirected to find the complete neighborhood N(i)
    undirected_edges = to_undirected(edge_index)
    row, col = undirected_edges
    
    # Compute Fanin & Fanout from the ORIGINAL directed edges
    in_degree = torch.bincount(edge_index[1], minlength=num_nodes).float()
    out_degree = torch.bincount(edge_index[0], minlength=num_nodes).float()
    
    x_new[:, 6] = in_degree                          # f7 = Fanin_i
    x_new[:, 7] = out_degree                         # f8 = Fanout_i
    
    # Calculate Neighborhood Metrics iteratively
    # (Doing this natively avoids difficult torch_scatter pip installs in Colab)
    for i in range(num_nodes):
        # Find all neighbor indices for node i
        neighbors = col[row == i]
        
        if len(neighbors) > 0:
            n_tps = tp_tensor[neighbors]
            mean_val = n_tps.mean()
            # Unbiased=False to strictly follow the formula: 1/|N(i)| * sum(...)
            std_val = n_tps.std(unbiased=False) 
            dist_sum = torch.abs(tp_tensor[i] - n_tps).sum()
            
            x_new[i, 3] = mean_val                   # f4 = Neighborhood Mean
            x_new[i, 4] = dist_sum                   # f5 = Neighborhood Disturbance
            
            # f6 = Local Z-Score (Protect against division by zero if std == 0)
            if std_val > 1e-7:
                x_new[i, 5] = (tp_tensor[i] - mean_val) / std_val
            else:
                x_new[i, 5] = 0.0
                
    # Initialize the Labels (y) vector
    # 1 if it's a Trojan node, 0 if it's a Normal node
    y_tensor = torch.zeros(num_nodes, dtype=torch.long)
    for idx, node in enumerate(node_names):
        # In RS232, the trojan is typically instantiated as TjRS232 or similar.
        # In AES T100, the trojan is instantiated as "Trojan".
        # We can detect Trojan nodes by looking for "Trojan." or "Tj" or "tj" in the node hierarchy
        # or checking if TP_norm is exactly 0.0 while TP_ht is present, but substring is safer.
        is_trojan = any(sub in node for sub in ["Trojan.", "Tj", "tj"])
        if is_trojan:
            y_tensor[idx] = 1
            
    # Overwrite the graph features and save
    data.x = x_new
    data.y = y_tensor
    torch.save(data, output_pt_path)
    print(f"Feature computation complete! New tensor saved to {output_pt_path} with X shape {data.x.shape} and y shape {data.y.shape}")

if __name__ == "__main__":
    import os
    # Process RS232 T300
    if os.path.exists('RS232_circuit/rs232_RS232 T300_data.pt'):
        print("--- Processing RS232 T300 ---")
        generate_custom_features(
            pt_file_path='RS232_circuit/rs232_RS232 T300_data.pt', 
            ht_csv_path='RS232_circuit/tp_output_T300.csv', 
            clean_csv_path='RS232_circuit/tp_output_clean.csv', 
            output_pt_path='final_dataset_rs232_T300.pt'
        )
        
    # Process RS232 T400
    if os.path.exists('RS232_circuit/rs232_RS232 T400_data.pt'):
        print("--- Processing RS232 T400 ---")
        generate_custom_features(
            pt_file_path='RS232_circuit/rs232_RS232 T400_data.pt', 
            ht_csv_path='RS232_circuit/tp_output_T400.csv', 
            clean_csv_path='RS232_circuit/tp_output_clean.csv', 
            output_pt_path='final_dataset_rs232_T400.pt'
        )
    
    # Process AES T100
    if os.path.exists('AES_circuit/aes_AES T100_data.pt'):
        print("\n--- Processing AES T100 ---")
        generate_custom_features(
            pt_file_path='AES_circuit/aes_AES T100_data.pt', 
            ht_csv_path='AES_circuit/tp_output_aes_T100.csv', 
            clean_csv_path='AES_circuit/tp_output_aes_clean.csv', 
            output_pt_path='final_dataset_aes_T100.pt'
        )
    
    print("\nAll datasets generated successfully!")
