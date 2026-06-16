import os
import csv
import torch
import numpy as np
import pandas as pd
from torch_geometric.utils import to_undirected
from parser import NetlistGraphBuilder
from simulator import LogicSimulator

def load_10k_vectors(filepath, num_primary_inputs):
    """Loads the 10k 8-bit vectors. Pads remaining PIs with random 0/1."""
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        
    num_vectors = len(lines)
    # Convert string lines to a numpy array of shape (8, num_vectors)
    vector_array = np.zeros((8, num_vectors), dtype=int)
    for i, line in enumerate(lines):
        for j, bit in enumerate(line[:8]):
            vector_array[j, i] = int(bit)
            
    # Pad to num_primary_inputs if needed
    if num_primary_inputs > 8:
        padding = np.random.randint(0, 2, size=(num_primary_inputs - 8, num_vectors))
        vector_array = np.vstack([vector_array, padding])
        
    return vector_array, num_vectors

def run_simulation(files, top_module, vectors_filepath):
    print(f"Parsing {files}...")
    builder = NetlistGraphBuilder(files, top_module=top_module)
    graph = builder.build()
    
    sim = LogicSimulator(graph, num_vectors=10000)
    
    # Load custom vectors
    custom_vectors, actual_num = load_10k_vectors(vectors_filepath, len(sim.primary_inputs))
    sim.vectors = custom_vectors
    sim.num_vectors = actual_num
    
    print(f"Simulating {actual_num} vectors on {graph.number_of_nodes()} nodes...")
    sim.simulate()
    pyg_data = sim.embed_tp_and_export()
    
    tp_dict = {}
    for node in graph.nodes():
        tp_dict[node] = graph.nodes[node].get('TP', 0.0)
        
    return pyg_data, tp_dict, graph

def compute_and_save_dataset(pyg_data, ht_tp_dict, clean_tp_dict, node_names, output_csv):
    num_nodes = pyg_data.num_nodes
    
    x_new = torch.zeros((num_nodes, 8), dtype=torch.float)
    tp_tensor = torch.zeros(num_nodes)
    tp_diff_tensor = torch.zeros(num_nodes)
    
    for idx, node in enumerate(node_names):
        val_ht = ht_tp_dict.get(node, 0.0)
        val_norm = clean_tp_dict.get(node, 0.0)
        
        tp_tensor[idx] = val_ht
        tp_diff_tensor[idx] = abs(val_ht - val_norm)
        
    x_new[:, 0] = tp_tensor                          # f1
    x_new[:, 1] = tp_diff_tensor                     # f2
    x_new[:, 2] = 1.0 - tp_tensor                    # f3
    
    edge_index = pyg_data.edge_index
    if edge_index.numel() > 0:
        x_new[:, 6] = torch.bincount(edge_index[1], minlength=num_nodes).float() # f7 (Fanin)
        x_new[:, 7] = torch.bincount(edge_index[0], minlength=num_nodes).float() # f8 (Fanout)
        
        row, col = to_undirected(edge_index)
        for i in range(num_nodes):
            neighbors = col[row == i]
            if len(neighbors) > 0:
                n_tps = tp_tensor[neighbors]
                mean_val = n_tps.mean()
                std_val = n_tps.std(unbiased=False) 
                
                x_new[i, 3] = mean_val                   # f4
                x_new[i, 4] = torch.abs(tp_tensor[i] - n_tps).sum() # f5
                if std_val > 1e-7:
                    x_new[i, 5] = (tp_tensor[i] - mean_val) / std_val # f6
                else:
                    x_new[i, 5] = 0.0
                    
    # Assign Labels
    y_list = []
    for node in node_names:
        if any(sub in node for sub in ["Trojan.", "Tj", "tj"]):
            y_list.append(1)
        else:
            # Fallback heuristic for synthesized netlists: 
            # If node exists in HT but NOT in normal circuit, it's highly likely part of the Trojan.
            if node not in clean_tp_dict:
                y_list.append(1)
            else:
                y_list.append(0)
                
    # Save CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Node', 'f1_TP', 'f2_TPDiff', 'f3_Rare', 'f4_NbMean', 'f5_NbDist', 'f6_LZ', 'f7_FanIn', 'f8_FanOut', 'Label'])
        for i in range(num_nodes):
            writer.writerow([
                node_names[i],
                f"{x_new[i,0].item():.6f}", f"{x_new[i,1].item():.6f}", f"{x_new[i,2].item():.6f}",
                f"{x_new[i,3].item():.6f}", f"{x_new[i,4].item():.6f}", f"{x_new[i,5].item():.6f}",
                f"{x_new[i,6].item():.0f}", f"{x_new[i,7].item():.0f}",
                y_list[i]
            ])
    print(f"--> Saved pristine dataset to {output_csv}")

def main():
    base_dir = "/Users/sudiptomaity/projects/TrojanNet-GNN/rs232"
    vectors_file = os.path.join(base_dir, "rs232_fixed_patterns_10k.txt")
    output_dir = "/Users/sudiptomaity/projects/TrojanNet-GNN/output_datasets"
    
    # 1. Process Normal Circuit First
    print("\n[PHASE 1] PROCESSING NORMAL CIRCUIT")
    normal_files = [
        os.path.join(base_dir, "normal", "uart.v"),
        os.path.join(base_dir, "normal", "u_rec.v"),
        os.path.join(base_dir, "normal", "u_xmit.v")
    ]
    normal_pyg, clean_tp_dict, clean_graph = run_simulation(normal_files, "uart", vectors_file)
    
    # Save Normal Dataset
    normal_csv = os.path.join(output_dir, "dataset_RS232_normal.csv")
    compute_and_save_dataset(normal_pyg, clean_tp_dict, clean_tp_dict, list(clean_graph.nodes()), normal_csv)
    
    # 2. Process Trojans
    print("\n[PHASE 2] PROCESSING TROJANS")
    trojans_dir = os.path.join(base_dir, "trojans")
    
    for trojan_folder in sorted(os.listdir(trojans_dir)):
        tf_path = os.path.join(trojans_dir, trojan_folder)
        if not os.path.isdir(tf_path) or not trojan_folder.startswith("T"):
            continue
            
        # Categorize by T-number
        t_num = int(trojan_folder[1:])
        
        print(f"\n--- Processing {trojan_folder} ---")
            
        if (100 <= t_num <= 901) or (2100 <= t_num <= 2400):
            files = [
                os.path.join(tf_path, "uart.v"),
                os.path.join(tf_path, "u_rec.v"),
                os.path.join(tf_path, "u_xmit.v")
            ]
            if all(os.path.exists(f) for f in files):
                pyg, ht_tp, g = run_simulation(files, "uart", vectors_file)
                out_csv = os.path.join(output_dir, f"dataset_RS232_{trojan_folder}.csv")
                compute_and_save_dataset(pyg, ht_tp, clean_tp_dict, list(g.nodes()), out_csv)
                
        elif 1000 <= t_num <= 1600:
            file_90 = os.path.join(tf_path, "90nm_uart.v")
            file_180 = os.path.join(tf_path, "180nm_uart.v")
            
            if os.path.exists(file_90):
                pyg, ht_tp, g = run_simulation([file_90], "uart", vectors_file)
                out_csv = os.path.join(output_dir, f"dataset_RS232_{trojan_folder}_90nm.csv")
                compute_and_save_dataset(pyg, ht_tp, clean_tp_dict, list(g.nodes()), out_csv)
                
            if os.path.exists(file_180):
                pyg, ht_tp, g = run_simulation([file_180], "uart", vectors_file)
                out_csv = os.path.join(output_dir, f"dataset_RS232_{trojan_folder}_180nm.csv")
                compute_and_save_dataset(pyg, ht_tp, clean_tp_dict, list(g.nodes()), out_csv)
                
        elif 1700 <= t_num <= 2000:
            file_90 = os.path.join(tf_path, "90nm_uart_scan_route.v")
            file_180 = os.path.join(tf_path, "180nm_uart_scan_route.v")
            
            if os.path.exists(file_90):
                pyg, ht_tp, g = run_simulation([file_90], "uart", vectors_file)
                out_csv = os.path.join(output_dir, f"dataset_RS232_{trojan_folder}_90nm_scan.csv")
                compute_and_save_dataset(pyg, ht_tp, clean_tp_dict, list(g.nodes()), out_csv)
                
            if os.path.exists(file_180):
                pyg, ht_tp, g = run_simulation([file_180], "uart", vectors_file)
                out_csv = os.path.join(output_dir, f"dataset_RS232_{trojan_folder}_180nm_scan.csv")
                compute_and_save_dataset(pyg, ht_tp, clean_tp_dict, list(g.nodes()), out_csv)

if __name__ == "__main__":
    main()
