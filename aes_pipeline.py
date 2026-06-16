import os
import csv
import torch
import numpy as np
from torch_geometric.utils import to_undirected
from parser import NetlistGraphBuilder
from simulator import LogicSimulator

def load_10k_vectors(filepath, num_primary_inputs):
    """Loads the vectors. Pads remaining PIs with random 0/1."""
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        
    num_vectors = len(lines)
    vector_array = np.zeros((num_primary_inputs, num_vectors), dtype=int)
    for i, line in enumerate(lines):
        for j in range(min(len(line), num_primary_inputs)):
            vector_array[j, i] = int(line[j])
            
    # Pad to num_primary_inputs if needed
    if num_primary_inputs > len(lines[0]):
        padding = np.random.randint(0, 2, size=(num_primary_inputs - len(lines[0]), num_vectors))
        vector_array[len(lines[0]):, :] = padding
        
    return vector_array, num_vectors

def run_simulation(files, top_module, vectors_filepath):
    print(f"Parsing {files}...")
    builder = NetlistGraphBuilder(files, top_module=top_module)
    graph = builder.build()
    
    # If top wasn't found, parser falls back to first module. Let's make sure it picks aes_128 or top
    if builder.top_module not in ['top', 'aes_128']:
        if 'top' in builder.modules:
            builder.top_module = 'top'
        elif 'aes_128' in builder.modules:
            builder.top_module = 'aes_128'
        # Re-flatten if we changed it
        builder.graph.clear()
        builder._flatten_module(builder.top_module, instance_prefix="")
        builder.extract_features()
        graph = builder.graph
    
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
        mapped_node = node
        if mapped_node.startswith("AES."):
            mapped_node = mapped_node[4:]
        elif mapped_node.startswith("aes_128_0."):
            mapped_node = mapped_node[10:]
            
        val_ht = ht_tp_dict.get(node, 0.0)
        val_norm = clean_tp_dict.get(mapped_node, 0.0)
        
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
        if any(sub in node for sub in ["Trojan", "Tj", "tj", "TSC", "lfsr", "AM_Transmission"]):
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
    base_dir = "/Users/sudiptomaity/projects/TrojanNet-GNN/aes"
    vectors_file = os.path.join(base_dir, "aes_fixed_patterns_10k.txt")
    output_dir = "/Users/sudiptomaity/projects/TrojanNet-GNN/aes_output_datasets"
    
    # 1. Process Normal Circuit First
    print("\n[PHASE 1] PROCESSING NORMAL CIRCUIT")
    normal_dir = os.path.join(base_dir, "normal")
    normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith('.v')]
    normal_pyg, clean_tp_dict, clean_graph = run_simulation(normal_files, "top", vectors_file)
    
    # Save Normal Dataset
    normal_csv = os.path.join(output_dir, "dataset_AES_normal.csv")
    compute_and_save_dataset(normal_pyg, clean_tp_dict, clean_tp_dict, list(clean_graph.nodes()), normal_csv)
    
    # 2. Process Trojans
    print("\n[PHASE 2] PROCESSING TROJANS")
    trojans_dir = os.path.join(base_dir, "trojan")
    
    for trojan_folder in sorted(os.listdir(trojans_dir)):
        tf_path = os.path.join(trojans_dir, trojan_folder)
        if not os.path.isdir(tf_path) or not trojan_folder.startswith("AES T"):
            continue
            
        print(f"\n--- Processing {trojan_folder} ---")
        
        files = [os.path.join(tf_path, f) for f in os.listdir(tf_path) if f.endswith('.v')]
        if len(files) > 0:
            try:
                pyg, ht_tp, g = run_simulation(files, "top", vectors_file)
                out_csv = os.path.join(output_dir, f"dataset_{trojan_folder.replace(' ', '_')}.csv")
                compute_and_save_dataset(pyg, ht_tp, clean_tp_dict, list(g.nodes()), out_csv)
            except Exception as e:
                print(f"Error processing {trojan_folder}: {e}")

if __name__ == "__main__":
    main()
