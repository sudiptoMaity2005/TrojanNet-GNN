import os
import csv
import torch
import numpy as np
from torch_geometric.utils import to_undirected
from parser import NetlistGraphBuilder
from simulator import LogicSimulator
import glob
import shutil

def compute_and_save_dataset(pyg_data, tp_dict, node_names, output_node_csv, output_edge_csv):
    num_nodes = pyg_data.num_nodes
    
    x_new = torch.zeros((num_nodes, 8), dtype=torch.float)
    tp_tensor = torch.zeros(num_nodes)
    
    for idx, node in enumerate(node_names):
        val_ht = tp_dict.get(node, 0.0)
        tp_tensor[idx] = val_ht
        
    x_new[:, 0] = tp_tensor                          # f1 (TP)
    x_new[:, 1] = 0.0                                # f2 (TPDiff - Not used for TRIT-TC)
    x_new[:, 2] = 1.0 - tp_tensor                    # f3 (Rare)
    
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
        lower_node = str(node).lower()
        # TRIT-TC standard Trojan naming
        if "trojan" in lower_node or "tj" in lower_node or "_t_" in lower_node:
            y_list.append(1)
        else:
            y_list.append(0)
                
    # Save Node CSV
    with open(output_node_csv, 'w', newline='') as f:
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
            
    # Save Edge CSV
    with open(output_edge_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['src', 'dst'])
        if edge_index.numel() > 0:
            for i in range(edge_index.shape[1]):
                src_idx = edge_index[0, i].item()
                dst_idx = edge_index[1, i].item()
                writer.writerow([node_names[src_idx], node_names[dst_idx]])
                
    print(f"      Saved {output_node_csv}")

def process_circuit(file_path, output_node_csv, output_edge_csv):
    try:
        builder = NetlistGraphBuilder([file_path])
        graph = builder.build()
        
        # We simulate 1000 vectors for fast processing while maintaining statistical accuracy
        sim = LogicSimulator(graph, num_vectors=1000) 
        sim.generate_vectors()
        sim.simulate()
        pyg_data = sim.embed_tp_and_export()
        
        tp_dict = {}
        for node in graph.nodes():
            tp_dict[node] = graph.nodes[node].get('TP', 0.0)
            
        compute_and_save_dataset(pyg_data, tp_dict, list(graph.nodes()), output_node_csv, output_edge_csv)
    except Exception as e:
        print(f"    [!] Failed to process {file_path}: {e}")

def main():
    base_trit_tc = "/Users/sudiptomaity/projects/TrojanNet-GNN/TARMAC_trigger_activation/benchmarks/TRIT-TC"
    out_dir = "/Users/sudiptomaity/projects/TrojanNet-GNN/datasets_for_gnn"
    
    # 9 circuits specified
    sizes = {
        "Small": ['c2670', 'c3540', 's1423'],
        "Medium": ['c5315', 'c6288', 's13207'],
        "Large": ['s15850', 's35932'] # AES will be copied manually
    }
    
    # Create Folders
    for size in sizes.keys():
        os.makedirs(os.path.join(out_dir, size, "train", "nodes"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, size, "train", "edges"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, size, "test", "nodes"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, size, "test", "edges"), exist_ok=True)
        
    print("==================================================")
    print("STARTING GNN DATASET GENERATION")
    print("==================================================")
    
    for size, circuits in sizes.items():
        print(f"\n---> Processing {size} Circuits: {circuits}")
        for circ in circuits:
            print(f"  -> Circuit: {circ}")
            
            # Find all variations (e.g. c2670_T000, c2670_T001...)
            variations = sorted(glob.glob(os.path.join(base_trit_tc, f"{circ}_T*")))
            
            if not variations:
                print(f"    [!] Could not find any variations for {circ}")
                continue
                
            # Sample 5 for Train, 2 for Test
            train_vars = variations[:5]
            test_vars = variations[5:7] if len(variations) >= 7 else variations[-1:]
            
            # Process Train
            for var_path in train_vars:
                basename = os.path.basename(var_path)
                v_file = os.path.join(var_path, f"{basename}.v")
                if os.path.exists(v_file):
                    node_csv = os.path.join(out_dir, size, "train", "nodes", f"dataset_{basename}.csv")
                    edge_csv = os.path.join(out_dir, size, "train", "edges", f"edges_{basename}.csv")
                    process_circuit(v_file, node_csv, edge_csv)
                    
            # Process Test
            for var_path in test_vars:
                basename = os.path.basename(var_path)
                v_file = os.path.join(var_path, f"{basename}.v")
                if os.path.exists(v_file):
                    node_csv = os.path.join(out_dir, size, "test", "nodes", f"dataset_{basename}.csv")
                    edge_csv = os.path.join(out_dir, size, "test", "edges", f"edges_{basename}.csv")
                    process_circuit(v_file, node_csv, edge_csv)

    print("\n==================================================")
    print("DATASET GENERATION COMPLETE")
    print("==================================================")

if __name__ == "__main__":
    main()
