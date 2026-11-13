import os
import glob
import pandas as pd
import torch
import numpy as np
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from parser import NetlistGraphBuilder
from simulator import LogicSimulator

def simulate_folder(v_files, top_module='wb_conmax_top'):
    print(f"Parsing {len(v_files)} files...")
    builder = NetlistGraphBuilder(v_files, top_module=top_module)
    graph = builder.build()
    
    sim = LogicSimulator(graph, num_vectors=2000) # 2k vectors for speed given its size
    sim.vectors = sim.generate_vectors()
    
    print(f"Simulating {sim.num_vectors} vectors on {graph.number_of_nodes()} nodes...")
    sim.simulate()
    pyg_data = sim.embed_tp_and_export()
    
    tp_dict = {}
    for node in graph.nodes():
        tp_dict[node] = graph.nodes[node].get('TP', 0.0)
        
    return pyg_data, tp_dict, graph, list(graph.nodes())

def compute_and_save_dataset(pyg_data, ht_tp_dict, clean_tp_dict, node_names, output_csv):
    num_nodes = pyg_data.num_nodes
    
    # pyg_data.x currently has 14 features (11 one-hot + in_deg + out_deg + TP)
    # The base datasets have 23 features (15 struct + 8 TARMAC). We pad with zeros.
    x_original = pyg_data.x
    x_padded = torch.zeros((num_nodes, 23), dtype=torch.float)
    x_padded[:, :14] = x_original
    
    # We can inject tp_diff as the 15th feature (index 14)
    tp_diff_tensor = torch.zeros(num_nodes)
    tp_tensor = torch.zeros(num_nodes)
    for idx, node in enumerate(node_names):
        val_ht = ht_tp_dict.get(node, 0.0)
        val_norm = clean_tp_dict.get(node, 0.0)
        tp_tensor[idx] = val_ht
        tp_diff_tensor[idx] = abs(val_ht - val_norm)
        
    x_padded[:, 14] = tp_diff_tensor
    
    pyg_data.x = x_padded
                    
    # Assign Labels
    y_list = []
    for node in node_names:
        if any(sub in node for sub in ["Trojan", "Tj", "tj", "TSC", "lfsr", "AM_Transmission"]):
            y_list.append(1)
        else:
            y_list.append(0)
            
    pyg_data.y = torch.tensor(y_list, dtype=torch.long)
    
    df_dict = {'Node': node_names, 'Label': y_list}
    for i in range(23):
        df_dict[f'f{i}'] = x_padded[:, i].numpy()
        
    df = pd.DataFrame(df_dict)
    
    df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv} | Trojans: {sum(y_list)} / {len(y_list)}")
    return pyg_data

def process_b19():
    base_dir = os.path.expanduser("~/Downloads")
    variants = ["b19-T200"]
    top_module = "b19_scan"
    
    out_dir = "b19_datasets"
    os.makedirs(out_dir, exist_ok=True)
    
    for c in variants:
        print(f"\n--- Processing {c} ---")
        c_dir = os.path.join(base_dir, c, 'src')
        
        tj_free_files = glob.glob(os.path.join(c_dir, 'TjFree', '*.v'))
        tj_in_files = glob.glob(os.path.join(c_dir, 'TjIn', '*.v'))
        
        if not tj_free_files or not tj_in_files:
            print(f"Missing files for {c}")
            continue
            
        orig_dir = os.getcwd()
        
        print("1. Simulating TjFree (Clean)")
        os.chdir(os.path.join(c_dir, 'TjFree'))
        tj_free_files = glob.glob('*.v')
        _, clean_tp_dict, _, _ = simulate_folder(tj_free_files, top_module=top_module)
        
        print("2. Simulating TjIn (Trojan)")
        os.chdir(os.path.join(c_dir, 'TjIn'))
        tj_in_files = glob.glob('*.v')
        pyg_data, ht_tp_dict, _, node_names = simulate_folder(tj_in_files, top_module=top_module)
        
        os.chdir(orig_dir)
        
        csv_path = os.path.join(out_dir, f"{c}.csv")
        pt_path = os.path.join(out_dir, f"{c}.pt")
        
        final_data = compute_and_save_dataset(pyg_data, ht_tp_dict, clean_tp_dict, node_names, csv_path)
        torch.save(final_data, pt_path)
        
if __name__ == "__main__":
    process_b19()
