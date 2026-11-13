import os
import glob
import csv
import numpy as np
import torch  # pyrefly: ignore [missing-import]
from parser import NetlistGraphBuilder
from simulator import LogicSimulator

def run_simulation_for_folder(folder_path, output_csv):
    folder_name = os.path.basename(os.path.normpath(folder_path))
    print(f"\n=============================================")
    print(f"       ANALYZING: {folder_name.upper()}")
    print(f"=============================================")
    
    # 1. Dynamically find all Verilog files
    all_files = glob.glob(os.path.join(folder_path, "*.v"))
    
    # 2. Filter out testbenches (they are not synthesizeable hardware)
    files = [f for f in all_files if not os.path.basename(f).startswith("test_")]
    
    # 3. Determine the Top Module
    # In BOTH the clean and infected circuits, the highest level module is literally named 'top'.
    top_module = "top"
    print(f"Detected Top Module: {top_module}")
             
    print("--- Phase 1: Hierarchical Parsing ---")
    builder = NetlistGraphBuilder(files, top_module=top_module)
    graph = builder.build()
    print(f"Success! Hierarchical graph flattened into {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    
    print("\n--- Phase 2: Logic Simulation & TP Calculation ---")
    sim = LogicSimulator(graph, num_vectors=1000)

    # --- INPUT FILE GENERATION & LOADING ---
    input_file = "input_vectors_aes.csv"
    if os.path.exists(input_file):
        print(f"Loading existing test vectors from {input_file}...")
        sim.vectors = np.loadtxt(input_file, delimiter=',', dtype=int)
        if sim.vectors.ndim == 1:
            sim.vectors = sim.vectors.reshape(1, -1)
        sim.num_vectors = sim.vectors.shape[1]
    else:
        print(f"Generating 1,000 random test vectors and saving to {input_file}...")
        sim.generate_vectors()
        np.savetxt(input_file, sim.vectors, delimiter=',', fmt='%d')
        
    print(f"Simulating {sim.num_vectors} test vectors...")
    sim.simulate()
    
    # Calculate TP and export
    pyg_data = sim.embed_tp_and_export()
    
    # --- OUTPUT FILE GENERATION ---
    print(f"\n--- Saving TP Results to {output_csv} ---")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Node', 'TP'])
        for node in graph.nodes():
            tp = graph.nodes[node].get('TP', 0.0)
            writer.writerow([node, f"{tp:.4f}"])
            
    print(f"Saved TPs for {graph.number_of_nodes()} nodes.")
    
    # Export PyG Data
    pt_file = f"aes_{folder_name}_data.pt"
    torch.save(pyg_data, pt_file)
    print(f"Graph tensor saved to {pt_file}!")

if __name__ == "__main__":
    # Point to the new AES_circuit folders
    run_simulation_for_folder("AES_circuit/AES T100", "tp_output_aes_T100.csv")
    run_simulation_for_folder("AES_circuit/AES", "tp_output_aes_clean.csv")
    print("\nALL DONE! You can now compare the two AES CSV files.")
