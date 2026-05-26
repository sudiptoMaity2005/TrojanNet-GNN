import os
import csv
import numpy as np
import torch  # pyrefly: ignore [missing-import]
from parser import NetlistGraphBuilder
from simulator import LogicSimulator

def run_simulation_for_folder(folder_name, output_csv):
    print(f"\n=============================================")
    print(f"       ANALYZING: {folder_name.upper()}")
    print(f"=============================================")
    
    files = [os.path.join(folder_name, "uart.v"), 
             os.path.join(folder_name, "u_xmit.v"), 
             os.path.join(folder_name, "u_rec.v")]
             
    print("--- Phase 1: Hierarchical Parsing ---")
    builder = NetlistGraphBuilder(files, top_module="uart")
    graph = builder.build()
    print(f"Success! Hierarchical graph flattened into {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    
    print("\n--- Phase 2: Logic Simulation & TP Calculation ---")
    sim = LogicSimulator(graph, num_vectors=1000)

    # --- INPUT FILE GENERATION & LOADING ---
    input_file = "input_vectors.csv"
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
    pt_file = f"rs232_{folder_name}_data.pt"
    torch.save(pyg_data, pt_file)
    print(f"Graph tensor saved to {pt_file}!")

if __name__ == "__main__":
    run_simulation_for_folder("RS232 T400", "tp_output_T400.csv")
    run_simulation_for_folder("RS232", "tp_output_clean.csv")
    print("\nALL DONE! You can now compare the two CSV files.")