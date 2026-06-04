import os
import glob
import pandas as pd
from tarmac import tarmac_test_generation

def run_experiments():
    results = []
    
    # Automatically grab ALL 581 TRIT-TC benchmark circuits!
    base_dir = "benchmarks/TRIT-TC"
    benchmark_dirs = []
    if os.path.exists(base_dir):
        for d in os.listdir(base_dir):
            full_path = os.path.join(base_dir, d)
            if os.path.isdir(full_path):
                benchmark_dirs.append(full_path)
    
    benchmark_dirs.sort() # Ensure consistent ordering
    
    for b_dir in benchmark_dirs:
        # Find the .v file in the directory
        v_files = glob.glob(os.path.join(b_dir, "*.v"))
        if not v_files:
            print(f"Skipping {b_dir}, no .v file found.")
            continue
            
        circuit_file = v_files[0]
        circuit_name = os.path.basename(circuit_file).replace(".v", "")
        
        print(f"\n======================================")
        print(f"Starting experiment on: {circuit_name}")
        print(f"======================================")
        
        try:
            res = {}
            # Try progressively more relaxed thresholds if no rare signals are found at 10%
            for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
                print(f"   --> Trying with Rare Threshold: {threshold}")
                res = tarmac_test_generation(
                    circuit_file=circuit_file,
                    top_module=circuit_name,
                    num_vectors=3,        
                    rare_threshold=threshold    
                )
                if len(res) > 0:
                    break
            
            if len(res) == 0:
                print(f"Failed to generate vectors for {circuit_name}")
                # We add a failed row so it still shows up in your table
                results.append({
                    "Circuit": circuit_name,
                    "Nodes": "N/A",
                    "Edges": "N/A",
                    "Rare Signals": 0,
                    "Vectors Generated": 0,
                    "Time (Seconds)": "Failed"
                })
                continue
                
            results.append({
                "Circuit": circuit_name,
                "Nodes": res["nodes"],
                "Edges": res["edges"],
                "Rare Signals": res["rare_signals"],
                "Vectors Generated": res["vectors_generated"],
                "Time (Seconds)": round(res["time_seconds"], 2)
            })
        except Exception as e:
            print(f"Error running {circuit_name}: {e}")
            
    # Save to CSV
    if results:
        df = pd.DataFrame(results)
        csv_path = "tarmac_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSuccessfully saved results to {csv_path}!")
        print(df.to_string())
    else:
        print("\nNo results generated.")

if __name__ == "__main__":
    run_experiments()
