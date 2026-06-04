import os
import pandas as pd
from tarmac import tarmac_test_generation
from tarmac_optimized import tarmac_test_generation as optimized_test_generation

def run_gps_experiment():
    print("\n======================================")
    print("Starting COMPARISON on Massive GPS Circuit")
    print("WARNING: This circuit contains ~588K nodes.")
    print("We will run both BASE and OPTIMIZED algorithms.")
    print("======================================")
    
    circuit_file = "benchmarks/GPS/gps.v"
    circuit_name = "gps" # Top module name
    
    results = []
    
    # Run Base Algorithm
    try:
        print("\n--- [1/2] RUNNING BASE ALGORITHM ---")
        res_base = tarmac_test_generation(
            circuit_file=circuit_file,
            top_module=circuit_name,
            num_vectors=1,        # Only generate 1 vector since it's huge
            rare_threshold=0.01   # Using a stricter threshold for massive circuits
        )
        if len(res_base) > 0:
            results.append({
                "Algorithm": "Base TARMAC",
                "Circuit": circuit_name,
                "Nodes": res_base["nodes"],
                "Rare Signals": res_base["rare_signals"],
                "Time (Seconds)": round(res_base["time_seconds"], 2)
            })
    except Exception as e:
        print(f"Error running Base GPS: {e}")

    # Run Optimized Algorithm
    try:
        print("\n--- [2/2] RUNNING OPTIMIZED ALGORITHM ---")
        res_opt = optimized_test_generation(
            circuit_file=circuit_file,
            top_module=circuit_name,
            num_vectors=1,        
            rare_threshold=0.01   
        )
        if len(res_opt) > 0:
            results.append({
                "Algorithm": "Optimized TARMAC",
                "Circuit": circuit_name,
                "Nodes": res_opt["nodes"],
                "Rare Signals": res_opt["rare_signals"],
                "Time (Seconds)": round(res_opt["time_seconds"], 2)
            })
    except Exception as e:
        print(f"Error running Optimized GPS: {e}")

    if results:
        # Save to its own CSV
        df = pd.DataFrame(results)
        csv_path = "gps_tarmac_comparison.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n======================================")
        print(f"SUCCESS: Comparison saved to {csv_path}!")
        print("======================================")
        print(df.to_string(index=False))
        
        if len(results) == 2:
            base_time = results[0]["Time (Seconds)"]
            opt_time = results[1]["Time (Seconds)"]
            speedup = base_time / opt_time
            print(f"\n--> The Optimized Algorithm is {speedup:.2f}x faster!")

if __name__ == "__main__":
    run_gps_experiment()
