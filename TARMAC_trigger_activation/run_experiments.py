import os
import glob
import pandas as pd
import signal
from tarmac_optimized import tarmac_test_generation

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Timeout")

def run_experiments():
    results = []
    finished_circuits = set()
    
    csv_file = "tarmac_results.csv"
    if os.path.exists(csv_file):
        df_old = pd.read_csv(csv_file)
        results = df_old.to_dict('records')
        finished_circuits = set(df_old['Circuit'].tolist())
        print(f"Loaded {len(finished_circuits)} previously completed circuits from CSV.")

    base_dir = "benchmarks/TRIT-TC"
    benchmark_dirs = []
    if os.path.exists(base_dir):
        for d in os.listdir(base_dir):
            full_path = os.path.join(base_dir, d)
            if os.path.isdir(full_path):
                benchmark_dirs.append(full_path)
    
    benchmark_dirs.sort() 
    
    # Set the signal handler for the alarm
    signal.signal(signal.SIGALRM, timeout_handler)
    
    for b_dir in benchmark_dirs:
        v_files = glob.glob(os.path.join(b_dir, "*.v"))
        if not v_files:
            continue
            
        circuit_file = v_files[0]
        circuit_name = os.path.basename(circuit_file).replace(".v", "")
        
        if circuit_name in finished_circuits:
            continue

        print(f"\n======================================")
        print(f"Starting OPTIMIZED experiment on: {circuit_name}")
        print(f"======================================")
        
        try:
            res = {}
            for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
                print(f"   --> Trying with Rare Threshold: {threshold}")
                
                # Set a strict 60-second timeout for the computation
                signal.alarm(60)
                try:
                    res = tarmac_test_generation(
                        circuit_file=circuit_file,
                        top_module=circuit_name,
                        num_vectors=3,        
                        rare_threshold=threshold    
                    )
                    # Disable alarm if successful
                    signal.alarm(0)
                except TimeoutException:
                    print(f"TIMEOUT! Hit 60-second limit for {circuit_name}")
                    signal.alarm(0)
                    res = "TIMEOUT"
                    break
                except Exception as e:
                    signal.alarm(0)
                    raise e
                    
                if isinstance(res, dict) and len(res) > 0:
                    break
            
            if res == "TIMEOUT":
                results.append({
                    "Circuit": circuit_name,
                    "Nodes": "N/A",
                    "Edges": "N/A",
                    "Rare Signals": 0,
                    "Vectors Generated": 0,
                    "Time (Seconds)": "TIMEOUT"
                })
            elif len(res) == 0:
                print(f"Failed to generate vectors for {circuit_name}")
                results.append({
                    "Circuit": circuit_name,
                    "Nodes": "N/A",
                    "Edges": "N/A",
                    "Rare Signals": 0,
                    "Vectors Generated": 0,
                    "Time (Seconds)": "Failed"
                })
            else:
                results.append({
                    "Circuit": circuit_name,
                    "Nodes": res["nodes"],
                    "Edges": res["edges"],
                    "Rare Signals": res["rare_signals"],
                    "Vectors Generated": res["vectors_generated"],
                    "Time (Seconds)": round(res["time_seconds"], 2)
                })
            
            df = pd.DataFrame(results)
            df.to_csv(csv_file, index=False)
            
        except Exception as e:
            print(f"Error running {circuit_name}: {e}")

    if results:
        print(f"\nSuccessfully updated results in {csv_file}!")
    else:
        print("\nNo results generated.")

if __name__ == "__main__":
    run_experiments()
