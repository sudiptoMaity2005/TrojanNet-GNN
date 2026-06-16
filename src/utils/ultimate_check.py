import os
import pandas as pd
import numpy as np

def ultimate_check():
    folder = "/Users/sudiptomaity/projects/TrojanNet-GNN/aes_output_datasets"
    csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    
    total_errors = 0
    total_files = len(csv_files)
    
    print("==================================================")
    print(f"STARTING ULTIMATE ANOMALY CHECK ACROSS {total_files} DATASETS")
    print("==================================================")
    
    for file in sorted(csv_files):
        path = os.path.join(folder, file)
        df = pd.read_csv(path)
        errors = 0
        
        # 1. Null / NaN Check
        if df.isnull().values.any():
            print(f"[FAIL] {file}: Contains NaNs or Nulls")
            errors += 1
            
        # 2. Duplicate Node Check
        if df['Node'].duplicated().any():
            print(f"[FAIL] {file}: Contains duplicate node names")
            errors += 1
            
        # 3. Label Value Check (must be 0 or 1)
        if not df['Label'].isin([0, 1]).all():
            print(f"[FAIL] {file}: Label contains values other than 0 and 1")
            errors += 1
            
        # 4. Feature Bounds Check [0, 1]
        for col in ['f1_TP', 'f2_TPDiff', 'f3_Rare', 'f4_NbMean']:
            if (df[col] < 0).any() or (df[col] > 1).any():
                print(f"[FAIL] {file}: {col} out of bounds [0, 1]")
                errors += 1
                
        # 5. Math Consistency Check (f3 = 1 - f1)
        # Using a tiny tolerance due to floating point precision
        f3_diff = np.abs(df['f3_Rare'] - (1.0 - df['f1_TP']))
        if (f3_diff > 1e-5).any():
            print(f"[FAIL] {file}: Math violation f3_Rare != 1 - f1_TP")
            errors += 1
            
        # 6. Graph Connectivity Check
        # Nodes shouldn't have exactly 0 incoming AND 0 outgoing edges unless they are completely disconnected
        isolated = (df['f7_FanIn'] == 0) & (df['f8_FanOut'] == 0)
        if isolated.sum() > 200: # Allow a few isolated nodes (e.g. constant ties), but not hundreds
            print(f"[FAIL] {file}: Abnormal number of isolated nodes ({isolated.sum()})")
            errors += 1
            
        total_errors += errors

    print("==================================================")
    if total_errors == 0:
        print("ALL DATASETS PASSED EVERY MATHEMATICAL AND STRUCTURAL CHECK.")
        print("STATUS: PRISTINE (0 ERRORS)")
    else:
        print(f"DETECTED {total_errors} TOTAL ERRORS.")
    print("==================================================")

if __name__ == "__main__":
    ultimate_check()
