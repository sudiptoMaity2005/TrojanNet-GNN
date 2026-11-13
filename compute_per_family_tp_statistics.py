import os
import glob
import csv
import math
import pandas as pd
import numpy as np
from collections import defaultdict

INVERTED_RS232_VARIANTS = {
    'RS232_T1000_180nm', 'RS232_T1000_90nm',
    'RS232_T1100_180nm', 'RS232_T1100_90nm',
    'RS232_T1200_180nm', 'RS232_T1200_90nm',
    'RS232_T1300_180nm', 'RS232_T1300_90nm',
    'RS232_T1400_180nm', 'RS232_T1400_90nm',
    'RS232_T1500_180nm', 'RS232_T1500_90nm',
    'RS232_T1600_180nm', 'RS232_T1600_90nm',
    'RS232_T1700_180nm_scan', 'RS232_T1700_90nm_scan',
    'RS232_T1800_180nm_scan', 'RS232_T1800_90nm_scan',
    'RS232_T1900_180nm_scan', 'RS232_T1900_90nm_scan',
    'RS232_T2000_180nm_scan', 'RS232_T2000_90nm_scan'
}

def determine_family(circuit_name):
    lower_name = circuit_name.lower()
    if 'b19' in lower_name: return 'b19'
    if 'ethernet' in lower_name: return 'EthernetMAC10GE'
    if 'wb_conmax' in lower_name or 'wb-conmax' in lower_name: return 'wb_conmax'
    if 'rs232' in lower_name: return 'RS232'
    if 'aes' in lower_name: return 'AES'
    if circuit_name.startswith('c') or circuit_name.startswith('s'):
        return circuit_name.split('_')[0]
    return 'unknown'

def is_true_trojan_iscas(node_name):
    n = str(node_name).lower()
    if n.startswith('test_') or 'crc_out' in n:
        return 0
    return 1 if any(n.startswith(k) for k in ['troj', 'trigger', 'tempn', 'tempg']) else 0

def load_rs232_verdicts():
    verdicts = {}
    verdict_path = "results/rs232_trojan_nodes_with_verdict.csv"
    if os.path.exists(verdict_path):
        df_v = pd.read_csv(verdict_path)
        for _, row in df_v.iterrows():
            verdicts[(str(row['Circuit']), str(row['Node']))] = int(row['Verified_Trojan'])
    return verdicts

def main():
    print("=== Computing Per-Family Transition Probability (TP) Statistics (Task E6 & Item 6) ===")
    rs232_verdicts = load_rs232_verdicts()
    
    folders = [
        "datasets_for_gnn", "b19_datasets", "ethernet_datasets", "wb_conmax_datasets",
        "aes_output_datasets", "rs232_output_datasets"
    ]
    
    all_circuit_files = {}
    for fldr in folders:
        for root, dirs, files in os.walk(fldr):
            for f in files:
                if f.endswith('.csv') and not f.startswith('edges_') and 'results' not in f.lower() and 'summary' not in f.lower():
                    c_name = f.replace("dataset_", "").replace(".csv", "")
                    if c_name in INVERTED_RS232_VARIANTS:
                        continue
                    if c_name not in all_circuit_files:
                        all_circuit_files[c_name] = os.path.join(root, f)

    print(f"Aggregating statistics across {len(all_circuit_files)} benchmark circuits...")
    
    family_data = defaultdict(lambda: {
        'circuits': set(),
        'all_tp': [],
        'benign_tp': [],
        'trojan_tp': []
    })

    for c_name, csv_path in sorted(all_circuit_files.items()):
        fam = determine_family(c_name)
        df = pd.read_csv(csv_path)
        node_names = df['Node'].astype(str).tolist() if 'Node' in df.columns else [str(i) for i in range(len(df))]
        
        # Ground truth
        if c_name.startswith(('c', 's')) and node_names:
            y_true = [is_true_trojan_iscas(n) for n in node_names]
        elif 'RS232' in c_name:
            y_true = [rs232_verdicts.get((c_name, n), int(df.loc[df['Node'] == n, 'Label'].values[0] if 'Label' in df.columns else 0)) for n in node_names]
        elif 'Label' in df.columns:
            y_true = df['Label'].values.tolist()
        else:
            y_true = [0] * len(df)
            
        # For B19, Ethernet, wb_conmax, the columns are f0-f22 where f13 is the true simulated TP
        if fam in ['b19', 'EthernetMAC10GE', 'wb_conmax'] and 'f13' in df.columns:
            f1_col = 'f13'
        else:
            feat_cols = [c for c in df.columns if c not in ['Node', 'Label']]
            if len(feat_cols) > 8:
                f_last8 = feat_cols[-8:]
            else:
                f_last8 = feat_cols
            f1_col = f_last8[0]
            
        tp_vals = df[f1_col].values.astype(float)
        
        family_data[fam]['circuits'].add(c_name)
        family_data[fam]['all_tp'].extend(tp_vals)
        for yt, tp in zip(y_true, tp_vals):
            if yt == 1:
                family_data[fam]['trojan_tp'].append(tp)
            else:
                family_data[fam]['benign_tp'].append(tp)
                
    table_rows = []
    target_order = ['c2670', 'c3540', 'c5315', 'c6288', 's13207', 's1423', 's15850', 's35932', 'RS232', 'AES', 'b19', 'EthernetMAC10GE', 'wb_conmax']
    
    for fam in target_order:
        if fam not in family_data: continue
        d = family_data[fam]
        all_tp = np.array(d['all_tp'])
        benign_tp = np.array(d['benign_tp'])
        trojan_tp = np.array(d['trojan_tp'])
        
        n_circs = len(d['circuits'])
        n_total = len(all_tp)
        n_trojan = len(trojan_tp)
        
        mean_tp = float(np.mean(all_tp))
        std_tp = float(np.std(all_tp))
        share_tp_zero = float(np.mean(all_tp == 0.0) * 100.0)
        
        benign_share_lt_0_1 = float(np.mean(benign_tp < 0.1) * 100.0) if len(benign_tp) > 0 else 0.0
        trojan_share_lt_0_1 = float(np.mean(trojan_tp < 0.1) * 100.0) if len(trojan_tp) > 0 else 0.0
        
        median_benign = float(np.median(benign_tp)) if len(benign_tp) > 0 else 0.0
        median_trojan = float(np.median(trojan_tp)) if len(trojan_tp) > 0 else 0.0
        
        table_rows.append({
            "Family": fam,
            "Circuits": n_circs,
            "Total_Nodes": n_total,
            "Trojan_Nodes": n_trojan,
            "Mean_TP": round(mean_tp, 4),
            "Std_TP": round(std_tp, 4),
            "Share_TP_Zero_Pct": round(share_tp_zero, 2),
            "Benign_TP_lt_0.1_Pct": round(benign_share_lt_0_1, 2),
            "Trojan_TP_lt_0.1_Pct": round(trojan_share_lt_0_1, 2),
            "Median_Benign_TP": round(median_benign, 4),
            "Median_Trojan_TP": round(median_trojan, 4)
        })
        
    df_out = pd.DataFrame(table_rows)
    csv_out = "results/per_family_tp_statistics.csv"
    df_out.to_csv(csv_out, index=False)
    print(f"Saved to {csv_out}")
    print("\n" + df_out.to_string(index=False))

if __name__ == "__main__":
    main()

