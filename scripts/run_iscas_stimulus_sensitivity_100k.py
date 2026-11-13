import os
import sys
import time
import csv
import numpy as np

sys.path.insert(0, '/Users/sudiptomaity/projects/TrojanNet-GNN/venv/lib/python3.13/site-packages')
sys.path.insert(0, '/Users/sudiptomaity/projects/TrojanNet-GNN')

from parser import NetlistGraphBuilder
from simulator import LogicSimulator

def main():
    print("=== Task E5: Stimulus Sensitivity (1k, 10k, 100k Vectors) on 8 ISCAS T000 Circuits ===")
    
    iscas_circuits = [
        ('c2670_T000', 'TARMAC_trigger_activation/benchmarks/TRIT-TC/c2670_T000/c2670_T000.v', 'c2670_T000'),
        ('c3540_T000', 'TARMAC_trigger_activation/benchmarks/TRIT-TC/c3540_T000/c3540_T000.v', 'c3540_T000'),
        ('c5315_T000', 'TARMAC_trigger_activation/benchmarks/TRIT-TC/c5315_T000/c5315_T000.v', 'c5315_T000'),
        ('c6288_T000', 'TARMAC_trigger_activation/benchmarks/TRIT-TC/c6288_T000/c6288_T000.v', 'c6288_T000'),
        ('s1423_T000', 'TARMAC_trigger_activation/benchmarks/TRIT-TC/s1423_T000/s1423_T000.v', 's1423_T000'),
        ('s13207_T000', 'TARMAC_trigger_activation/benchmarks/TRIT-TC/s13207_T000/s13207_T000.v', 's13207_T000'),
        ('s15850_T000', 'TARMAC_trigger_activation/benchmarks/TRIT-TC/s15850_T000/s15850_T000.v', 's15850_T000'),
        ('s35932_T000', 'TARMAC_trigger_activation/benchmarks/TRIT-TC/s35932_T000/s35932_T000.v', 's35932_T000')
    ]
    
    results = []
    
    for name, vfile, top in iscas_circuits:
        if not os.path.exists(vfile):
            print(f"File not found: {vfile}, skipping.")
            continue
            
        print(f"\nProcessing {name}...")
        t_parse_start = time.time()
        builder = NetlistGraphBuilder([vfile], top_module=top)
        graph = builder.build()
        t_parse = time.time() - t_parse_start
        num_nodes = graph.number_of_nodes()
        pis = [n for n, d in graph.nodes(data=True) if d.get('type') == 'INPUT']
        num_pis = len(pis)
        
        print(f"  Parsed {num_nodes} nodes, {num_pis} PIs in {t_parse:.2f}s")
        if num_pis == 0 or num_nodes == 0:
            continue
            
        # Generate 100k vectors once
        np.random.seed(42)
        vectors_100k = np.random.randint(0, 2, size=(num_pis, 100000))
        vectors_10k = vectors_100k[:, :10000]
        vectors_1k = vectors_100k[:, :1000]
        
        # 1k
        sim_1k = LogicSimulator(graph, num_vectors=1000)
        sim_1k.vectors = vectors_1k
        t0 = time.time()
        sim_1k.simulate()
        t_1k = time.time() - t0
        tp_1k = np.array([sim_1k.transitions[n] / 999.0 for n in graph.nodes()])
        rare_nodes_1k = set([n for n in graph.nodes() if (sim_1k.transitions[n] / 999.0) < 0.1])
        
        # 10k
        sim_10k = LogicSimulator(graph, num_vectors=10000)
        sim_10k.vectors = vectors_10k
        t0 = time.time()
        sim_10k.simulate()
        t_10k = time.time() - t0
        tp_10k = np.array([sim_10k.transitions[n] / 9999.0 for n in graph.nodes()])
        rare_nodes_10k = set([n for n in graph.nodes() if (sim_10k.transitions[n] / 9999.0) < 0.1])
        
        # 100k
        sim_100k = LogicSimulator(graph, num_vectors=100000)
        sim_100k.vectors = vectors_100k
        t0 = time.time()
        sim_100k.simulate()
        t_100k = time.time() - t0
        tp_100k = np.array([sim_100k.transitions[n] / 99999.0 for n in graph.nodes()])
        rare_nodes_100k = set([n for n in graph.nodes() if (sim_100k.transitions[n] / 99999.0) < 0.1])
        
        # Correlation
        corr_1k_10k = np.corrcoef(tp_1k, tp_10k)[0, 1] if np.std(tp_1k) > 1e-8 and np.std(tp_10k) > 1e-8 else 1.0
        corr_10k_100k = np.corrcoef(tp_10k, tp_100k)[0, 1] if np.std(tp_10k) > 1e-8 and np.std(tp_100k) > 1e-8 else 1.0
        
        mae_1k_10k = np.mean(np.abs(tp_1k - tp_10k))
        mae_10k_100k = np.mean(np.abs(tp_10k - tp_100k))
        
        overlap_1k_100k = len(rare_nodes_1k.intersection(rare_nodes_100k)) / max(1, len(rare_nodes_1k.union(rare_nodes_100k))) * 100.0 if len(rare_nodes_100k) > 0 else 100.0
        
        row = {
            'Circuit': name,
            'Gate_Count': num_nodes,
            'PI_Count': num_pis,
            'Time_1k_sec': round(t_1k, 2),
            'Time_10k_sec': round(t_10k, 2),
            'Time_100k_sec': round(t_100k, 2),
            'Corr_1k_10k': round(float(corr_1k_10k), 4),
            'Corr_10k_100k': round(float(corr_10k_100k), 4),
            'MAE_1k_10k': round(float(mae_1k_10k), 4),
            'MAE_10k_100k': round(float(mae_10k_100k), 4),
            'Rare_1k': len(rare_nodes_1k),
            'Rare_10k': len(rare_nodes_10k),
            'Rare_100k': len(rare_nodes_100k),
            'Rare_Set_Jaccard_Overlap_Pct': round(overlap_1k_100k, 2)
        }
        results.append(row)
        print(f"  Done: 1k={t_1k:.2f}s, 10k={t_10k:.2f}s, 100k={t_100k:.2f}s | Corr(10k,100k)={corr_10k_100k:.4f} | Rare: {len(rare_nodes_1k)} -> {len(rare_nodes_10k)} -> {len(rare_nodes_100k)} (Overlap: {overlap_1k_100k:.1f}%)")
        
    os.makedirs('results', exist_ok=True)
    out_csv = 'results/iscas_stimulus_sensitivity_1k_10k_100k.csv'
    with open(out_csv, 'w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\n[DONE] Saved Task E5 results to {out_csv}")

if __name__ == '__main__':
    main()

