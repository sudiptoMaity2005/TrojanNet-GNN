import pandas as pd
import networkx as nx
import numpy as np
import os
import glob
import time

def build_graph(nodes_file, edges_file):
    print(f"Loading Graph from: {nodes_file} ...")
    df_nodes = pd.read_csv(nodes_file)
    df_edges = pd.read_csv(edges_file)
    
    # Extract TP and Labels
    node_tp = {row['Node']: row['f1_TP'] for _, row in df_nodes.iterrows()}
    node_labels = {row['Node']: row['Label'] for _, row in df_nodes.iterrows()}
    
    G = nx.DiGraph()
    for _, row in df_nodes.iterrows():
        G.add_node(str(row['Node']), tp=row['f1_TP'], label=row['Label'])
        
    edges_to_add = [(str(row['src']), str(row['dst'])) for _, row in df_edges.iterrows()]
    G.add_edges_from(edges_to_add)
    
    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G, df_nodes

def analyze_rare_regions(G, tp_threshold=0.01):
    print(f"\n--- Stage 3: Rare Node Identification (TP < {tp_threshold}) ---")
    rare_nodes = [n for n, d in G.nodes(data=True) if d['tp'] < tp_threshold]
    print(f"Identified {len(rare_nodes)} extremely rare nodes.")
    
    print("\n--- Stage 4: Rare Region Formation ---")
    # Induce subgraph and find weakly connected components
    sub_G = G.subgraph(rare_nodes)
    regions = list(nx.weakly_connected_components(sub_G))
    # Filter out isolated rare nodes (size <= 2) to reduce false positives
    regions = [r for r in regions if len(r) > 2] 
    print(f"Formed {len(regions)} connected behavioral regions (size > 2).")
    
    print("\n--- Stage 5 & 6 & 7: Propagation, Structure, and Scoring ---")
    region_scores = []
    
    for i, r in enumerate(regions):
        r = list(r)
        
        # 1. TP Propagation Anomaly PA(R)
        pa_r_sum = 0
        for v in r:
            neighbors = list(G.predecessors(v)) + list(G.successors(v))
            if not neighbors:
                continue
            v_tp = G.nodes[v]['tp']
            pa_v = sum(abs(v_tp - G.nodes[u]['tp']) for u in neighbors) / len(neighbors)
            pa_r_sum += pa_v
        
        pa_r = pa_r_sum / len(r) if len(r) > 0 else 0
        
        # 2. Rarity Metric Rare(R)
        avg_tp = sum(G.nodes[v]['tp'] for v in r) / len(r)
        rare_r = 1.0 - avg_tp # Higher is rarer
        
        # 3. Density (internal edges / possible edges)
        sub_r = G.subgraph(r)
        density = nx.density(sub_r)
        
        # 4. Final Suspicion Score
        # PA(R) is the most critical feature because Trojans have abnormal jumps to normal logic
        score = (0.3 * rare_r) + (0.5 * pa_r) + (0.2 * density)
        
        # Check if true Trojan
        true_trojans = sum(1 for v in r if G.nodes[v]['label'] == 1)
        
        region_scores.append({
            'Region_ID': i,
            'Size': len(r),
            'Avg_TP': avg_tp,
            'Propagation_Anomaly': pa_r,
            'Density': density,
            'Score': score,
            'True_Trojans': true_trojans,
            'Nodes': r
        })
        
    # Sort by score descending
    region_scores.sort(key=lambda x: x['Score'], reverse=True)
    return region_scores

if __name__ == "__main__":
    # Let's run this on an existing Trust-Hub AES dataset (e.g. T100)
    base_dir = "aes_output_datasets"
    edge_dir = "aes_edges_datasets"
    
    node_file = os.path.join(base_dir, "dataset_AES_T100.csv")
    edge_file = os.path.join(edge_dir, "edges_AES_T100.csv")
    
    if not os.path.exists(node_file):
        print(f"Error: {node_file} not found. Please ensure datasets are unzipped.")
        exit(1)
        
    start_time = time.time()
    G, _ = build_graph(node_file, edge_file)
    
    regions = analyze_rare_regions(G, tp_threshold=0.01)
    
    print("\n========================================================")
    print("TOP 5 MOST SUSPICIOUS REGIONS (LOCALIZATION PREDICTIONS)")
    print("========================================================")
    for i, r in enumerate(regions[:5]):
        print(f"Rank {i+1} - Score: {r['Score']:.4f} | Size: {r['Size']} | PA: {r['Propagation_Anomaly']:.4f} | Avg TP: {r['Avg_TP']:.4f}")
        print(f"  -> Contains True Trojans: {r['True_Trojans']} / {r['Size']}")
        print(f"  -> Predicted Trojan Gates: {', '.join(r['Nodes'][:5])}{'...' if len(r['Nodes']) > 5 else ''}")
        print("-")
        
    end_time = time.time()
    print(f"\nLocalization completed in {end_time - start_time:.2f} seconds.")
