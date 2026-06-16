import os
import pandas as pd
from parser import NetlistGraphBuilder

def extract_edges_aes():
    base_dir = "/Users/sudiptomaity/projects/TrojanNet-GNN/aes"
    output_dir = "/Users/sudiptomaity/projects/TrojanNet-GNN/aes_edges_datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Process Normal
    print("Extracting edges for Normal...")
    normal_dir = os.path.join(base_dir, "normal")
    files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith('.v')]
    
    builder = NetlistGraphBuilder(files, top_module="top")
    graph = builder.build()
    if builder.top_module not in ['top', 'aes_128']:
        if 'top' in builder.modules:
            builder.top_module = 'top'
        elif 'aes_128' in builder.modules:
            builder.top_module = 'aes_128'
        builder.graph.clear()
        builder._flatten_module(builder.top_module, instance_prefix="")
        builder.extract_features()
        graph = builder.graph
        
    edges = []
    for src, dst in graph.edges:
        edges.append({'src': src, 'dst': dst})
    df_edges = pd.DataFrame(edges)
    df_edges.to_csv(os.path.join(output_dir, "edges_AES_normal.csv"), index=False)
    
    # 2. Process Trojans
    trojans_dir = os.path.join(base_dir, "trojan")
    folders = [f for f in os.listdir(trojans_dir) if os.path.isdir(os.path.join(trojans_dir, f)) and f.startswith("AES T")]
    
    for trojan_folder in sorted(folders):
        print(f"Extracting edges for {trojan_folder}...")
        tf_path = os.path.join(trojans_dir, trojan_folder)
        files = [os.path.join(tf_path, f) for f in os.listdir(tf_path) if f.endswith('.v')]
        
        if not files:
            continue
            
        builder = NetlistGraphBuilder(files, top_module="top")
        graph = builder.build()
        if builder.top_module not in ['top', 'aes_128']:
            if 'top' in builder.modules:
                builder.top_module = 'top'
            elif 'aes_128' in builder.modules:
                builder.top_module = 'aes_128'
            builder.graph.clear()
            builder._flatten_module(builder.top_module, instance_prefix="")
            builder.extract_features()
            graph = builder.graph
            
        edges = []
        for src, dst in graph.edges:
            edges.append({'src': src, 'dst': dst})
            
        out_csv = os.path.join(output_dir, f"edges_{trojan_folder.replace(' ', '_')}.csv")
        pd.DataFrame(edges).to_csv(out_csv, index=False)
            
    print(f"All AES edges successfully extracted to {output_dir}")

if __name__ == "__main__":
    extract_edges_aes()
