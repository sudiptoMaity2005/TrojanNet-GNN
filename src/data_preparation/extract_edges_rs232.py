import os
import glob
import pandas as pd
from parser import NetlistGraphBuilder

def get_rs232_files(folder):
    """Get the correct verilog files to parse for a given RS232 folder."""
    files = glob.glob(os.path.join(folder, "*.v"))
    if not files:
        return []
        
    # Standard 3 files
    if any("u_rec.v" in f for f in files) and any("u_xmit.v" in f for f in files) and any("uart.v" in f for f in files):
        if len(files) == 3:
            return files
        # T300-T901 have 4 files (e.g. u_rec, u_xmit, uart, plus trojan file)
        # T100, T200 have 3 files
        # We just parse all .v files in these RTL folders
        return files
        
    # Synthesized files
    if any("90nm" in f for f in files) or any("180nm" in f for f in files):
        return files
        
    return files

def extract_edges():
    base_dir = "/Users/sudiptomaity/projects/TrojanNet-GNN/rs232"
    output_dir = "/Users/sudiptomaity/projects/TrojanNet-GNN/rs232_edges_datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process Normal
    print("Extracting edges for Normal...")
    normal_dir = os.path.join(base_dir, "normal")
    files = [os.path.join(normal_dir, f) for f in ["uart.v", "u_rec.v", "u_xmit.v"]]
    graph = NetlistGraphBuilder(files, top_module="uart").build()
    edges = []
    for src, dst in graph.edges:
        edges.append({'src': src, 'dst': dst})
    df_edges = pd.DataFrame(edges)
    df_edges.to_csv(os.path.join(output_dir, "edges_RS232_normal.csv"), index=False)
    
    # Process Trojans
    trojan_base = os.path.join(base_dir, "trojans")
    folders = [f for f in os.listdir(trojan_base) if os.path.isdir(os.path.join(trojan_base, f))]
    
    for folder_name in sorted(folders):
        folder_path = os.path.join(trojan_base, folder_name)
        v_files = get_rs232_files(folder_path)
        
        if not v_files:
            continue
            
        # Group into 90nm/180nm if needed
        synthesized_90nm = [f for f in v_files if '90nm' in f.lower()]
        synthesized_180nm = [f for f in v_files if '180nm' in f.lower()]
        
        if synthesized_90nm or synthesized_180nm:
            if synthesized_90nm:
                print(f"Extracting edges for {folder_name} (90nm)...")
                graph = NetlistGraphBuilder(synthesized_90nm, top_module="uart").build()
                edges = []
                for src, dst in graph.edges:
                    edges.append({'src': src, 'dst': dst})
                suffix = "90nm_scan" if "scan" in synthesized_90nm[0] else "90nm"
                pd.DataFrame(edges).to_csv(os.path.join(output_dir, f"edges_RS232_{folder_name}_{suffix}.csv"), index=False)
                
            if synthesized_180nm:
                print(f"Extracting edges for {folder_name} (180nm)...")
                graph = NetlistGraphBuilder(synthesized_180nm, top_module="uart").build()
                edges = []
                for src, dst in graph.edges:
                    edges.append({'src': src, 'dst': dst})
                suffix = "180nm_scan" if "scan" in synthesized_180nm[0] else "180nm"
                pd.DataFrame(edges).to_csv(os.path.join(output_dir, f"edges_RS232_{folder_name}_{suffix}.csv"), index=False)
        else:
            print(f"Extracting edges for {folder_name}...")
            graph = NetlistGraphBuilder(v_files, top_module="uart").build()
            edges = []
            for src, dst in graph.edges:
                edges.append({'src': src, 'dst': dst})
            pd.DataFrame(edges).to_csv(os.path.join(output_dir, f"edges_RS232_{folder_name}.csv"), index=False)
            
    print(f"All edges successfully extracted to {output_dir}")

if __name__ == "__main__":
    extract_edges()
