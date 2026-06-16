import os
import shutil
import random
import glob

def setup_dirs(base_name):
    dirs = [
        f"{base_name}_split/train/nodes",
        f"{base_name}_split/train/edges",
        f"{base_name}_split/test/nodes",
        f"{base_name}_split/test/edges",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return f"{base_name}_split"

def split_datasets(node_dir, edge_dir, base_name, prefix):
    split_base = setup_dirs(base_name)
    
    node_files = sorted(glob.glob(os.path.join(node_dir, "*.csv")))
    random.shuffle(node_files)
    
    train_size = int(0.8 * len(node_files))
    train_files = node_files[:train_size]
    test_files = node_files[train_size:]
    
    # Process Train
    for nf in train_files:
        basename = os.path.basename(nf)
        edge_basename = basename.replace(f"dataset_{prefix}", f"edges_{prefix}")
        ef = os.path.join(edge_dir, edge_basename)
        
        if os.path.exists(ef):
            shutil.copy(nf, os.path.join(split_base, "train/nodes", basename))
            shutil.copy(ef, os.path.join(split_base, "train/edges", edge_basename))
            
    # Process Test
    for nf in test_files:
        basename = os.path.basename(nf)
        edge_basename = basename.replace(f"dataset_{prefix}", f"edges_{prefix}")
        ef = os.path.join(edge_dir, edge_basename)
        
        if os.path.exists(ef):
            shutil.copy(nf, os.path.join(split_base, "test/nodes", basename))
            shutil.copy(ef, os.path.join(split_base, "test/edges", edge_basename))

    print(f"{base_name.upper()} Split Complete: {len(train_files)} Train, {len(test_files)} Test")

if __name__ == "__main__":
    # Ensure reproducibility
    random.seed(42)
    
    # RS232 Split
    split_datasets(
        node_dir="rs232_output_datasets",
        edge_dir="rs232_edges_datasets",
        base_name="rs232",
        prefix="RS232"
    )
    
    # AES Split
    split_datasets(
        node_dir="aes_output_datasets",
        edge_dir="aes_edges_datasets",
        base_name="aes",
        prefix="AES"
    )
