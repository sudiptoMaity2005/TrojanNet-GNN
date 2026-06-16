import os
import glob

base_dir = '/Users/sudiptomaity/projects/TrojanNet-GNN/aes'

def clean_dir(d):
    for root, dirs, files in os.walk(d):
        for f in files:
            # Keep only .v files, but EXCLUDE testbenches
            if not f.endswith('.v'):
                os.remove(os.path.join(root, f))
                print(f"Removed non-verilog: {os.path.join(root, f)}")
                continue
            
            # Exclude testbenches
            lower_f = f.lower()
            if 'tb' in lower_f or 'test' in lower_f:
                os.remove(os.path.join(root, f))
                print(f"Removed testbench: {os.path.join(root, f)}")

clean_dir(os.path.join(base_dir, 'normal'))
clean_dir(os.path.join(base_dir, 'trojan'))
print("Cleanup complete.")
