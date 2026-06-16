import os

base_dir = "/Users/sudiptomaity/projects/TrojanNet-GNN/rs232/trojans"
folders = ["T2100", "T2200", "T2300", "T2400"]
files_to_check = ["uart.v", "u_rec.v", "u_xmit.v"]

for folder in folders:
    for filename in files_to_check:
        filepath = os.path.join(base_dir, folder, filename)
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                lines = f.readlines()
            
            with open(filepath, "w") as f:
                for line in lines:
                    if '`include "inc.h"' in line:
                        f.write('// ' + line)
                    else:
                        f.write(line)
            print(f"Patched {filepath}")
