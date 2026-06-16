import os

base = "/Users/sudiptomaity/projects/TrojanNet-GNN/rs232/trojans/T901/"
files = ["uart.v", "u_rec.v", "u_xmit.v"]

for file in files:
    filepath = os.path.join(base, file)
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            lines = f.readlines()
        
        with open(filepath, "w") as f:
            for line in lines:
                if '`include' in line:
                    f.write('// ' + line)
                else:
                    f.write(line)
        print(f"Patched {filepath}")
