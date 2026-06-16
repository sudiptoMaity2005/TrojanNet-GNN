import os
import glob

def patch_files():
    base_dir = "/Users/sudiptomaity/projects/TrojanNet-GNN/aes/trojan"
    files = glob.glob(os.path.join(base_dir, "AES T*", "aes_128.v"))
    
    for f in files:
        with open(f, 'r') as file:
            content = file.read()
            
        # Replace empty arguments
        if "k9,   , k9b" in content:
            content = content.replace("k9,   , k9b", "k9, k10, k9b")
        elif "k9,  , k9b" in content:
            content = content.replace("k9,  , k9b", "k9, k10, k9b")
        elif "k9, , k9b" in content:
            content = content.replace("k9, , k9b", "k9, k10, k9b")
            
        with open(f, 'w') as file:
            file.write(content)
            
    print(f"Patched {len(files)} files.")

if __name__ == "__main__":
    patch_files()
