import random

filepath = '/Users/sudiptomaity/projects/TrojanNet-GNN/aes/aes_fixed_patterns_10k.txt'

with open(filepath, 'w') as f:
    for _ in range(10000):
        # AES has 128 bit state, 128 bit key, plus clk/rst, so at least 258 bits.
        # Let's generate 300 random bits per vector just to be safe for any extra test pins.
        bits = ''.join(str(random.randint(0, 1)) for _ in range(300))
        f.write(bits + '\n')

print(f"Generated {filepath}")
