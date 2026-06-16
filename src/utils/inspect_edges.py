import torch
data = torch.load("aes_AES_data.pt", weights_only=False)
node_list = []
import csv
with open("tp_output_aes_clean.csv", "r") as f:
    r = csv.reader(f)
    next(r)
    for row in r:
        node_list.append(row[0])
idx = node_list.index("rf.S4_2.out1")
print(f"Index of rf.S4_2.out1: {idx}")
edges_src = (data.edge_index[0] == idx).nonzero(as_tuple=True)[0]
edges_dst = (data.edge_index[1] == idx).nonzero(as_tuple=True)[0]
print(f"Edges leaving rf.S4_2.out1: {edges_src}")
print(f"Edges entering rf.S4_2.out1: {edges_dst}")
