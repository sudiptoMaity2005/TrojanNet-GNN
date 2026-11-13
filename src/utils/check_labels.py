import os
import pandas as pd
from parser import NetlistGraphBuilder

csv_path = 'datasets_for_gnn_extracted/datasets_for_gnn/small/train/nodes/dataset_c2670_T000.csv'
v_path = 'TARMAC_trigger_activation/benchmarks/TRIT-TC/c2670_T000/c2670_T000.v'

if not os.path.exists(csv_path) or not os.path.exists(v_path):
    print("Files not found! Make sure datasets_for_gnn is unzipped.")
    import sys; sys.exit()

df = pd.read_csv(csv_path)
tarmac_nodes = set(df['Node'].values)
print(f"Loaded {len(tarmac_nodes)} nodes from TARMAC CSV.")

builder = NetlistGraphBuilder(v_path)
graph = builder.build()
pyverilog_nodes = set(graph.nodes())
print(f"Parsed {len(pyverilog_nodes)} nodes from PyVerilog.")

intersection = tarmac_nodes.intersection(pyverilog_nodes)
print(f"Matching Node Names: {len(intersection)}")

missing_in_pyverilog = tarmac_nodes - pyverilog_nodes
missing_in_tarmac = pyverilog_nodes - tarmac_nodes

print("\nSample missing in PyVerilog (in TARMAC but not graph):")
print(list(missing_in_pyverilog)[:10])

print("\nSample missing in TARMAC (in graph but not TARMAC):")
print(list(missing_in_tarmac)[:10])
