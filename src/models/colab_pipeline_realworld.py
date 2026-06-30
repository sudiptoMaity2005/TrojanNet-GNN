# ==============================================================================
# COPY THIS ENTIRE SCRIPT INTO A GOOGLE COLAB NOTEBOOK CELL AND RUN IT
# END-TO-END PIPELINE: SYNTHESIS -> GRAPH EXTRACTION -> ML INFERENCE
# FOR AES_128 & GPS CIRCUITS
# ==============================================================================

# ---------------------------------------------------------
# 1. Install System and Python Dependencies
# ---------------------------------------------------------
import os
import sys

# Install Yosys and Icarus Verilog
print("Installing Yosys, Icarus Verilog, and Python packages...")
os.system("apt-get update -qq && apt-get install -y yosys iverilog > /dev/null")
os.system("pip install -q pyverilog networkx torch_geometric xgboost catboost pandas scikit-learn")

import glob
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, JumpingKnowledge
from torch_geometric.utils import dropout_edge
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Removed external dependencies. All logic is now ultra-fast and standalone in this script.

# ---------------------------------------------------------
# 2. Yosys Synthesis Pipeline
# ---------------------------------------------------------
def synthesize_circuit(files, top_module, output_file):
    json_output = output_file.replace(".v", ".json")
    if os.path.exists(json_output):
        print(f"\n--- {json_output} already exists. Skipping Yosys Synthesis! ---", flush=True)
        return True
        
    print(f"\n--- Synthesizing {top_module} using Yosys ---", flush=True)
    files_str = " ".join(files)
    yosys_cmd = f"yosys -p 'read_verilog -sv {files_str}; prep -top {top_module} -flatten; techmap; opt; clean; write_json {json_output}' > yosys_{top_module}.log 2>&1"
    
    start = time.time()
    ret = os.system(yosys_cmd)
    end = time.time()
    
    if ret != 0 or not os.path.exists(json_output):
        print(f"ERROR: Synthesis failed for {top_module}. Check yosys_{top_module}.log", flush=True)
        return False
        
    print(f"Synthesis successful! Saved to {json_output} (Took {end-start:.2f}s)", flush=True)
    return True

# ---------------------------------------------------------
# 3. Graph Extraction Pipeline (Ultra-Fast JSON & NumPy)
# ---------------------------------------------------------
import json

def extract_graph_features(synth_file, top_module, dataset_dir="realworld_datasets"):
    print(f"--- Extracting Graph Features for {top_module} using Fast JSON Parser ---", flush=True)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # 1. Parse JSON
    json_file = synth_file.replace(".v", ".json")
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    mod = data["modules"][top_module]
    graph = nx.DiGraph()
    bit_to_node = {}
    
    start_parse = time.time()
    for p_name, p_data in mod.get("ports", {}).items():
        for b in p_data["bits"]:
            if isinstance(b, int):
                name = f"PORT_{p_name}_{b}"
                graph.add_node(name, type='INPUT' if p_data['direction']=='input' else 'OUTPUT')
                bit_to_node[b] = name
                
    for c_name, c_data in mod.get("cells", {}).items():
        c_type = c_data["type"].replace("$_", "").replace("_", "").upper()
        g_node = f"GATE_{c_name}"
        graph.add_node(g_node, type=c_type)
        for port, bits in c_data.get("connections", {}).items():
            p_dir = c_data.get("port_directions", {}).get(port, "input")
            for b in bits:
                if isinstance(b, int):
                    w_node = bit_to_node.get(b)
                    if not w_node:
                        w_node = f"WIRE_{b}"
                        graph.add_node(w_node, type='WIRE')
                        bit_to_node[b] = w_node
                    if p_dir == "input":
                        graph.add_edge(w_node, g_node)
                    elif p_dir == "output":
                        graph.add_edge(g_node, w_node)
                        
    end_parse = time.time()
    print(f"Parsed {graph.number_of_nodes()} nodes in {end_parse-start_parse:.2f}s", flush=True)
    
    # 2. Fast NumPy Simulation
    start_sim = time.time()
    num_vectors = 10000
    states = {}
    
    dag = graph.copy()
    sources = []
    for n, d in dag.nodes(data=True):
        if d.get('type') in ['INPUT', 'DFFP', 'DFFN'] or dag.in_degree(n) == 0:
            sources.append(n)
            dag.remove_edges_from(list(dag.in_edges(n)))
            
    for n in sources:
        states[n] = np.random.randint(0, 2, size=num_vectors, dtype=np.int8)
        
    try:
        topo = list(nx.topological_sort(dag))
    except nx.NetworkXUnfeasible:
        topo = list(dag.nodes())
        
    for n in topo:
        if n in states: continue
        preds = list(graph.predecessors(n))
        if not preds:
            states[n] = np.zeros(num_vectors, dtype=np.int8)
            continue
            
        gt = graph.nodes[n].get('type', 'WIRE')
        if gt in ['WIRE', 'OUTPUT']:
            states[n] = states[preds[0]]
        elif gt == 'NOT':
            states[n] = 1 - states[preds[0]]
        elif gt == 'AND':
            res = states[preds[0]]
            for p in preds[1:]: res = res & states[p]
            states[n] = res
        elif gt == 'OR':
            res = states[preds[0]]
            for p in preds[1:]: res = res | states[p]
            states[n] = res
        elif gt == 'XOR':
            res = states[preds[0]]
            for p in preds[1:]: res = res ^ states[p]
            states[n] = res
        elif gt == 'NAND':
            res = states[preds[0]]
            for p in preds[1:]: res = res & states[p]
            states[n] = 1 - res
        elif gt == 'NOR':
            res = states[preds[0]]
            for p in preds[1:]: res = res | states[p]
            states[n] = 1 - res
        elif gt == 'XNOR':
            res = states[preds[0]]
            for p in preds[1:]: res = res ^ states[p]
            states[n] = 1 - res
        elif gt == 'MUX':
            if len(preds) >= 3:
                a, b, s = states[preds[0]], states[preds[1]], states[preds[2]]
                states[n] = np.where(s == 1, b, a)
            else:
                states[n] = states[preds[0]]
        else:
            res = states[preds[0]]
            for p in preds[1:]: res = res ^ states[p]
            states[n] = res
            
    tps = {n: np.sum(states[n][1:] != states[n][:-1]) / max(1, num_vectors - 1) if n in states else 0.0 for n in graph.nodes()}
    end_sim = time.time()
    print(f"NumPy Simulation completed in {end_sim-start_sim:.2f}s", flush=True)
    
    # 3. Calculate 8 Features & Export
    nodes_csv = os.path.join(dataset_dir, f"dataset_{top_module}.csv")
    edges_csv = os.path.join(dataset_dir, f"edges_{top_module}.csv")
    
    # Calculate depth
    depth = {n: 0 for n in topo}
    for n in topo:
        if graph.in_degree(n) > 0:
            depth[n] = max([depth.get(p, 0) for p in graph.predecessors(n)], default=0) + 1
    max_d = max(depth.values()) if depth else 1
    
    y_list = []
    with open(nodes_csv, 'w', newline='') as f:
        f.write("Node,f1_TP,f2_TPDiff,f3_Rare,f4_NbMean,f5_NbDist,f6_LZ,f7_FanIn,f8_FanOut,Label\n")
        for node in graph.nodes():
            lower_node = str(node).lower()
            label = 1 if any(sub in lower_node for sub in ["trojan", "tj", "tss", "malicious"]) else 0
            y_list.append(label)
            
            nb_tps = [tps.get(n, 0) for n in graph.neighbors(node)]
            tp = tps.get(node, 0)
            tp_mean = np.mean(nb_tps) if nb_tps else tp
            tp_std = np.std(nb_tps) if nb_tps else 0.0
            tp_dist = (max(nb_tps) - min(nb_tps)) if nb_tps else 0.0
            rare = 1.0 if tp < 0.1 else 0.0
            lz = depth.get(node, 0) / max(1, max_d)
            fanin = graph.in_degree(node)
            fanout = graph.out_degree(node)
            
            f.write(f"{node},{tp:.6f},{tp_std:.6f},{rare:.6f},{tp_mean:.6f},{tp_dist:.6f},{lz:.6f},{fanin},{fanout},{label}\n")
            
    with open(edges_csv, 'w', newline='') as f:
        f.write("src,dst\n")
        for src, dst in graph.edges():
            f.write(f"{src},{dst}\n")
            
    print(f"Saved dataset CSVs to {dataset_dir}/", flush=True)
    return nodes_csv, edges_csv

# ---------------------------------------------------------
# 4. Model Architectures
# ---------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = (1 - pt) ** self.gamma * BCE_loss
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()

class GNN4GateHybrid(torch.nn.Module):
    def __init__(self, in_channels=8, hidden_channels=64, out_channels=2):
        super(GNN4GateHybrid, self).__init__()
        self.embedding = torch.nn.Linear(in_channels, hidden_channels)
        self.bn_emb = torch.nn.LayerNorm(hidden_channels)
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.bn1 = torch.nn.LayerNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.LayerNorm(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.bn3 = torch.nn.LayerNorm(hidden_channels)
        self.jk = JumpingKnowledge("max")
        self.fc1 = torch.nn.Linear(hidden_channels, 32)
        self.fc2 = torch.nn.Linear(32, out_channels)

    def forward(self, x, edge_index):
        if self.training:
            noise = torch.randn_like(x) * 0.01 
            x = x + noise
            edge_index, _ = dropout_edge(edge_index, p=0.2)
        x = F.relu(self.bn_emb(self.embedding(x)))
        h1 = self.conv1(x, edge_index)
        h1 = F.relu(self.bn1(h1))
        h1 = F.dropout(h1, p=0.2, training=self.training) + x 
        h2 = self.conv2(h1, edge_index)
        h2 = F.relu(self.bn2(h2))
        h2 = F.dropout(h2, p=0.2, training=self.training) + h1 
        h3 = self.conv3(h2, edge_index)
        h3 = F.relu(self.bn3(h3))
        h3 = F.dropout(h3, p=0.2, training=self.training) + h2 
        h_jk = self.jk([h1, h2, h3])
        out = F.relu(self.fc1(h_jk))
        out = self.fc2(out)
        return out
        
    def extract_embeddings(self, x, edge_index):
        x = F.relu(self.bn_emb(self.embedding(x)))
        h1 = F.relu(self.bn1(self.conv1(x, edge_index))) + x
        h2 = F.relu(self.bn2(self.conv2(h1, edge_index))) + h1
        h3 = F.relu(self.bn3(self.conv3(h2, edge_index))) + h2
        h_jk = self.jk([h1, h2, h3])
        return F.relu(self.fc1(h_jk))

# ---------------------------------------------------------
# 5. Core Execution Block
# ---------------------------------------------------------
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- STEP 1: Process Real-World Circuits ---
    test_graphs = {}
    
    # 3. Simulate AES
    # USE THE INFECTED JSON FOR AES!
    if synthesize_circuit(["AES_128/aes_128.v", "AES_128/round.v", "AES_128/table.v"], "aes_128", "aes_synth.json"):
        # We assume aes_synth_infected.json was generated by rl_trojan_inserter.py
        infected_json = "aes_synth_infected.json"
        if os.path.exists(infected_json):
            print(f"Loading INFECTED AES circuit: {infected_json}")
            aes_nodes_csv, aes_edges_csv = extract_graph_features(infected_json, "aes_128")
        else:
            print(f"WARNING: {infected_json} not found! Falling back to original.")
            aes_nodes_csv, aes_edges_csv = extract_graph_features("aes_synth.json", "aes_128")
            
        df = pd.read_csv(aes_nodes_csv)
        x = torch.tensor(df.iloc[:, 1:9].values, dtype=torch.float)
        y = torch.tensor(df['Label'].values, dtype=torch.long)
        
        node_to_idx = {name: idx for idx, name in enumerate(df['Node'].values)}
        df_edges = pd.read_csv(aes_edges_csv)
        src_indices, dst_indices = [], []
        for _, row in df_edges.iterrows():
            if str(row['src']) in node_to_idx and str(row['dst']) in node_to_idx:
                src_indices.append(node_to_idx[str(row['src'])])
                dst_indices.append(node_to_idx[str(row['dst'])])
                
        if len(src_indices) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
            
        data = Data(x=x, edge_index=edge_index, y=y)
        data.circuit_name = "aes_128"
        test_graphs["aes_128"] = data
    # 4. Process GPS
    circuits = {
        "gps": {
            "files": ["GPS/cacode.v", "GPS/gps.v", "GPS/pcode.v", "GPS/aes_192.v"],
            "top": "gps",
            "synth": "gps_synth.v"
        }
    }
    
    
    for c_name, c_data in circuits.items():
        # Synthesize
        success = synthesize_circuit(c_data["files"], c_data["top"], c_data["synth"])
        if success:
            # Extract Graph & Features
            nodes_csv, edges_csv = extract_graph_features(c_data["synth"], c_data["top"])
            
            # Load into PyG Data object
            df = pd.read_csv(nodes_csv)
            x = torch.tensor(df.iloc[:, 1:9].values, dtype=torch.float)
            y = torch.tensor(df['Label'].values, dtype=torch.long)
            
            node_to_idx = {name: idx for idx, name in enumerate(df['Node'].values)}
            df_edges = pd.read_csv(edges_csv)
            src_indices, dst_indices = [], []
            for _, row in df_edges.iterrows():
                if str(row['src']) in node_to_idx and str(row['dst']) in node_to_idx:
                    src_indices.append(node_to_idx[str(row['src'])])
                    dst_indices.append(node_to_idx[str(row['dst'])])
                    
            if len(src_indices) == 0:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
                
            data = Data(x=x, edge_index=edge_index, y=y)
            data.circuit_name = c_name
            test_graphs[c_name] = data
            
    print(f"\nSuccessfully generated {len(test_graphs)} real-world graphs.")

    # --- STEP 2: Load RS232 and AES Datasets for Training ---
    print("\nLoading Training Datasets (RS232 + AES)...")
    train_graphs = []
    
    # 1. Load RS232 Datasets
    rs232_files = sorted(glob.glob("training_datasets/rs232_output_datasets/dataset_RS232_*.csv"))
    
    # 2. Load AES Datasets (Domain Adaptation)
    aes_files = sorted(glob.glob("aes_output_datasets/dataset_AES_*.csv"))
    
    node_files = rs232_files + aes_files
    
    if len(node_files) == 0:
        print("WARNING: No training datasets found!")
        
    for nf in node_files:
        if "AES" in nf:
            ef = nf.replace("dataset_", "edges_").replace("aes_output_datasets", "aes_edges_datasets")
        else:
            ef = nf.replace("dataset_", "edges_").replace("rs232_output_datasets", "rs232_edges_datasets")
            
        if os.path.exists(ef):
            df = pd.read_csv(nf)
            x = torch.tensor(df.iloc[:, 1:9].values, dtype=torch.float)
            y = torch.tensor(df['Label'].values, dtype=torch.long)
            
            if y.sum() == 0: continue
            
            node_to_idx = {name: idx for idx, name in enumerate(df['Node'].values)}
            df_edges = pd.read_csv(ef)
            src_indices, dst_indices = [], []
            for _, row in df_edges.iterrows():
                if str(row['src']) in node_to_idx and str(row['dst']) in node_to_idx:
                    src_indices.append(node_to_idx[str(row['src'])])
                    dst_indices.append(node_to_idx[str(row['dst'])])
            edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long) if src_indices else torch.empty((2, 0), dtype=torch.long)
            train_graphs.append(Data(x=x, edge_index=edge_index, y=y))
            
    print(f"Loaded {len(train_graphs)} RS232 graphs for training.")
    
    # --- STEP 3: Train GNN Extractor ---
    if len(train_graphs) == 0:
        print("ERROR: Training graphs not found! Cannot train model.")
        sys.exit(1)
        
    train_loader = DataLoader(train_graphs, batch_size=4, shuffle=True)
    total_nodes = sum([g.num_nodes for g in train_graphs])
    total_ht = sum([g.y.sum().item() for g in train_graphs])
    class_weights = torch.tensor([1.0, min((total_nodes - total_ht) / max(1, total_ht) * 0.25, 50.0)], dtype=torch.float).to(device)
    
    gnn = GNN4GateHybrid().to(device)
    optimizer = torch.optim.AdamW(gnn.parameters(), lr=0.005)
    criterion = FocalLoss(alpha=class_weights)
    
    print("\nTraining GNN Feature Extractor on RS232...")
    gnn.train()
    for epoch in range(30):
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = gnn(data.x, data.edge_index)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            
    # Extract Training Embeddings
    gnn.eval()
    X_train_emb, y_train_emb = [], []
    with torch.no_grad():
        for data in train_loader:
            data = data.to(device)
            emb = gnn.extract_embeddings(data.x, data.edge_index)
            X_train_emb.append(emb.cpu())
            y_train_emb.append(data.y.cpu())
    X_train_emb = torch.cat(X_train_emb).numpy()
    y_train_emb = torch.cat(y_train_emb).numpy()
    
    # --- STEP 4: Train ML Models ---
    print("\nTraining ML Models...")
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=50, class_weight="balanced", n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=50, class_weight="balanced", n_jobs=-1),
        "SVM": SVC(class_weight="balanced"),
        "XGBoost": XGBClassifier(scale_pos_weight=class_weights[1].item(), use_label_encoder=False, eval_metric='logloss'),
        "CatBoost": CatBoostClassifier(auto_class_weights="Balanced", verbose=0)
    }
    
    for name, model in models.items():
        model.fit(X_train_emb, y_train_emb)
        
    # --- STEP 5: Evaluate on Real-World Circuits ---
    results = []
    print("\n========================================================")
    print("INFERENCE AND EVALUATION ON REAL-WORLD CIRCUITS")
    print("========================================================")
    
    for c_name, c_graph in test_graphs.items():
        print(f"\n--- Evaluating {c_name} ---")
        
        c_graph = c_graph.to(device)
        
        # 1. GNN Solely Inference (with Softmax)
        gnn_start = time.time()
        with torch.no_grad():
            gnn_logits = gnn(c_graph.x, c_graph.edge_index)
            gnn_preds = gnn_logits.argmax(dim=1).cpu().numpy()
            c_emb = gnn.extract_embeddings(c_graph.x, c_graph.edge_index).cpu().numpy()
        gnn_inf_time = time.time() - gnn_start
        
        y_true = c_graph.y.cpu().numpy()
        
        # Safe metric calculation (handles cases with 0 Trojans)
        def calc_metrics(y_t, y_p):
            acc = accuracy_score(y_t, y_p)
            prec = precision_score(y_t, y_p, zero_division=0)
            rec = recall_score(y_t, y_p, zero_division=0)
            f1 = f1_score(y_t, y_p, zero_division=0)
            return acc, prec, rec, f1
            
        acc, prec, rec, f1 = calc_metrics(y_true, gnn_preds)
        results.append({
            "Circuit": c_name,
            "Model": "GNN_Solely",
            "Accuracy": acc, "Precision": prec, "Recall": rec, "F1_Score": f1,
            "Inference_Time_s": gnn_inf_time
        })
        print(f"GNN_Solely | Precision: {prec:.4f} | Recall: {rec:.4f} | Time: {gnn_inf_time:.4f}s")
        
        # 2. ML Models Inference
        for m_name, model in models.items():
            ml_start = time.time()
            m_preds = model.predict(c_emb)
            ml_inf_time = (time.time() - ml_start) + gnn_inf_time # Total time = GNN extraction + ML inference
            
            acc, prec, rec, f1 = calc_metrics(y_true, m_preds)
            results.append({
                "Circuit": c_name,
                "Model": m_name,
                "Accuracy": acc, "Precision": prec, "Recall": rec, "F1_Score": f1,
                "Inference_Time_s": ml_inf_time
            })
            print(f"{m_name.ljust(10)} | Precision: {prec:.4f} | Recall: {rec:.4f} | Time: {ml_inf_time:.4f}s")
            
    # Save Final Report
    df_results = pd.DataFrame(results)
    df_results.to_csv("realworld_evaluation_metrics.csv", index=False)
    print("\nSaved comprehensive results to realworld_evaluation_metrics.csv!")
