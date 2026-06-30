# ==============================================================================
# COPY THIS ENTIRE SCRIPT INTO A GOOGLE COLAB NOTEBOOK CELL AND RUN IT
# FOR THE FINAL SUPERCHARGED GRAPHSAGE BENCHMARK EVALUATION TABLE
# ==============================================================================

import os
import sys

import torch_geometric

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, JumpingKnowledge
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
import glob
import time
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==============================================================================
# 2. Dataset Loading Function
# ==============================================================================

def load_pyg_dataset(csv_path, edge_csv_path):
    df = pd.read_csv(csv_path)
    nodes = df['Node'].values
    node_to_idx = {name: idx for idx, name in enumerate(nodes)}
    
    feature_cols = ['f1_TP', 'f2_TPDiff', 'f3_Rare', 'f4_NbMean', 'f5_NbDist', 'f6_LZ', 'f7_FanIn', 'f8_FanOut']
    x = df[feature_cols].values
    
    # Normalize features
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0) + 1e-8
    x = (x - x_mean) / x_std
    x = torch.tensor(x, dtype=torch.float)
    
    y = torch.tensor(df['Label'].values, dtype=torch.long)
    
    df_edges = pd.read_csv(edge_csv_path)
    src_indices = []
    dst_indices = []
    
    for _, row in df_edges.iterrows():
        src_name = str(row['src'])
        dst_name = str(row['dst'])
        if src_name in node_to_idx and dst_name in node_to_idx:
            src_indices.append(node_to_idx[src_name])
            dst_indices.append(node_to_idx[dst_name])
            
    if len(src_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        
    return Data(x=x, edge_index=edge_index, y=y)

# ==============================================================================
# 3. Model Definition & Training
# ==============================================================================

class SuperchargedSAGE(torch.nn.Module):
    def __init__(self, num_features):
        super(SuperchargedSAGE, self).__init__()
        
        # 1. Expand features to fixed size
        self.embedding = torch.nn.Linear(num_features, 64)
        self.bn_emb = torch.nn.LayerNorm(64)
        
        # 2. Pure Spatial GNN Layers (NO RNN. Pure GraphSAGE like Phase 1)
        self.conv1 = SAGEConv(64, 64)
        self.bn1 = torch.nn.LayerNorm(64)
        
        self.conv2 = SAGEConv(64, 64)
        self.bn2 = torch.nn.LayerNorm(64)
        
        self.conv3 = SAGEConv(64, 64)
        self.bn3 = torch.nn.LayerNorm(64)
        
        # 3. Anti-Oversmoothing: Jumping Knowledge Network
        self.jk = JumpingKnowledge("max")
        
        # 4. MLP Head
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, 2)

    def forward(self, x, edge_index):
        # Initial Embedding
        x = self.embedding(x)
        x = self.bn_emb(x)
        x = F.relu(x)
        
        # Layer 1 (1-hop topology)
        h1 = self.conv1(x, edge_index)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=0.2, training=self.training)
        h1 = h1 + x # Residual Skip Connection
        
        # Layer 2 (2-hop topology)
        h2 = self.conv2(h1, edge_index)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=0.2, training=self.training)
        h2 = h2 + h1 # Residual Skip Connection
        
        # Layer 3 (3-hop topology)
        h3 = self.conv3(h2, edge_index)
        h3 = self.bn3(h3)
        h3 = F.relu(h3)
        h3 = F.dropout(h3, p=0.2, training=self.training)
        h3 = h3 + h2 # Residual Skip Connection
        
        # Jumping Knowledge: Combine all hops
        h_jk = self.jk([h1, h2, h3])
        
        # MLP Head: Classification
        out = self.fc1(h_jk)
        out = F.relu(out)
        out = self.fc2(out)
        return out

class FocalLoss(torch.nn.Module):
    # EXTREME GAMMA: Forces model to hyper-focus on minority class (sequential Trojans)
    def __init__(self, alpha=None, gamma=3.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt)**self.gamma * BCE_loss
        
        if self.alpha is not None:
            at = self.alpha.gather(0, targets.data.view(-1))
            F_loss = F_loss * at
            
        return F_loss.mean()

def train_supercharged(train_graphs):
    loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
    
    model = SuperchargedSAGE(num_features=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    
    # Calculate class weights for highly imbalanced Trojan dataset
    total_nodes = sum([g.num_nodes for g in train_graphs])
    total_positives = sum([g.y.sum().item() for g in train_graphs])
    total_negatives = total_nodes - total_positives
    
    # UNCAPPED MATHEMATICAL RATIO: Pushes the weight to extreme levels
    weight_0 = 1.0
    weight_1 = float(total_negatives) / max(total_positives, 1)
    
    class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float).to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=3.0)
    
    print(f"\n--- Training Supercharged GraphSAGE on {len(train_graphs)} graphs ---")
    print(f"Dataset Imbalance: {total_negatives} Clean vs {total_positives} Trojans")
    print(f"Applying RAW Class Weights: Class 0: {weight_0:.2f}, Class 1: {weight_1:.2f}")
    print(f"Applying EXTREME Focal Loss Gamma: 3.0")
    
    model.train()
    for epoch in range(250):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    print("Training Complete!")
    return model

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    base_dir = "datasets_for_gnn"
    sizes = ["small", "medium", "Large"]
    
    if not os.path.exists(base_dir):
        print("ERROR: Could not find datasets_for_gnn folder! Please unzip it in Colab.")
        sys.exit()

    print("Loading Datasets...")
    
    circuit_families = {}
    
    for size in sizes:
        # Load Train
        train_nodes = glob.glob(f"{base_dir}/{size}/train/nodes/*.csv")
        for n_csv in train_nodes:
            basename = os.path.basename(n_csv).replace("dataset_", "").replace(".csv", "")
            circ_name = basename.split('_')[0]
            e_csv = os.path.join(base_dir, size, "train", "edges", f"edges_{basename}.csv")
            if os.path.exists(e_csv):
                if circ_name not in circuit_families:
                    circuit_families[circ_name] = {'size': size, 'train': [], 'test': []}
                circuit_families[circ_name]['train'].append(load_pyg_dataset(n_csv, e_csv))
                
        # Load Test
        test_nodes = glob.glob(f"{base_dir}/{size}/test/nodes/*.csv")
        for n_csv in test_nodes:
            basename = os.path.basename(n_csv).replace("dataset_", "").replace(".csv", "")
            circ_name = basename.split('_')[0]
            e_csv = os.path.join(base_dir, size, "test", "edges", f"edges_{basename}.csv")
            if os.path.exists(e_csv):
                if circ_name not in circuit_families:
                    circuit_families[circ_name] = {'size': size, 'train': [], 'test': []}
                circuit_families[circ_name]['test'].append(load_pyg_dataset(n_csv, e_csv))

    print("\n==================================================")
    print("FINAL EVALUATION TABLE (SUPERCHARGED GRAPHSAGE)")
    print("==================================================")
    print("| Circuit Size | Circuit Name | Node Count | Accuracy | Precision | Recall | F1-Score | Inference Time |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    csv_data = []

    global_train_graphs = []
    for circ_name, data_dict in circuit_families.items():
        global_train_graphs.extend(data_dict['train'])
        
    print(f"\n[GLOBAL TRAINING] Training a single master model on {len(global_train_graphs)} total graphs...")
    global_model = train_supercharged(global_train_graphs)
    global_model.eval()

    for circ_name, data_dict in circuit_families.items():
        test_graphs = data_dict['test']
        size = data_dict['size']
        
        if len(test_graphs) == 0:
            continue
        
        stats = []
        for data in test_graphs:
            data = data.to(device)
            start_time = time.time()
            with torch.no_grad():
                pred = global_model(data.x, data.edge_index).argmax(dim=1)
            inf_time = time.time() - start_time
            
            y_true = data.y.cpu().numpy()
            y_pred = pred.cpu().numpy()
            
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            nodes = data.num_nodes
            stats.append((nodes, acc, prec, rec, f1, inf_time))
            
        # Average metrics
        avg_nodes = np.mean([s[0] for s in stats])
        avg_acc = np.mean([s[1] for s in stats])
        avg_prec = np.mean([s[2] for s in stats])
        avg_rec = np.mean([s[3] for s in stats])
        avg_f1 = np.mean([s[4] for s in stats])
        avg_time = np.mean([s[5] for s in stats])
        
        display_size = "Small" if size == "small" else "Medium" if size == "medium" else "Large"
        
        print(f"| **{display_size}** | **{circ_name}** | ~{int(avg_nodes)} | {avg_acc*100:.1f}% | {avg_prec*100:.1f}% | {avg_rec*100:.1f}% | {avg_f1:.2f} | {avg_time:.4f} s |")
        
        csv_data.append({
            "Circuit Size": display_size,
            "Circuit Name": circ_name,
            "Node Count": int(avg_nodes),
            "Accuracy (%)": round(avg_acc * 100, 1),
            "Precision (%)": round(avg_prec * 100, 1),
            "Recall (%)": round(avg_rec * 100, 1),
            "F1-Score": round(avg_f1, 2),
            "Inference Time (s)": round(avg_time, 4)
        })

    # Save to CSV
    results_df = pd.DataFrame(csv_data)
    results_df.to_csv('evaluation_results_supercharged.csv', index=False)
    
    print("\nEVALUATION COMPLETE.")
    print("The table has also been saved to 'evaluation_results_supercharged.csv'.")
    print("You can download the CSV from the 'Files' tab on the left in Colab!")
