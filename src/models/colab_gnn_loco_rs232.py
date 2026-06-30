# ==============================================================================
# COPY THIS ENTIRE SCRIPT INTO A GOOGLE COLAB NOTEBOOK CELL AND RUN IT
# RS232 LEAVE-ONE-CIRCUIT-OUT (LOCO) EVALUATION
# ==============================================================================

# Install dependencies if not present
try:
    import torch_geometric
except ImportError:
    import os
    os.system("pip install torch_geometric xgboost catboost")
    import torch_geometric

import os
import glob
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, JumpingKnowledge
from torch_geometric.utils import dropout_edge
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import ExtraTreesClassifier

# ---------------------------------------------------------
# 1. Dataset Loading for RS232
# ---------------------------------------------------------
def load_rs232_pyg_dataset(csv_path, edge_csv_path, circuit_name):
    df = pd.read_csv(csv_path)
    nodes = df['Node'].values
    node_to_idx = {name: idx for idx, name in enumerate(nodes)}
    
    # RS232 features from autoencoder parsing
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
        
    data = Data(x=x, edge_index=edge_index, y=y)
    data.circuit_name = circuit_name
    return data

def load_all_rs232_datasets(base_node_dir="rs232_output_datasets", base_edge_dir="rs232_edges_datasets", max_circuits=20):
    print(f"Loading RS232 graphs from: {base_node_dir} ...")
    all_graphs = []
    
    # Fetch all RS232 datasets
    node_files = sorted(glob.glob(f"{base_node_dir}/dataset_RS232_*.csv"))
    
    for node_file in node_files:
        basename = os.path.basename(node_file)
        circuit_name = basename.replace("dataset_", "").replace(".csv", "")
        edge_basename = basename.replace("dataset_", "edges_")
        edge_file = os.path.join(base_edge_dir, edge_basename)
        
        if os.path.exists(edge_file):
            try:
                g = load_rs232_pyg_dataset(node_file, edge_file, circuit_name)
                
                # Verify that there are actually Trojans in the circuit to avoid undefined precisions
                if g.y.sum() > 0:
                    all_graphs.append(g)
            except Exception as e:
                print(f"Skipping {circuit_name} due to error: {e}")
                
        if len(all_graphs) >= max_circuits:
            break
            
    print(f"Loaded {len(all_graphs)} valid RS232 graphs (Trojans present).")
    return all_graphs

# ---------------------------------------------------------
# 2. Hybrid GNN Architecture (Feature Extractor)
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
    # Notice in_channels=8 since RS232 has exactly 8 native features
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
            
        x = self.embedding(x)
        x = self.bn_emb(x)
        x = F.relu(x)
        
        h1 = self.conv1(x, edge_index)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=0.2, training=self.training)
        h1 = h1 + x 
        
        h2 = self.conv2(h1, edge_index)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=0.2, training=self.training)
        h2 = h2 + h1 
        
        h3 = self.conv3(h2, edge_index)
        h3 = self.bn3(h3)
        h3 = F.relu(h3)
        h3 = F.dropout(h3, p=0.2, training=self.training)
        h3 = h3 + h2 
        
        h_jk = self.jk([h1, h2, h3])
        
        out = self.fc1(h_jk)
        out = F.relu(out)
        out = self.fc2(out)
        return out
        
    def extract_embeddings(self, x, edge_index):
        x = self.embedding(x)
        x = self.bn_emb(x)
        x = F.relu(x)
        
        h1 = self.conv1(x, edge_index)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = h1 + x
        
        h2 = self.conv2(h1, edge_index)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)
        h2 = h2 + h1
        
        h3 = self.conv3(h2, edge_index)
        h3 = self.bn3(h3)
        h3 = F.relu(h3)
        h3 = h3 + h2
        
        h_jk = self.jk([h1, h2, h3])
        
        out = self.fc1(h_jk)
        out = F.relu(out) # 32-dimensional embedding
        return out

# ---------------------------------------------------------
# 3. LOCO Cross Validation Loop
# ---------------------------------------------------------

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    all_graphs = load_all_rs232_datasets(max_circuits=20)
    
    if len(all_graphs) == 0:
        print("ERROR: No RS232 graphs found. Ensure you have the rs232_output_datasets directory available.")
        import sys; sys.exit(1)
        
    print(f"\n========================================================")
    print(f"STARTING LEAVE-ONE-CIRCUIT-OUT (LOCO) CV ON {len(all_graphs)} CIRCUITS")
    print(f"========================================================")
    
    loco_results = []
    
    # We will test holding out EVERY SINGLE graph, one by one.
    for i, test_graph in enumerate(all_graphs):
        circuit_name = getattr(test_graph, 'circuit_name', f"Circuit_{i}")
        print(f"\n[{i+1}/{len(all_graphs)}] LOCO Fold: Holding out {circuit_name} as TEST SET")
        
        # 19 circuits for Train, 1 circuit for Test
        train_graphs = [g for j, g in enumerate(all_graphs) if j != i]
        
        train_loader = DataLoader(train_graphs, batch_size=4, shuffle=True)
        test_loader = DataLoader([test_graph], batch_size=1, shuffle=False)
        
        # Calculate Class Weights for this specific fold
        total_nodes = sum([g.num_nodes for g in train_graphs])
        total_ht = sum([g.y.sum().item() for g in train_graphs])
        total_normal = total_nodes - total_ht
        
        raw_ratio = total_normal / max(1, total_ht)
        dampened_ratio = min(raw_ratio * 0.25, 50.0)
        class_weights = torch.tensor([1.0, dampened_ratio], dtype=torch.float).to(device)
        
        # Initialize Fresh GNN for each Fold
        model = GNN4GateHybrid().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
        criterion = FocalLoss(alpha=class_weights)
        
        # Phase 1: Train GNN Feature Extractor
        model.train()
        epochs = 30  # Optimized for faster LOCO loops
        for epoch in range(epochs):
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()
                
        # Phase 2: Extract Embeddings
        model.eval()
        X_train_emb, y_train_emb = [], []
        with torch.no_grad():
            for data in train_loader:
                data = data.to(device)
                emb = model.extract_embeddings(data.x, data.edge_index)
                X_train_emb.append(emb.cpu())
                y_train_emb.append(data.y.cpu())
                
        X_train_emb = torch.cat(X_train_emb).numpy()
        y_train_emb = torch.cat(y_train_emb).numpy()
        
        X_test_emb, y_test_emb = [], []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                emb = model.extract_embeddings(data.x, data.edge_index)
                X_test_emb.append(emb.cpu())
                y_test_emb.append(data.y.cpu())
                
        X_test_emb = torch.cat(X_test_emb).numpy()
        y_test_emb = torch.cat(y_test_emb).numpy()
            
        # Phase 3: Train Traditional ML (ExtraTrees)
        et = ExtraTreesClassifier(n_estimators=50, class_weight="balanced", random_state=42, n_jobs=-1)
        et.fit(X_train_emb, y_train_emb)
        et_preds = et.predict(X_test_emb)
        
        # Record Metrics
        prec = precision_score(y_test_emb, et_preds, zero_division=0)
        rec = recall_score(y_test_emb, et_preds, zero_division=0)
        f1 = f1_score(y_test_emb, et_preds, zero_division=0)
        
        print(f"  -> Fold Results | ExtraTrees Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
        
        loco_results.append({
            "Fold": i+1,
            "Held_Out_Circuit": circuit_name,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1
        })
        
    # Phase 4: Final Aggregation
    df_results = pd.DataFrame(loco_results)
    
    print(f"\n========================================================")
    print(f"FINAL LOCO CROSS-VALIDATION RESULTS (ExtraTrees on RS232)")
    print(f"========================================================")
    print(f"Average Precision: {df_results['Precision'].mean():.4f}")
    print(f"Average Recall:    {df_results['Recall'].mean():.4f}")
    print(f"Average F1 Score:  {df_results['F1_Score'].mean():.4f}")
    
    # Save Final Report
    df_results.to_csv("rs232_loco_results.csv", index=False)
    print("\nSaved comprehensive results to rs232_loco_results.csv! Download this for your instructor.")
