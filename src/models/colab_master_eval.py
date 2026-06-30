# ==============================================================================
# COPY THIS ENTIRE SCRIPT INTO A GOOGLE COLAB NOTEBOOK CELL AND RUN IT
# MASTER EVALUATION SCRIPT: ISCAS+RS232 (TRAIN) -> AES/GPS (TEST)
# ==============================================================================

# Install PyTorch Geometric and Boosting libraries before importing
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
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, JumpingKnowledge
from torch_geometric.utils import dropout_edge
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# ---------------------------------------------------------
# 1. Dataset Loading
# ---------------------------------------------------------
def load_datasets_from_folder(folder_path):
    print(f"Loading graph datasets from: {folder_path} ...")
    graphs = []
    
    # 1. Load .pt files
    pt_files = glob.glob(os.path.join(folder_path, "**", "*.pt"), recursive=True)
    for pt_file in pt_files:
        try:
            data = torch.load(pt_file, weights_only=False)
            if data.x.shape[0] > 0:
                x_mean = data.x.mean(dim=0, keepdim=True)
                x_std = data.x.std(dim=0, unbiased=False, keepdim=True) + 1e-8
                data.x = (data.x - x_mean) / x_std
            data.circuit_name = os.path.basename(pt_file).replace('.pt', '')
            if not hasattr(data, 'y') or data.y is None:
                data.y = torch.zeros(data.x.shape[0], dtype=torch.long)
            graphs.append(data)
        except Exception as e:
            print(f"Failed to load {pt_file}: {e}")
            
    # 2. Load .csv files (from previous RS232 / Pipeline outputs)
    csv_node_files = glob.glob(os.path.join(folder_path, "**", "dataset_*.csv"), recursive=True)
    for node_csv in csv_node_files:
        try:
            edge_csv = node_csv.replace("dataset_", "edges_").replace("rs232_output_datasets", "rs232_edges_datasets")
            if not os.path.exists(edge_csv):
                # Fallback check
                alt_edge = os.path.join(os.path.dirname(node_csv).replace("rs232_output_datasets", "rs232_edges_datasets"), "edges_" + os.path.basename(node_csv).replace("dataset_", ""))
                if os.path.exists(alt_edge):
                    edge_csv = alt_edge
                else:
                    continue
                    
            df_nodes = pd.read_csv(node_csv)
            # Find feature columns (exclude Node, Label)
            feat_cols = [c for c in df_nodes.columns if c not in ['Node', 'Label']]
            x = torch.tensor(df_nodes[feat_cols].values, dtype=torch.float)
            
            # Normalization
            if x.shape[0] > 0:
                x_mean = x.mean(dim=0, keepdim=True)
                x_std = x.std(dim=0, unbiased=False, keepdim=True) + 1e-8
                x = (x - x_mean) / x_std
                
            y = torch.tensor(df_nodes['Label'].values, dtype=torch.long) if 'Label' in df_nodes.columns else torch.zeros(x.shape[0], dtype=torch.long)
            
            df_edges = pd.read_csv(edge_csv)
            node_to_idx = {name: idx for idx, name in enumerate(df_nodes['Node'].values)}
            src_indices, dst_indices = [], []
            for _, row in df_edges.iterrows():
                if str(row['src']) in node_to_idx and str(row['dst']) in node_to_idx:
                    src_indices.append(node_to_idx[str(row['src'])])
                    dst_indices.append(node_to_idx[str(row['dst'])])
                    
            edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long) if src_indices else torch.empty((2, 0), dtype=torch.long)
            
            data = Data(x=x, edge_index=edge_index, y=y)
            data.circuit_name = os.path.basename(node_csv).replace('.csv', '').replace('dataset_', '')
            graphs.append(data)
        except Exception as e:
            print(f"Failed to load CSV {node_csv}: {e}")
            
    print(f"Successfully loaded {len(graphs)} graphs from {folder_path}.")
    return graphs

# ---------------------------------------------------------
# 2. Hybrid GNN Architecture
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
    # Depending on what files you upload, features might be 23 or 8.
    # We will dynamically set this below based on the first loaded graph.
    def __init__(self, in_channels, hidden_channels=64, out_channels=2):
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
        return self.fc2(out)
        
    def extract_embeddings(self, x, edge_index):
        x = F.relu(self.bn_emb(self.embedding(x)))
        h1 = F.relu(self.bn1(self.conv1(x, edge_index))) + x
        h2 = F.relu(self.bn2(self.conv2(h1, edge_index))) + h1
        h3 = F.relu(self.bn3(self.conv3(h2, edge_index))) + h2
        h_jk = self.jk([h1, h2, h3])
        return F.relu(self.fc1(h_jk))


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Folder paths - Create these in Colab and upload your .pt files!
    TRAIN_DIR = "training_datasets"
    TEST_DIR = "testing_datasets"
    
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    
    print("\n========================================================")
    print("STEP 1: LOADING DATASETS")
    print("========================================================")
    
    train_graphs = load_datasets_from_folder(TRAIN_DIR)
    test_graphs = load_datasets_from_folder(TEST_DIR)
    
    if len(train_graphs) == 0:
        print(f"ERROR: No .pt files found in '{TRAIN_DIR}'. Please upload them.")
        import sys; sys.exit(1)
    if len(test_graphs) == 0:
        print(f"ERROR: No .pt files found in '{TEST_DIR}'. Please upload them.")
        import sys; sys.exit(1)
        
    in_channels = train_graphs[0].x.shape[1]
    print(f"\nDetected {in_channels} features per node.")

    train_loader = DataLoader(train_graphs, batch_size=4, shuffle=True)
    
    # Compute Class Weights for extreme imbalance
    total_nodes = sum([g.num_nodes for g in train_graphs])
    total_ht = sum([g.y.sum().item() for g in train_graphs])
    total_normal = total_nodes - total_ht
    raw_ratio = total_normal / max(1, total_ht)
    dampened_ratio = min(raw_ratio * 0.25, 50.0)
    class_weights = torch.tensor([1.0, dampened_ratio], dtype=torch.float).to(device)
    
    print("\n========================================================")
    print("STEP 2: TRAINING GNN FEATURE EXTRACTOR")
    print("========================================================")
    
    gnn = GNN4GateHybrid(in_channels=in_channels).to(device)
    optimizer = torch.optim.AdamW(gnn.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = FocalLoss(alpha=class_weights)
    
    gnn.train()
    epochs = 40
    for epoch in range(epochs):
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = gnn(data.x, data.edge_index)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

    print("\nExtracting Training Embeddings for ML...")
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

    print("\n========================================================")
    print("STEP 3: TRAINING CLASSICAL ML MODELS")
    print("========================================================")
    
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=50, class_weight="balanced", n_jobs=-1, random_state=42),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=50, class_weight="balanced", n_jobs=-1, random_state=42),
        "SVM": SVC(class_weight="balanced", random_state=42),
        "XGBoost": XGBClassifier(scale_pos_weight=class_weights[1].item(), use_label_encoder=False, eval_metric='logloss', random_state=42),
        "CatBoost": CatBoostClassifier(auto_class_weights="Balanced", verbose=0, random_state=42)
    }
    
    for name, model in models.items():
        print(f"Fitting {name}...")
        model.fit(X_train_emb, y_train_emb)

    print("\n========================================================")
    print("STEP 4: EVALUATING UNSEEN CIRCUITS (AES_128, GPS)")
    print("========================================================")
    
    results = []
    
    def safe_metrics(y_true, y_pred):
        # Handles cases where the test circuit has exactly 0 Trojans
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        return acc, prec, rec, f1

    for c_graph in test_graphs:
        c_name = c_graph.circuit_name
        print(f"\n--- Testing on Circuit: {c_name} ---")
        
        # We use a dataloader to prevent in-place device modification bugs
        test_loader = DataLoader([c_graph], batch_size=1, shuffle=False)
        
        X_test_emb, y_test_emb, gnn_preds = [], [], []
        
        gnn_start = time.time()
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                
                logits = gnn(data.x, data.edge_index)
                preds = logits.argmax(dim=1)
                emb = gnn.extract_embeddings(data.x, data.edge_index)
                
                gnn_preds.append(preds.cpu())
                X_test_emb.append(emb.cpu())
                y_test_emb.append(data.y.cpu())
                
        gnn_inf_time = time.time() - gnn_start
        
        X_test_emb = torch.cat(X_test_emb).numpy()
        y_test_emb = torch.cat(y_test_emb).numpy()
        gnn_preds = torch.cat(gnn_preds).numpy()
        
        # 1. GNN Solely
        acc, prec, rec, f1 = safe_metrics(y_test_emb, gnn_preds)
        results.append({
            "Circuit": c_name, "Model": "GNN_Solely",
            "Accuracy": acc, "Precision": prec, "Recall": rec, "F1_Score": f1,
            "Inference_Time_s": gnn_inf_time
        })
        print(f"  > GNN_Solely | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | Time: {gnn_inf_time:.4f}s")
        
        # 2. ML Models
        for m_name, model in models.items():
            ml_start = time.time()
            m_preds = model.predict(X_test_emb)
            # Total time includes GNN embedding extraction + ML prediction
            ml_inf_time = (time.time() - ml_start) + gnn_inf_time
            
            acc, prec, rec, f1 = safe_metrics(y_test_emb, m_preds)
            results.append({
                "Circuit": c_name, "Model": m_name,
                "Accuracy": acc, "Precision": prec, "Recall": rec, "F1_Score": f1,
                "Inference_Time_s": ml_inf_time
            })
            print(f"  > {m_name.ljust(10)} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | Time: {ml_inf_time:.4f}s")

    # Save to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv("master_evaluation_metrics.csv", index=False)
    print("\n========================================================")
    print("SUCCESS! All results saved to 'master_evaluation_metrics.csv'")
    print("========================================================")
