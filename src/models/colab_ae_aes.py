# ==============================================================================
# COLAB SCRIPT FOR UNSUPERVISED AUTOENCODER TRAINING (AES)
# ==============================================================================

import os
import sys

try:
    import torch_geometric
except ImportError:
    print("Installing PyTorch Geometric. This might take a minute...")
    os.system("pip install torch_geometric")
    import torch_geometric

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
import glob
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score

print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch Geometric version: {torch_geometric.__version__}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==============================================================================
# 1. Dataset Loading Function
# ==============================================================================

def load_pyg_dataset(csv_path, edge_csv_path):
    df = pd.read_csv(csv_path)
    nodes = df['Node'].values
    node_to_idx = {name: idx for idx, name in enumerate(nodes)}
    
    feature_cols = ['f1_TP', 'f2_TPDiff', 'f3_Rare', 'f4_NbMean', 'f5_NbDist', 'f6_LZ', 'f7_FanIn', 'f8_FanOut']
    x = df[feature_cols].values
    
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
# 2. Autoencoder Architectures
# ==============================================================================

class MLPAutoencoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(MLPAutoencoder, self).__init__()
        self.enc1 = torch.nn.Linear(in_channels, hidden_channels)
        self.enc2 = torch.nn.Linear(hidden_channels, hidden_channels // 2)
        self.dec1 = torch.nn.Linear(hidden_channels // 2, hidden_channels)
        self.dec2 = torch.nn.Linear(hidden_channels, in_channels)

    def forward(self, x, edge_index=None):
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        h = F.relu(self.dec1(h))
        return self.dec2(h)

class GCNAutoencoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCNAutoencoder, self).__init__()
        self.enc1 = GCNConv(in_channels, hidden_channels)
        self.enc2 = GCNConv(hidden_channels, hidden_channels // 2)
        self.dec1 = GCNConv(hidden_channels // 2, hidden_channels)
        self.dec2 = GCNConv(hidden_channels, in_channels)

    def forward(self, x, edge_index):
        h = F.relu(self.enc1(x, edge_index))
        h = F.relu(self.enc2(h, edge_index))
        h = F.relu(self.dec1(h, edge_index))
        return self.dec2(h, edge_index)

class GATAutoencoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GATAutoencoder, self).__init__()
        self.enc1 = GATConv(in_channels, hidden_channels)
        self.enc2 = GATConv(hidden_channels, hidden_channels // 2)
        self.dec1 = GATConv(hidden_channels // 2, hidden_channels)
        self.dec2 = GATConv(hidden_channels, in_channels)

    def forward(self, x, edge_index):
        h = F.relu(self.enc1(x, edge_index))
        h = F.relu(self.enc2(h, edge_index))
        h = F.relu(self.dec1(h, edge_index))
        return self.dec2(h, edge_index)

# ==============================================================================
# 3. Training & Anomaly Detection Pipeline
# ==============================================================================

def train_autoencoder(train_loader, test_loader, model_class, epochs=100):
    print(f"\n--- Unsupervised Anomaly Detection ({model_class.__name__}) ---")
    
    model = model_class(in_channels=8, hidden_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training: Only train on Normal (Benign) nodes (y == 0)
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            x_recon = model(batch.x, batch.edge_index)
            
            # Mask to find only Benign nodes
            benign_mask = (batch.y == 0)
            
            # MSE Loss only on Benign nodes to learn perfect reconstruction of normal circuits
            loss = F.mse_loss(x_recon[benign_mask], batch.x[benign_mask])
            
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
                
    # Evaluation
    model.eval()
    
    # 1. Calculate the Threshold from the Training set's Benign nodes
    train_errors = []
    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device)
            x_recon = model(batch.x, batch.edge_index)
            benign_mask = (batch.y == 0)
            error = torch.mean((x_recon[benign_mask] - batch.x[benign_mask])**2, dim=1)
            train_errors.extend(error.cpu().numpy())
            
    # Set threshold as the 95th percentile of normal reconstruction errors
    if len(train_errors) == 0:
        threshold = 0.5
    else:
        threshold = np.percentile(train_errors, 95)
    
    # 2. Test on Unseen Circuits
    all_preds = []
    all_labels = []
    all_errors = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            x_recon = model(batch.x, batch.edge_index)
            
            # Calculate MSE error for ALL nodes in the test set
            error = torch.mean((x_recon - batch.x)**2, dim=1)
            
            # If error is greater than threshold, classify as Trojan (1)
            pred = (error > threshold).long()
            
            all_errors.extend(error.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_errors)
    
    print(f"Anomaly Threshold Calculated: {threshold:.4f}")
    print(f"Test Accuracy: {acc*100:.2f}%")
    print(f"Test Precision: {prec:.4f}")
    print(f"Test Recall: {rec:.4f}")
    print(f"Test F1-Score (Trojan class): {f1:.4f}")
    print(f"Test ROC-AUC Score:  {auc:.4f}")
    return model

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    train_node_dir = "aes_split/train/nodes"
    train_edge_dir = "aes_split/train/edges"
    test_node_dir = "aes_split/test/nodes"
    test_edge_dir = "aes_split/test/edges"
    
    if not os.path.exists(train_node_dir) or not os.path.exists(test_node_dir):
        print("ERROR: Cannot find the split directories. Please upload and extract aes_split_dataset.zip")
        sys.exit()
        
    train_graphs = []
    test_graphs = []
    
    print("Building PyTorch Geometric Graph Objects for Train Set (AES)...")
    for node_file in glob.glob(f"{train_node_dir}/*.csv"):
        basename = os.path.basename(node_file)
        edge_basename = basename.replace("dataset_AES", "edges_AES")
        edge_file = os.path.join(train_edge_dir, edge_basename)
        if os.path.exists(edge_file):
            train_graphs.append(load_pyg_dataset(node_file, edge_file))
            
    print("Building PyTorch Geometric Graph Objects for Test Set (AES)...")
    for node_file in glob.glob(f"{test_node_dir}/*.csv"):
        basename = os.path.basename(node_file)
        edge_basename = basename.replace("dataset_AES", "edges_AES")
        edge_file = os.path.join(test_edge_dir, edge_basename)
        if os.path.exists(edge_file):
            test_graphs.append(load_pyg_dataset(node_file, edge_file))
            
    print(f"Successfully constructed {len(train_graphs)} Train graphs and {len(test_graphs)} Test graphs!")
    
    train_loader = DataLoader(train_graphs, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=4, shuffle=False)
    
    mlp_ae = train_autoencoder(train_loader, test_loader, MLPAutoencoder, epochs=100)
    gcn_ae = train_autoencoder(train_loader, test_loader, GCNAutoencoder, epochs=100)
    gat_ae = train_autoencoder(train_loader, test_loader, GATAutoencoder, epochs=100)

    print("\nSaving AES Autoencoder Models...")
    torch.save(mlp_ae.state_dict(), "aes_mlp_ae.pth")
    torch.save(gcn_ae.state_dict(), "aes_gcn_ae.pth")
    torch.save(gat_ae.state_dict(), "aes_gat_ae.pth")
    print("Saved aes_mlp_ae.pth, aes_gcn_ae.pth, aes_gat_ae.pth successfully!")
