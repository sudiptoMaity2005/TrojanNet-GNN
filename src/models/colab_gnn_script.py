# ==============================================================================
# COPY THIS ENTIRE SCRIPT INTO A GOOGLE COLAB NOTEBOOK CELL AND RUN IT
# ==============================================================================

# 1. Install PyTorch Geometric (Colab usually has PyTorch pre-installed)
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
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch Geometric version: {torch_geometric.__version__}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==============================================================================
# 2. Dataset Loading Function
# ==============================================================================

def load_pyg_dataset(csv_path, edge_csv_path):
    # Load Nodes
    df = pd.read_csv(csv_path)
    nodes = df['Node'].values
    node_to_idx = {name: idx for idx, name in enumerate(nodes)}
    
    # Extract Features (Columns 1 to 8: f1 -> f8)
    # Scale FanIn/FanOut so they don't overpower probabilities
    feature_cols = ['f1_TP', 'f2_TPDiff', 'f3_Rare', 'f4_NbMean', 'f5_NbDist', 'f6_LZ', 'f7_FanIn', 'f8_FanOut']
    x = df[feature_cols].values
    
    # Normalize features
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0) + 1e-8
    x = (x - x_mean) / x_std
    x = torch.tensor(x, dtype=torch.float)
    
    # Labels
    y = torch.tensor(df['Label'].values, dtype=torch.long)
    
    # Load Edges
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
        # Create empty edge index if graph has no edges
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        
    return Data(x=x, edge_index=edge_index, y=y)

# ==============================================================================
# 3. Model Definitions (GCN and GAT)
# ==============================================================================

class TrojanGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(TrojanGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class TrojanGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(TrojanGAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=False)
        self.conv2 = GATConv(hidden_channels, 2, heads=4, concat=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# ==============================================================================
# 4. Training Pipelines
# ==============================================================================

def train_inductive(train_loader, test_loader, model_class, epochs=50):
    print(f"\n--- Inductive Training ({model_class.__name__}) across All RS232 Circuits ---")
    
    model = model_class(in_channels=8, hidden_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    model.eval()
    all_preds = []
    all_labels = []
    
    for batch in test_loader:
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch.x, batch.edge_index).argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    print(f"Test Accuracy: {acc*100:.2f}%")
    print(f"Test Precision: {prec:.4f}")
    print(f"Test Recall: {rec:.4f}")
    print(f"Test F1-Score (Trojan class): {f1:.4f}")
    return model

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    train_node_dir = "rs232_split/train/nodes"
    train_edge_dir = "rs232_split/train/edges"
    test_node_dir = "rs232_split/test/nodes"
    test_edge_dir = "rs232_split/test/edges"
    
    if not os.path.exists(train_node_dir) or not os.path.exists(test_node_dir):
        print("ERROR: Cannot find the split directories. Please upload and extract rs232_split_dataset.zip")
        sys.exit()
        
    train_graphs = []
    test_graphs = []
    
    print("Building PyTorch Geometric Graph Objects for Train Set...")
    for node_file in glob.glob(f"{train_node_dir}/*.csv"):
        basename = os.path.basename(node_file)
        edge_basename = basename.replace("dataset_RS232", "edges_RS232")
        edge_file = os.path.join(train_edge_dir, edge_basename)
        if os.path.exists(edge_file):
            train_graphs.append(load_pyg_dataset(node_file, edge_file))
            
    print("Building PyTorch Geometric Graph Objects for Test Set...")
    for node_file in glob.glob(f"{test_node_dir}/*.csv"):
        basename = os.path.basename(node_file)
        edge_basename = basename.replace("dataset_RS232", "edges_RS232")
        edge_file = os.path.join(test_edge_dir, edge_basename)
        if os.path.exists(edge_file):
            test_graphs.append(load_pyg_dataset(node_file, edge_file))
            
    print(f"Successfully constructed {len(train_graphs)} Train graphs and {len(test_graphs)} Test graphs!")
    
    if len(train_graphs) == 0 or len(test_graphs) == 0:
        sys.exit()
        
    train_loader = DataLoader(train_graphs, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=4, shuffle=False)
    
    gcn_induct = train_inductive(train_loader, test_loader, TrojanGCN, epochs=50)
    gat_induct = train_inductive(train_loader, test_loader, TrojanGAT, epochs=50)

    print("\nTraining completely finished!")
    
    print("\nSaving Models...")
    torch.save(gcn_induct.state_dict(), "rs232_gcn_model.pth")
    torch.save(gat_induct.state_dict(), "rs232_gat_model.pth")
    print("Saved rs232_gcn_model.pth and rs232_gat_model.pth successfully!")
