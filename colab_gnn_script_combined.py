# ==============================================================================
# COPY THIS ENTIRE SCRIPT INTO A GOOGLE COLAB NOTEBOOK CELL AND RUN IT
# FOR THE ULTIMATE COMBINED TEST (RS232 + AES)
# ==============================================================================

# 1. Install PyTorch Geometric
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
# 3. Model Definitions
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

def train_inductive(train_loader, test_loader, model_class, epochs=100):
    print(f"\n--- Inductive Training ({model_class.__name__}) across ALL COMBINED CIRCUITS (RS232 + AES) ---")
    
    model = model_class(in_channels=8, hidden_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
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
            
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    y_true = all_labels
    y_pred = all_preds
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"Test Accuracy: {acc*100:.2f}%")
    print(f"Test Precision: {prec:.4f}")
    print(f"Test Recall: {rec:.4f}")
    print(f"Test F1-Score (Trojan class): {f1:.4f}")
    return model

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    
    # 1. Load RS232
    rs232_train_node_dir = "rs232_split/train/nodes"
    rs232_train_edge_dir = "rs232_split/train/edges"
    rs232_test_node_dir = "rs232_split/test/nodes"
    rs232_test_edge_dir = "rs232_split/test/edges"
    
    # 2. Load AES
    aes_train_node_dir = "aes_split/train/nodes"
    aes_train_edge_dir = "aes_split/train/edges"
    aes_test_node_dir = "aes_split/test/nodes"
    aes_test_edge_dir = "aes_split/test/edges"
    
    train_graphs = []
    test_graphs = []
    
    def load_folder(node_dir, edge_dir, prefix, target_list):
        if not os.path.exists(node_dir) or not os.path.exists(edge_dir):
            return
        for node_file in glob.glob(f"{node_dir}/*.csv"):
            basename = os.path.basename(node_file)
            edge_basename = basename.replace(f"dataset_{prefix}", f"edges_{prefix}")
            edge_file = os.path.join(edge_dir, edge_basename)
            if os.path.exists(edge_file):
                target_list.append(load_pyg_dataset(node_file, edge_file))

    print("Loading RS232 and AES Split Circuits...")
    
    # Load RS232 directly into train/test
    load_folder(rs232_train_node_dir, rs232_train_edge_dir, "RS232", train_graphs)
    load_folder(rs232_test_node_dir, rs232_test_edge_dir, "RS232", test_graphs)
    
    # Load AES directly into train/test
    load_folder(aes_train_node_dir, aes_train_edge_dir, "AES", train_graphs)
    load_folder(aes_test_node_dir, aes_test_edge_dir, "AES", test_graphs)
                
    print(f"Successfully constructed {len(train_graphs)} Train graphs and {len(test_graphs)} Test graphs in total!")
    
    if len(train_graphs) == 0 or len(test_graphs) == 0:
        print("ERROR: Could not find datasets. Make sure all folders are unzipped.")
        sys.exit()
        
    train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=8, shuffle=False)
    
    gcn_induct = train_inductive(train_loader, test_loader, TrojanGCN, epochs=100)
    gat_induct = train_inductive(train_loader, test_loader, TrojanGAT, epochs=100)

    print("\nSaving Combined Models...")
    torch.save(gcn_induct.state_dict(), "combined_gcn_model.pth")
    torch.save(gat_induct.state_dict(), "combined_gat_model.pth")
    print("Saved the ultimate combined .pth models successfully!")
