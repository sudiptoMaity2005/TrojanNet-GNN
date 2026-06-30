# ==============================================================================
# COPY THIS ENTIRE SCRIPT INTO A GOOGLE COLAB NOTEBOOK CELL AND RUN IT
# FOR THE PHASE 7 GNN4GATE EVALUATION (PURE STRUCTURAL GRAPH)
# ==============================================================================

# Install PyTorch Geometric before importing
!pip install torch_geometric

import os
import glob
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import dropout_edge
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

# ---------------------------------------------------------
# 1. Dataset Loading
# ---------------------------------------------------------
from sklearn.decomposition import PCA

def load_gnn4gate_datasets(data_dir, n_components=5, batch_size=4):
    print(f"Loading GNN4Gate graphs from: {data_dir}")
    
    sizes = ["small", "medium", "Large"]
    train_graphs = []
    test_graphs = []
    
    for size in sizes:
        for pt_file in glob.glob(os.path.join(data_dir, size, "*.pt")):
            try:
                data = torch.load(pt_file, weights_only=False)
                if data.x.shape[1] != 23:
                    continue
                    
                # Feature Normalization (Z-score Standardization)
                x_mean = data.x.mean(dim=0, keepdim=True)
                x_std = data.x.std(dim=0, unbiased=False, keepdim=True) + 1e-8
                data.x = (data.x - x_mean) / x_std
                
                # Attach circuit name so we can identify it later!
                data.circuit_name = os.path.basename(pt_file).replace('.pt', '')
                
                # Split 80/20 train/test deterministically
                if hash(pt_file) % 5 == 0:
                    test_graphs.append(data)
                else:
                    train_graphs.append(data)
            except Exception as e:
                pass

    print(f"Loaded {len(train_graphs)} training graphs.")
    print(f"Loaded {len(test_graphs)} testing graphs.")
    
    # --- NEW: PCA Feature Reduction ---
    print(f"\n--- Applying PCA to reduce 23 features down to {n_components} ---")
    all_train_x = torch.cat([data.x for data in train_graphs], dim=0).numpy()
    
    pca = PCA(n_components=n_components)
    pca.fit(all_train_x)
    print(f"PCA Explained Variance Ratio: {sum(pca.explained_variance_ratio_):.4f} (This is how much original variance was kept)")
    
    for data in train_graphs:
        data.x = torch.tensor(pca.transform(data.x.numpy()), dtype=torch.float)
    for data in test_graphs:
        data.x = torch.tensor(pca.transform(data.x.numpy()), dtype=torch.float)
    
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)
    
    return train_loader, test_loader

# ---------------------------------------------------------
# 2. Focal Loss for Extreme Class Imbalance
# ---------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Expected to be a tensor of weights [weight_normal, weight_trojan]
        self.gamma = gamma

    def forward(self, inputs, targets):
        # inputs: [N, 2] logits, targets: [N] class indices
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        focal_loss = (1 - pt) ** self.gamma * BCE_loss

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()

from torch_geometric.nn import SAGEConv, JumpingKnowledge

# ---------------------------------------------------------
# 3. Hybrid GNN Architecture (AugmentedSAGE + 23 Features)
# ---------------------------------------------------------
class GNN4GateHybrid(torch.nn.Module):
    def __init__(self, in_channels=23, hidden_channels=64, out_channels=2):
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
        # Feature Noise Augmentation
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
        h1 = h1 + x # Residual
        
        h2 = self.conv2(h1, edge_index)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=0.2, training=self.training)
        h2 = h2 + h1 # Residual
        
        h3 = self.conv3(h2, edge_index)
        h3 = self.bn3(h3)
        h3 = F.relu(h3)
        h3 = F.dropout(h3, p=0.2, training=self.training)
        h3 = h3 + h2 # Residual
        
        h_jk = self.jk([h1, h2, h3])
        
        out = self.fc1(h_jk)
        out = F.relu(out)
        out = self.fc2(out)
        return out

# ---------------------------------------------------------
# 4. Training and Evaluation Loop
# ---------------------------------------------------------
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index) # DropEdge is disabled automatically by model.eval()
        probs = F.softmax(out, dim=1)[:, 1]
        preds = out.argmax(dim=1)
        
        all_preds.append(preds.cpu())
        all_labels.append(data.y.cpu())
        all_probs.append(probs.cpu())
        
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
        
    return {
        'Precision': precision_score(all_labels, all_preds, zero_division=0),
        'Recall': recall_score(all_labels, all_preds, zero_division=0),
        'F1': f1_score(all_labels, all_preds, zero_division=0),
        'Accuracy': accuracy_score(all_labels, all_preds),
        'AUC': auc
    }

# ---------------------------------------------------------
# 5. Main Execution
# ---------------------------------------------------------
if __name__ == '__main__':
    # Make sure you upload and unzip gnn4gate_datasets.zip first!
    if not os.path.exists("gnn4gate_datasets"):
        print("ERROR: gnn4gate_datasets directory not found! Please upload and unzip it.")
        import sys; sys.exit(1)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader, test_loader = load_gnn4gate_datasets("gnn4gate_datasets", n_components=5, batch_size=4)
    
    if len(train_loader.dataset) == 0:
        print("No training data found. Exiting.")
        import sys; sys.exit(1)
        
    # Model Setup (PCA Features)
    model = GNN4GateHybrid(in_channels=5, hidden_channels=64, out_channels=2).to(device)
    
    # Calculate Class Weights
    total_nodes = 0
    total_ht = 0
    for data in train_loader.dataset:
        total_nodes += data.num_nodes
        total_ht += data.y.sum().item()
    
    weight_ht = (total_nodes - total_ht) / max(1, total_ht)
    
    # We apply a dampening factor to the raw weight because Focal Loss already heavily penalizes the minority class.
    # Without dampening, combining weight=120 with gamma=2.0 would be overwhelmingly aggressive.
    dampened_weight = weight_ht * 0.25 
    class_weights = torch.tensor([1.0, dampened_weight], dtype=torch.float).to(device)
    print(f"\nComputed Dampened Class Weights for Focal Loss: Normal=1.0, Trojan={dampened_weight:.2f}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    
    # Using the new Focal Loss
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    
    # Training Loop
    epochs = 100 # Increased epochs because DropEdge and GATv2 take slightly longer to converge
    print("\n--- Starting Training ---")
    
    import csv
    import pandas as pd
    
    # Open CSV for logging
    with open("pca_training_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Loss", "Test_Precision", "Test_Recall", "Test_F1", "Test_Accuracy", "Test_AUC"])
        
        for epoch in range(1, epochs + 1):
            loss = train(model, train_loader, optimizer, criterion, device)
            if epoch % 10 == 0 or epoch == 1:
                metrics = evaluate(model, test_loader, device)
                print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Test Precision: {metrics['Precision']:.4f} | Test Recall: {metrics['Recall']:.4f} | Test F1: {metrics['F1']:.4f}")
                writer.writerow([epoch, loss, metrics['Precision'], metrics['Recall'], metrics['F1'], metrics['Accuracy'], metrics['AUC']])
            
    print("\n--- Final Evaluation ---")
    final_metrics = evaluate(model, test_loader, device)
    for k, v in final_metrics.items():
        print(f"{k}: {v:.4f}")
        
    # Save final predictions to CSV and print per-circuit metrics
    print("\nExtracting final predictions...")
    model.eval()
    all_preds, all_labels, all_probs, all_circuits = [], [], [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            probs = F.softmax(out, dim=1)[:, 1]
            preds = out.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(data.y.cpu())
            all_probs.append(probs.cpu())
            
            # Since test_loader has batch_size=1, data is exactly 1 graph
            circ_name = data.circuit_name[0] if hasattr(data, 'circuit_name') else "Unknown"
            all_circuits.extend([circ_name] * data.num_nodes)
            
    df_preds = pd.DataFrame({
        "Circuit_Name": all_circuits,
        "True_Label": torch.cat(all_labels).numpy(),
        "Predicted_Label": torch.cat(all_preds).numpy(),
        "Trojan_Probability": torch.cat(all_probs).numpy()
    })
    df_preds.to_csv("pca_final_predictions.csv", index=False)
    print("Saved final predictions to pca_final_predictions.csv!")
    
    # Print a summary per circuit
    print("\n--- Breakdown by Circuit ---")
    for circ in df_preds["Circuit_Name"].unique():
        circ_df = df_preds[df_preds["Circuit_Name"] == circ]
        prec = precision_score(circ_df["True_Label"], circ_df["Predicted_Label"], zero_division=0)
        rec = recall_score(circ_df["True_Label"], circ_df["Predicted_Label"], zero_division=0)
        print(f"{circ}: Precision = {prec:.4f} | Recall = {rec:.4f}")
