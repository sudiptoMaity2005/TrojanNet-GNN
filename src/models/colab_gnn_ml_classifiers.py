# ==============================================================================
# COPY THIS ENTIRE SCRIPT INTO A GOOGLE COLAB NOTEBOOK CELL AND RUN IT
# FOR THE PHASE 7 GNN4GATE EVALUATION (PURE STRUCTURAL GRAPH)
# ==============================================================================

# Install PyTorch Geometric and Boosting libraries before importing
!pip install torch_geometric xgboost catboost

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
def load_gnn4gate_datasets(data_dir, batch_size=4):
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
    
    train_loader, test_loader = load_gnn4gate_datasets("gnn4gate_datasets", batch_size=4)
    
    if len(train_loader.dataset) == 0:
        print("No training data found. Exiting.")
        import sys; sys.exit(1)
        
    # Model Setup (Using the successful AugmentedSAGE architecture adapted for Hybrid Features)
    model = GNN4GateHybrid(in_channels=23, hidden_channels=64, out_channels=2).to(device)
    
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
    with open("training_log.csv", "w", newline="") as f:
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
    df_preds.to_csv("final_predictions.csv", index=False)
    print("Saved final predictions to final_predictions.csv!")
    
    # Print a summary per circuit
    print("\n--- Breakdown by Circuit (GNN) ---")
    for circ in df_preds["Circuit_Name"].unique():
        circ_df = df_preds[df_preds["Circuit_Name"] == circ]
        prec = precision_score(circ_df["True_Label"], circ_df["Predicted_Label"], zero_division=0)
        rec = recall_score(circ_df["True_Label"], circ_df["Predicted_Label"], zero_division=0)
        print(f"{circ}: Precision = {prec:.4f} | Recall = {rec:.4f}")

    # =========================================================
    # 6. Representation Learning: Train Traditional ML (SVM/RF)
    # =========================================================
    print("\n\n========================================================")
    print("PHASE 2: EXTRACTING EMBEDDINGS FOR TRADITIONAL ML MODELS")
    print("========================================================")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    
    model.eval()
    
    print("Extracting 32-Dimensional embeddings from the GNN...")
    
    # 1. Extract Training Embeddings
    X_train_emb = []
    y_train_emb = []
    with torch.no_grad():
        for data in train_loader:
            data = data.to(device)
            emb = model.extract_embeddings(data.x, data.edge_index)
            X_train_emb.append(emb.cpu())
            y_train_emb.append(data.y.cpu())
            
    X_train_emb = torch.cat(X_train_emb).numpy()
    y_train_emb = torch.cat(y_train_emb).numpy()
    
    # 2. Extract Testing Embeddings
    X_test_emb = []
    y_test_emb = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            emb = model.extract_embeddings(data.x, data.edge_index)
            X_test_emb.append(emb.cpu())
            y_test_emb.append(data.y.cpu())
            
    X_test_emb = torch.cat(X_test_emb).numpy()
    y_test_emb = torch.cat(y_test_emb).numpy()
    
    print(f"Extracted Training Matrix Shape: {X_train_emb.shape}")
    print(f"Extracted Testing Matrix Shape: {X_test_emb.shape}")
    
    ml_results = []
    
    # 3. Train Random Forest
    print("\n--- Training Random Forest Classifier on GNN Embeddings ---")
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train_emb, y_train_emb)
    rf_preds = rf.predict(X_test_emb)
    
    rf_prec = precision_score(y_test_emb, rf_preds, zero_division=0)
    rf_rec = recall_score(y_test_emb, rf_preds, zero_division=0)
    rf_f1 = f1_score(y_test_emb, rf_preds, zero_division=0)
    print(f"Random Forest Precision: {rf_prec:.4f}")
    print(f"Random Forest Recall:    {rf_rec:.4f}")
    print(f"Random Forest F1 Score:  {rf_f1:.4f}")
    ml_results.append({"Model": "Random Forest", "Precision": rf_prec, "Recall": rf_rec, "F1_Score": rf_f1})
    
    # 4. Train Support Vector Machine
    print("\n--- Training Support Vector Machine (SVM) on GNN Embeddings ---")
    svm = SVC(kernel='rbf', class_weight="balanced", random_state=42)
    svm.fit(X_train_emb, y_train_emb)
    svm_preds = svm.predict(X_test_emb)
    
    svm_prec = precision_score(y_test_emb, svm_preds, zero_division=0)
    svm_rec = recall_score(y_test_emb, svm_preds, zero_division=0)
    svm_f1 = f1_score(y_test_emb, svm_preds, zero_division=0)
    print(f"SVM Precision: {svm_prec:.4f}")
    print(f"SVM Recall:    {svm_rec:.4f}")
    print(f"SVM F1 Score:  {svm_f1:.4f}")
    ml_results.append({"Model": "SVM (RBF Kernel)", "Precision": svm_prec, "Recall": svm_rec, "F1_Score": svm_f1})
    
    # 5. Export Embeddings Dataset to CSV
    print("\nExporting GNN features to gnn_extracted_features.csv...")
    feature_cols = [f"emb_{i}" for i in range(32)]
    df_train = pd.DataFrame(X_train_emb, columns=feature_cols)
    df_train['Label'] = y_train_emb
    df_train['Split'] = 'Train'
    
    df_test = pd.DataFrame(X_test_emb, columns=feature_cols)
    df_test['Label'] = y_test_emb
    df_test['Split'] = 'Test'
    
    df_all_features = pd.concat([df_train, df_test], ignore_index=True)
    df_all_features.to_csv("gnn_extracted_features.csv", index=False)
    
    # =========================================================
    # 7. Additional Advanced ML Models (Instructor Requested)
    # =========================================================
    print("\n\n========================================================")
    print("PHASE 3: ADVANCED ENSEMBLE MODELS (ExtraTrees, XGB, CatBoost)")
    print("========================================================")
    
    import numpy as np
    from sklearn.ensemble import ExtraTreesClassifier
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    
    # 5. Train Extra Trees
    print("\n--- Training Extra Trees Classifier ---")
    et = ExtraTreesClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
    et.fit(X_train_emb, y_train_emb)
    et_preds = et.predict(X_test_emb)
    
    et_prec = precision_score(y_test_emb, et_preds, zero_division=0)
    et_rec = recall_score(y_test_emb, et_preds, zero_division=0)
    et_f1 = f1_score(y_test_emb, et_preds, zero_division=0)
    print(f"Extra Trees Precision: {et_prec:.4f}")
    print(f"Extra Trees Recall:    {et_rec:.4f}")
    print(f"Extra Trees F1 Score:  {et_f1:.4f}")
    ml_results.append({"Model": "Extra Trees", "Precision": et_prec, "Recall": et_rec, "F1_Score": et_f1})

    # 6. Train XGBoost
    print("\n--- Training XGBoost Classifier ---")
    ratio = float(np.sum(y_train_emb == 0)) / max(1, np.sum(y_train_emb == 1))
    xgb = XGBClassifier(scale_pos_weight=ratio, random_state=42, n_jobs=-1, eval_metric='logloss')
    xgb.fit(X_train_emb, y_train_emb)
    xgb_preds = xgb.predict(X_test_emb)
    
    xgb_prec = precision_score(y_test_emb, xgb_preds, zero_division=0)
    xgb_rec = recall_score(y_test_emb, xgb_preds, zero_division=0)
    xgb_f1 = f1_score(y_test_emb, xgb_preds, zero_division=0)
    print(f"XGBoost Precision: {xgb_prec:.4f}")
    print(f"XGBoost Recall:    {xgb_rec:.4f}")
    print(f"XGBoost F1 Score:  {xgb_f1:.4f}")
    ml_results.append({"Model": "XGBoost", "Precision": xgb_prec, "Recall": xgb_rec, "F1_Score": xgb_f1})

    # 7. Train CatBoost
    print("\n--- Training CatBoost Classifier ---")
    cb = CatBoostClassifier(auto_class_weights='Balanced', random_state=42, verbose=0, thread_count=-1)
    cb.fit(X_train_emb, y_train_emb)
    cb_preds = cb.predict(X_test_emb)
    
    cb_prec = precision_score(y_test_emb, cb_preds, zero_division=0)
    cb_rec = recall_score(y_test_emb, cb_preds, zero_division=0)
    cb_f1 = f1_score(y_test_emb, cb_preds, zero_division=0)
    print(f"CatBoost Precision: {cb_prec:.4f}")
    print(f"CatBoost Recall:    {cb_rec:.4f}")
    print(f"CatBoost F1 Score:  {cb_f1:.4f}")
    ml_results.append({"Model": "CatBoost", "Precision": cb_prec, "Recall": cb_rec, "F1_Score": cb_f1})

    # Final Export of Metrics
    df_ml_results = pd.DataFrame(ml_results)
    df_ml_results.to_csv("ml_models_evaluation_results.csv", index=False)
    print("\nSaved all ML metrics to ml_models_evaluation_results.csv!")
    print("Export Complete! You now have the mathematical representation of the circuits and all metrics in CSV format.")
