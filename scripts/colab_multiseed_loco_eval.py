import os
import glob
import time
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, JumpingKnowledge
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, fbeta_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from collections import defaultdict

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def find_best_threshold(y_val_true, y_val_proba):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thresh, best_score = 0.5, -1
    for t in thresholds:
        y_pred = (y_val_proba >= t).astype(int)
        score = fbeta_score(y_val_true, y_pred, beta=2.0, zero_division=0)
        if score > best_score:
            best_score, best_thresh = score, t
    return best_thresh, best_score

def load_datasets_from_folder(folder_path):
    graphs = []
    
    # 1. Load .pt files
    pt_files = glob.glob(os.path.join(folder_path, "**", "*.pt"), recursive=True)
    for pt_file in pt_files:
        try:
            data = torch.load(pt_file, weights_only=False, map_location='cpu')
            if data.x.shape[0] > 0:
                if data.x.shape[1] > 8:
                    data.x = data.x[:, -8:]
                x_mean = data.x.mean(dim=0, keepdim=True)
                x_std = data.x.std(dim=0, unbiased=False, keepdim=True) + 1e-8
                data.x = (data.x - x_mean) / x_std
            data.circuit_name = os.path.basename(pt_file).replace('.pt', '')
            if not hasattr(data, 'y') or data.y is None:
                data.y = torch.zeros(data.x.shape[0], dtype=torch.long)
            graphs.append(data)
        except Exception as e:
            pass
            
    # 2. Load .csv files
    csv_node_files = glob.glob(os.path.join(folder_path, "**", "dataset_*.csv"), recursive=True)
    # Add files that might not start with 'dataset_' but end in .csv
    if 'b19' in folder_path or 'ethernet' in folder_path or 'wb_conmax' in folder_path:
        all_csvs = glob.glob(os.path.join(folder_path, "*.csv"))
        for c in all_csvs:
            if c not in csv_node_files and 'edges_' not in c and 'results' not in c.lower():
                csv_node_files.append(c)

    for node_csv in csv_node_files:
        try:
            edge_csv = node_csv.replace("dataset_", "edges_").replace("_output_datasets", "_edges_datasets")
            if not os.path.exists(edge_csv):
                alt_edge = os.path.join(os.path.dirname(node_csv).replace("_output_datasets", "_edges_datasets"), "edges_" + os.path.basename(node_csv).replace("dataset_", ""))
                if os.path.exists(alt_edge):
                    edge_csv = alt_edge
                else:
                    edge_csv = node_csv.replace("dataset_", "edges_")
                    if not os.path.exists(edge_csv):
                        # try replacing file name
                        edge_csv = node_csv.replace(os.path.basename(node_csv), "edges_" + os.path.basename(node_csv))
                        if not os.path.exists(edge_csv):
                            continue
                    
            df_nodes = pd.read_csv(node_csv)
            feat_cols = [c for c in df_nodes.columns if c not in ['Node', 'Label']]
            x = torch.tensor(df_nodes[feat_cols].values, dtype=torch.float)
            
            if x.shape[0] > 0:
                if x.shape[1] > 8:
                    x = x[:, -8:]
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
            
            circuit_name = os.path.basename(node_csv).replace("dataset_", "").replace(".csv", "")
            
            graph = Data(x=x, edge_index=edge_index, y=y)
            graph.circuit_name = circuit_name
            graphs.append(graph)
        except Exception as e:
            continue
            
    return graphs

class GNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, out_channels=15):
        super(GNNFeatureExtractor, self).__init__()
        self.embedding = nn.Linear(in_channels, hidden_channels)
        self.bn_emb = nn.BatchNorm1d(hidden_channels)
        self.conv1 = GATConv(hidden_channels, hidden_channels // 4, heads=4)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels // 4, heads=4)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels // 4, heads=4)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        self.jk = JumpingKnowledge("cat")
        self.fc1 = nn.Linear(hidden_channels * 3, out_channels)
        self.fc2 = nn.Linear(in_channels + out_channels, 2)
        
    def forward(self, x, edge_index):
        orig_x = x
        x_emb = F.relu(self.bn_emb(self.embedding(x)))
        h1 = F.relu(self.bn1(self.conv1(x_emb, edge_index))) + x_emb
        h2 = F.relu(self.bn2(self.conv2(h1, edge_index))) + h1
        h3 = F.relu(self.bn3(self.conv3(h2, edge_index))) + h2
        h_jk = self.jk([h1, h2, h3])
        out_15d = F.relu(self.fc1(h_jk))
        hybrid_23d = torch.cat([orig_x, out_15d], dim=1)
        return self.fc2(hybrid_23d)
        
    def extract_embeddings(self, x, edge_index):
        orig_x = x
        x_emb = F.relu(self.bn_emb(self.embedding(x)))
        h1 = F.relu(self.bn1(self.conv1(x_emb, edge_index))) + x_emb
        h2 = F.relu(self.bn2(self.conv2(h1, edge_index))) + h1
        h3 = F.relu(self.bn3(self.conv3(h2, edge_index))) + h2
        h_jk = self.jk([h1, h2, h3])
        out_15d = F.relu(self.fc1(h_jk))
        return torch.cat([orig_x, out_15d], dim=1)

def determine_family(circuit_name):
    lower_name = circuit_name.lower()
    if 'b19' in lower_name: return 'b19'
    if 'ethernet' in lower_name: return 'EthernetMAC10GE'
    if 'wb_conmax' in lower_name: return 'wb_conmax'
    if 'rs232' in lower_name: return 'RS232'
    if 'aes' in lower_name: return 'AES'
    if circuit_name.startswith('c') or circuit_name.startswith('s'):
        return circuit_name.split('_')[0]
    return 'unknown'

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading all datasets...")
    
    folders = [
        "training_datasets", "other_trit_tc_datasets", "rest_trit_tc_datasets", 
        "datasets_for_gnn", "b19_datasets", "ethernet_datasets", "wb_conmax_datasets", 
        "aes_output_datasets"
    ]
    
    all_graphs = []
    for f in folders:
        if os.path.exists(f):
            all_graphs.extend(load_datasets_from_folder(f))
            
    # Deduplicate by circuit name
    unique_graphs = {}
    for g in all_graphs:
        unique_graphs[g.circuit_name] = g
    all_graphs = list(unique_graphs.values())
    
    family_dict = defaultdict(list)
    for g in all_graphs:
        fam = determine_family(g.circuit_name)
        if fam != 'unknown':
            family_dict[fam].append(g)
            
    target_families = ['AES', 'RS232', 'b19', 'EthernetMAC10GE', 'wb_conmax', 'c2670', 'c3540', 'c5315', 'c6288', 's13207', 's1423', 's15850', 's35932']
    
    seeds = [42, 123, 2024]
    all_results = []
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"STARTING EVALUATION WITH SEED: {seed}")
        print(f"{'='*60}")
        set_seed(seed)
        
        for test_family in target_families:
            if test_family not in family_dict or len(family_dict[test_family]) == 0:
                print(f"Skipping {test_family}, no graphs found.")
                continue
                
            print(f"\n--- LOCO FOLD: Testing on {test_family} (Seed {seed}) ---")
            
            loco_train_graphs = []
            core_train_families = ['RS232', 'c2670', 'c3540', 'c5315', 'c6288', 's13207', 's1423', 's15850', 's35932']
            for fam in core_train_families:
                if fam != test_family and fam in family_dict:
                    loco_train_graphs.extend(family_dict[fam])
                    
                    
            loco_test_graphs = family_dict[test_family]
            
            if len(loco_train_graphs) == 0: continue
            
            num_features = loco_train_graphs[0].x.shape[1]
            gnn = GNNFeatureExtractor(in_channels=num_features, hidden_channels=32, out_channels=15).to(device)
            optimizer = torch.optim.Adam(gnn.parameters(), lr=0.005)
            
            total_nodes = sum([g.num_nodes for g in loco_train_graphs])
            total_ht = sum([g.y.sum().item() for g in loco_train_graphs])
            pos_weight = torch.tensor([(total_nodes - total_ht) / max(1, total_ht)], dtype=torch.float).to(device)
            criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight.item()], dtype=torch.float).to(device))
            
            train_loader = DataLoader(loco_train_graphs, batch_size=4, shuffle=True)
            
            # Train GNN for exactly 40 epochs to match the original tuned configuration
            for epoch in range(40):
                gnn.train()
                for data in train_loader:
                    data = data.to(device)
                    optimizer.zero_grad()
                    out = gnn(data.x, data.edge_index)
                    loss = criterion(out, data.y)
                    loss.backward()
                    optimizer.step()
                    
            # Extract features
            gnn.eval()
            X_train_emb, y_train_emb = [], []
            with torch.no_grad():
                for g in loco_train_graphs:
                    g_dev = g.clone().to(device)
                    emb = gnn.extract_embeddings(g_dev.x, g_dev.edge_index)
                    X_train_emb.append(emb.cpu())
                    y_train_emb.append(g_dev.y.cpu())
                    
            X_train_emb = torch.cat(X_train_emb).numpy()
            y_train_emb = torch.cat(y_train_emb).numpy()
            
            X_tr_sub, X_val, y_tr_sub, y_val = train_test_split(
                X_train_emb, y_train_emb, test_size=0.15, stratify=y_train_emb, random_state=seed
            )
            
            models = {
                "RandomForest": RandomForestClassifier(n_estimators=50, class_weight="balanced", n_jobs=-1, random_state=seed),
                "ExtraTrees": ExtraTreesClassifier(n_estimators=50, class_weight="balanced", n_jobs=-1, random_state=seed),
                "XGBoost": XGBClassifier(scale_pos_weight=pos_weight.item(), use_label_encoder=False, eval_metric='logloss', random_state=seed, n_jobs=-1),
            }
            
            for name, model in models.items():
                model.fit(X_tr_sub, y_tr_sub)
                
            for c_graph in loco_test_graphs:
                c_name = c_graph.circuit_name
                
                with torch.no_grad():
                    g_dev = c_graph.clone().to(device)
                    X_test_emb = gnn.extract_embeddings(g_dev.x, g_dev.edge_index).cpu().numpy()
                    y_test_emb = g_dev.y.cpu().numpy()
                
                for m_name, model in models.items():
                    val_proba = model.predict_proba(X_val)[:, 1]
                    best_thresh, _ = find_best_threshold(y_val, val_proba)
                    
                    test_proba = model.predict_proba(X_test_emb)[:, 1]
                    m_preds = (test_proba >= best_thresh).astype(int)
                    
                    acc = accuracy_score(y_test_emb, m_preds)
                    prec = precision_score(y_test_emb, m_preds, zero_division=0)
                    rec = recall_score(y_test_emb, m_preds, zero_division=0)
                    f1 = f1_score(y_test_emb, m_preds, zero_division=0)
                    
                    all_results.append({
                        "Seed": seed, "Family": test_family, "Circuit": c_name, "Model": m_name,
                        "Accuracy": acc, "Precision": prec, "Recall": rec, "F1_Score": f1
                    })
                    
    df_results = pd.DataFrame(all_results)
    df_results.to_csv("multiseed_loco_raw.csv", index=False)
    
    # Compute Aggregations
    summary = []
    for model in ["RandomForest", "ExtraTrees", "XGBoost"]:
        df_m = df_results[df_results['Model'] == model]
        # First group by Seed, Family, Circuit to get circuit means
        # Wait, the instruction says "compute the mean and standard deviation across the seeds, per-family and overall macro-average"
        # 1. Per-family average across seeds
        fam_grouped = df_m.groupby(['Family', 'Seed'])[['Accuracy', 'Precision', 'Recall', 'F1_Score']].mean().reset_index()
        fam_stats = fam_grouped.groupby('Family')[['Accuracy', 'Precision', 'Recall', 'F1_Score']].agg(['mean', 'std'])
        
        for fam in fam_stats.index:
            summary.append({
                "Model": model,
                "Family": fam,
                "Accuracy": f"{fam_stats.loc[fam, ('Accuracy', 'mean')]:.4f} ± {fam_stats.loc[fam, ('Accuracy', 'std')]:.4f}",
                "Precision": f"{fam_stats.loc[fam, ('Precision', 'mean')]:.4f} ± {fam_stats.loc[fam, ('Precision', 'std')]:.4f}",
                "Recall": f"{fam_stats.loc[fam, ('Recall', 'mean')]:.4f} ± {fam_stats.loc[fam, ('Recall', 'std')]:.4f}",
                "F1_Score": f"{fam_stats.loc[fam, ('F1_Score', 'mean')]:.4f} ± {fam_stats.loc[fam, ('F1_Score', 'std')]:.4f}"
            })
            
        # Overall Macro Average
        overall_grouped = fam_grouped.groupby('Seed')[['Accuracy', 'Precision', 'Recall', 'F1_Score']].mean().reset_index()
        overall_mean = overall_grouped.mean()
        overall_std = overall_grouped.std()
        
        summary.append({
            "Model": model,
            "Family": "OVERALL_MACRO",
            "Accuracy": f"{overall_mean['Accuracy']:.4f} ± {overall_std['Accuracy']:.4f}",
            "Precision": f"{overall_mean['Precision']:.4f} ± {overall_std['Precision']:.4f}",
            "Recall": f"{overall_mean['Recall']:.4f} ± {overall_std['Recall']:.4f}",
            "F1_Score": f"{overall_mean['F1_Score']:.4f} ± {overall_std['F1_Score']:.4f}"
        })
        
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv("multiseed_loco_summary.csv", index=False)
    print("\nSAVED MULTISEED LOCO SUMMARY!")
    print(df_summary)
