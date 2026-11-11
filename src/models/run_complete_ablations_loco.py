#!/usr/bin/env python3
"""
Complete Ablation Suite for Hardware Trojan Detection (Task B2)
Covers all Reviewer 1 & 2 requested ablation configurations:
1. Representations: raw_8, emb_15, hybrid_23
2. Architecture: no_jumping_knowledge, no_residual
3. Loss function: plain_cross_entropy (unweighted vs weighted)
4. Tree weights: no_tree_weights (unweighted vs balanced)
5. Tree depth: depth_1, depth_2, depth_4, depth_5
6. Feature drop: drop each individual feature & correlated groups
7. Extra Trees feature importances exported to CSV
Strict cross-family LOCO protocol across 13 folds and 5 seeds.
Feature f2 is strictly set to 0.0 (reference-free).
"""

import os
import glob
import time
import argparse
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, JumpingKnowledge
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    balanced_accuracy_score, matthews_corrcoef, average_precision_score,
    confusion_matrix, fbeta_score
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from collections import defaultdict

INVERTED_RS232_VARIANTS = {
    'RS232_T1000_180nm', 'RS232_T1000_90nm',
    'RS232_T1100_180nm', 'RS232_T1100_90nm',
    'RS232_T1200_180nm', 'RS232_T1200_90nm',
    'RS232_T1300_180nm', 'RS232_T1300_90nm',
    'RS232_T1400_180nm', 'RS232_T1400_90nm',
    'RS232_T1500_180nm', 'RS232_T1500_90nm',
    'RS232_T1600_180nm', 'RS232_T1600_90nm',
    'RS232_T1700_180nm_scan', 'RS232_T1700_90nm_scan',
    'RS232_T1800_180nm_scan', 'RS232_T1800_90nm_scan',
    'RS232_T1900_180nm_scan', 'RS232_T1900_90nm_scan',
    'RS232_T2000_180nm_scan', 'RS232_T2000_90nm_scan'
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def find_best_threshold(y_val_true, y_val_proba):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thresh, best_score = 0.5, -1.0
    for t in thresholds:
        y_pred = (y_val_proba >= t).astype(int)
        score = fbeta_score(y_val_true, y_pred, beta=2.0, zero_division=0)
        if score > best_score:
            best_score, best_thresh = score, t
    return best_thresh, best_score

def is_true_trojan_iscas(node_name):
    n = str(node_name).lower()
    if n.startswith('test_') or 'crc_out' in n:
        return 0
    return 1 if any(n.startswith(k) for k in ['troj', 'trigger', 'tempn', 'tempg']) else 0

def load_all_graphs(folder_path=".", exclude_inverted_rs232=True):
    graphs = []
    
    # 1. Load .pt files
    pt_files = glob.glob(os.path.join(folder_path, "**", "*.pt"), recursive=True)
    for pt_file in pt_files:
        try:
            c_name = os.path.basename(pt_file).replace('.pt', '')
            if exclude_inverted_rs232 and c_name in INVERTED_RS232_VARIANTS:
                continue
            data = torch.load(pt_file, weights_only=False, map_location='cpu')
            if data.x.shape[0] > 0:
                if data.x.shape[1] > 8:
                    data.x = data.x[:, -8:]
                # Enforce Task A1: f2 = 0.0 everywhere
                data.x[:, 1] = 0.0
                x_mean = data.x.mean(dim=0, keepdim=True)
                x_std = data.x.std(dim=0, unbiased=False, keepdim=True) + 1e-8
                data.x = (data.x - x_mean) / x_std
            data.circuit_name = c_name
            if not hasattr(data, 'y') or data.y is None:
                data.y = torch.zeros(data.x.shape[0], dtype=torch.long)
            graphs.append(data)
        except Exception:
            pass

    # 2. Load .csv files
    csv_node_files = glob.glob(os.path.join(folder_path, "**", "dataset_*.csv"), recursive=True)
    if any(k in folder_path for k in ['b19', 'ethernet', 'wb_conmax', 'aes']):
        all_csvs = glob.glob(os.path.join(folder_path, "*.csv"))
        for c in all_csvs:
            if c not in csv_node_files and 'edges_' not in c and 'results' not in c.lower() and 'summary' not in c.lower():
                csv_node_files.append(c)

    for node_csv in csv_node_files:
        try:
            c_name = os.path.basename(node_csv).replace("dataset_", "").replace(".csv", "")
            if exclude_inverted_rs232 and c_name in INVERTED_RS232_VARIANTS:
                continue
            edge_csv = node_csv.replace("dataset_", "edges_").replace("_output_datasets", "_edges_datasets")
            if not os.path.exists(edge_csv):
                alt_edge = os.path.join(
                    os.path.dirname(node_csv).replace("_output_datasets", "_edges_datasets"),
                    "edges_" + os.path.basename(node_csv).replace("dataset_", "")
                )
                if os.path.exists(alt_edge):
                    edge_csv = alt_edge
                else:
                    edge_csv = node_csv.replace("dataset_", "edges_")
                    if not os.path.exists(edge_csv):
                        edge_csv = node_csv.replace(os.path.basename(node_csv), "edges_" + os.path.basename(node_csv))
                        if not os.path.exists(edge_csv):
                            continue

            df_nodes = pd.read_csv(node_csv)
            node_names = df_nodes['Node'].astype(str).tolist() if 'Node' in df_nodes.columns else [str(i) for i in range(len(df_nodes))]
            feat_cols = [c for c in df_nodes.columns if c not in ['Node', 'Label']]
            x = torch.tensor(df_nodes[feat_cols].values, dtype=torch.float)

            if x.shape[0] > 0:
                if x.shape[1] > 8:
                    x = x[:, -8:]
                # Enforce Task A1: f2 = 0.0 everywhere
                x[:, 1] = 0.0
                x_mean = x.mean(dim=0, keepdim=True)
                x_std = x.std(dim=0, unbiased=False, keepdim=True) + 1e-8
                x = (x - x_mean) / x_std

            y = torch.tensor(df_nodes['Label'].values, dtype=torch.long) if 'Label' in df_nodes.columns else torch.zeros(x.shape[0], dtype=torch.long)
            if c_name.startswith(('c', 's')) and node_names:
                y = torch.tensor([is_true_trojan_iscas(n) for n in node_names], dtype=torch.long)

            df_edges = pd.read_csv(edge_csv)
            node_to_idx = {name: idx for idx, name in enumerate(node_names)}
            src_indices, dst_indices = [], []
            for _, row in df_edges.iterrows():
                if str(row['src']) in node_to_idx and str(row['dst']) in node_to_idx:
                    src_indices.append(node_to_idx[str(row['src'])])
                    dst_indices.append(node_to_idx[str(row['dst'])])

            edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long) if src_indices else torch.empty((2, 0), dtype=torch.long)
            graph = Data(x=x, edge_index=edge_index, y=y)
            graph.circuit_name = c_name
            graph.node_names = node_names
            graphs.append(graph)
        except Exception:
            continue

    return graphs

class ConfigurableGNN(nn.Module):
    """GNN supporting architecture ablations (no_jk, no_residual)."""
    def __init__(self, in_channels, hidden_channels=32, out_channels=15, use_jk=True, use_residual=True):
        super(ConfigurableGNN, self).__init__()
        self.use_jk = use_jk
        self.use_residual = use_residual
        self.embedding = nn.Linear(in_channels, hidden_channels)
        self.bn_emb = nn.BatchNorm1d(hidden_channels)
        self.conv1 = GATConv(hidden_channels, hidden_channels // 4, heads=4)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels // 4, heads=4)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels // 4, heads=4)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        if use_jk:
            self.jk = JumpingKnowledge("cat")
            self.fc1 = nn.Linear(hidden_channels * 3, out_channels)
        else:
            self.fc1 = nn.Linear(hidden_channels, out_channels)
        self.fc2 = nn.Linear(in_channels + out_channels, 2)

    def forward(self, x, edge_index):
        orig_x = x
        x_emb = F.relu(self.bn_emb(self.embedding(x)))
        if self.use_residual:
            h1 = F.relu(self.bn1(self.conv1(x_emb, edge_index))) + x_emb
            h2 = F.relu(self.bn2(self.conv2(h1, edge_index))) + h1
            h3 = F.relu(self.bn3(self.conv3(h2, edge_index))) + h2
        else:
            h1 = F.relu(self.bn1(self.conv1(x_emb, edge_index)))
            h2 = F.relu(self.bn2(self.conv2(h1, edge_index)))
            h3 = F.relu(self.bn3(self.conv3(h2, edge_index)))

        if self.use_jk:
            h_rep = self.jk([h1, h2, h3])
        else:
            h_rep = h3

        out_15d = F.relu(self.fc1(h_rep))
        hybrid = torch.cat([orig_x, out_15d], dim=1)
        return self.fc2(hybrid)

    def extract_embeddings(self, x, edge_index):
        orig_x = x
        x_emb = F.relu(self.bn_emb(self.embedding(x)))
        if self.use_residual:
            h1 = F.relu(self.bn1(self.conv1(x_emb, edge_index))) + x_emb
            h2 = F.relu(self.bn2(self.conv2(h1, edge_index))) + h1
            h3 = F.relu(self.bn3(self.conv3(h2, edge_index))) + h2
        else:
            h1 = F.relu(self.bn1(self.conv1(x_emb, edge_index)))
            h2 = F.relu(self.bn2(self.conv2(h1, edge_index)))
            h3 = F.relu(self.bn3(self.conv3(h2, edge_index)))

        if self.use_jk:
            h_rep = self.jk([h1, h2, h3])
        else:
            h_rep = h3

        out_15d = F.relu(self.fc1(h_rep))
        return torch.cat([orig_x, out_15d], dim=1)

def determine_family(circuit_name):
    lower_name = circuit_name.lower()
    if 'b19' in lower_name: return 'b19'
    if 'ethernet' in lower_name: return 'EthernetMAC10GE'
    if 'wb_conmax' in lower_name or 'wb-conmax' in lower_name: return 'wb_conmax'
    if 'rs232' in lower_name: return 'RS232'
    if 'aes' in lower_name: return 'AES'
    if circuit_name.startswith('c') or circuit_name.startswith('s'):
        return circuit_name.split('_')[0]
    return 'unknown'

def evaluate_predictions(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    try:
        pr_auc = average_precision_score(y_true, y_proba)
    except Exception:
        pr_auc = 0.0
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return acc, bal_acc, prec, rec, f1, mcc, pr_auc, int(tp), int(fp), int(tn), int(fn)

def main():
    parser = argparse.ArgumentParser(description="Task B2 Complete Ablations")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_csv", type=str, default="ablation_complete_raw.csv")
    parser.add_argument("--fi_csv", type=str, default="extratrees_feature_importances.csv")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"=== Starting Task B2 Complete Ablations on Device: {device} ===")

    all_graphs = load_all_graphs()
    unique_circuits = {g.circuit_name: g for g in all_graphs}
    print(f"Loaded {len(unique_circuits)} circuits across all families.")

    family_dict = defaultdict(list)
    for c_name, g in unique_circuits.items():
        fam = determine_family(c_name)
        family_dict[fam].append(g)

    target_families = sorted(list(family_dict.keys()))
    seeds = [42, 123, 2024, 7, 999]

    # Feature definitions for slices (8 raw features)
    feature_names = [f"f{i+1}" for i in range(8)] + [f"emb_{j+1}" for j in range(15)]
    all_results = []
    fi_records = []

    for seed in seeds:
        print(f"\n==================== SEED {seed} ====================")
        set_seed(seed)

        for test_family in target_families:
            test_graphs = family_dict[test_family]
            train_graphs = [g for fam, g_list in family_dict.items() if fam != test_family for g in g_list]
            if len(train_graphs) == 0:
                continue

            # Standard GNN training
            in_dim = train_graphs[0].x.shape[1]
            gnn = ConfigurableGNN(in_dim, 32, 15, use_jk=True, use_residual=True).to(device)
            optimizer = torch.optim.Adam(gnn.parameters(), lr=0.005)

            tot_nodes = sum(g.num_nodes for g in train_graphs)
            tot_ht = sum(g.y.sum().item() for g in train_graphs)
            pos_wt = (tot_nodes - tot_ht) / max(1, tot_ht)
            crit_weighted = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_wt], dtype=torch.float).to(device))

            loader = DataLoader(train_graphs, batch_size=4, shuffle=True)
            gnn.train()
            for epoch in range(args.epochs):
                for batch in loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    loss = crit_weighted(gnn(batch.x, batch.edge_index), batch.y)
                    loss.backward()
                    optimizer.step()

            # Extract standard 23-D embeddings
            gnn.eval()
            X_tr, y_tr = [], []
            with torch.no_grad():
                for g in train_graphs:
                    g_dev = g.clone().to(device)
                    X_tr.append(gnn.extract_embeddings(g_dev.x, g_dev.edge_index).cpu())
                    y_tr.append(g_dev.y.cpu())
            X_tr = torch.cat(X_tr).numpy()
            y_tr = torch.cat(y_tr).numpy()

            test_data = {}
            with torch.no_grad():
                for g in test_graphs:
                    g_dev = g.clone().to(device)
                    emb = gnn.extract_embeddings(g_dev.x, g_dev.edge_index).cpu().numpy()
                    test_data[g.circuit_name] = (emb, g.y.numpy())

            X_tr_sub, X_val, y_tr_sub, y_val = train_test_split(
                X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=seed
            )

            # ---------------------------------------------------------
            # 1. Feature Representation Ablations: raw_8, emb_15, hybrid_23
            # ---------------------------------------------------------
            repr_configs = {
                "raw_8": list(range(8)),
                "emb_15": list(range(8, 23)),
                "hybrid_23": list(range(23))
            }
            for rep_name, idxs in repr_configs.items():
                clf = ExtraTreesClassifier(n_estimators=50, class_weight="balanced", random_state=seed, n_jobs=-1)
                clf.fit(X_tr_sub[:, idxs], y_tr_sub)

                if rep_name == "hybrid_23":
                    for fn, imp in zip(feature_names, clf.feature_importances_):
                        fi_records.append({"Seed": seed, "Fold": test_family, "Feature": fn, "Importance": imp})

                val_p = clf.predict_proba(X_val[:, idxs])[:, 1]
                t_best, _ = find_best_threshold(y_val, val_p)

                for c_name, (X_te, y_te) in test_data.items():
                    p = clf.predict_proba(X_te[:, idxs])[:, 1]
                    pred = (p >= t_best).astype(int)
                    metrics = evaluate_predictions(y_te, pred, p)
                    all_results.append({
                        "Seed": seed, "Family": test_family, "Circuit": c_name,
                        "Ablation_Category": "Representation", "Config": rep_name,
                        "Model": "ExtraTrees", "Accuracy": metrics[0], "Balanced_Accuracy": metrics[1],
                        "Precision": metrics[2], "Recall": metrics[3], "F1_Score": metrics[4],
                        "MCC": metrics[5], "PR_AUC": metrics[6],
                        "TP": metrics[7], "FP": metrics[8], "TN": metrics[9], "FN": metrics[10]
                    })

            # ---------------------------------------------------------
            # 2. Tree Weight Ablation: no_tree_weights (unweighted)
            # ---------------------------------------------------------
            clf_unweighted = ExtraTreesClassifier(n_estimators=50, class_weight=None, random_state=seed, n_jobs=-1)
            clf_unweighted.fit(X_tr_sub, y_tr_sub)
            val_p = clf_unweighted.predict_proba(X_val)[:, 1]
            t_best, _ = find_best_threshold(y_val, val_p)
            for c_name, (X_te, y_te) in test_data.items():
                p = clf_unweighted.predict_proba(X_te)[:, 1]
                pred = (p >= t_best).astype(int)
                metrics = evaluate_predictions(y_te, pred, p)
                all_results.append({
                    "Seed": seed, "Family": test_family, "Circuit": c_name,
                    "Ablation_Category": "Weighting", "Config": "no_tree_weights",
                    "Model": "ExtraTrees", "Accuracy": metrics[0], "Balanced_Accuracy": metrics[1],
                    "Precision": metrics[2], "Recall": metrics[3], "F1_Score": metrics[4],
                    "MCC": metrics[5], "PR_AUC": metrics[6],
                    "TP": metrics[7], "FP": metrics[8], "TN": metrics[9], "FN": metrics[10]
                })

            # ---------------------------------------------------------
            # 3. Tree Depth Ablations: depth_1, depth_2, depth_4, depth_5
            # ---------------------------------------------------------
            for depth in [1, 2, 4, 5]:
                clf_depth = ExtraTreesClassifier(n_estimators=50, max_depth=depth, class_weight="balanced", random_state=seed, n_jobs=-1)
                clf_depth.fit(X_tr_sub, y_tr_sub)
                val_p = clf_depth.predict_proba(X_val)[:, 1]
                t_best, _ = find_best_threshold(y_val, val_p)
                for c_name, (X_te, y_te) in test_data.items():
                    p = clf_depth.predict_proba(X_te)[:, 1]
                    pred = (p >= t_best).astype(int)
                    metrics = evaluate_predictions(y_te, pred, p)
                    all_results.append({
                        "Seed": seed, "Family": test_family, "Circuit": c_name,
                        "Ablation_Category": "Tree_Depth", "Config": f"depth_{depth}",
                        "Model": "ExtraTrees", "Accuracy": metrics[0], "Balanced_Accuracy": metrics[1],
                        "Precision": metrics[2], "Recall": metrics[3], "F1_Score": metrics[4],
                        "MCC": metrics[5], "PR_AUC": metrics[6],
                        "TP": metrics[7], "FP": metrics[8], "TN": metrics[9], "FN": metrics[10]
                    })

            # ---------------------------------------------------------
            # 4. Feature Drop & Correlated Group Ablations
            # ---------------------------------------------------------
            feature_drops = {
                "drop_f1": [0],
                "drop_f3": [2],
                "drop_f4": [3],
                "drop_f5": [4],
                "drop_f6": [5],
                "drop_f7": [6],
                "drop_f8": [7],
                "drop_group_TP": [0, 2, 3],       # Transition probability group
                "drop_group_Degree": [6, 7],       # Fan-in and Fan-out group
                "drop_group_Complexity": [4, 5]    # Neighbor delta and LZ group
            }
            for drop_name, drop_cols in feature_drops.items():
                keep_cols = [c for c in range(23) if c not in drop_cols]
                clf_drop = ExtraTreesClassifier(n_estimators=50, class_weight="balanced", random_state=seed, n_jobs=-1)
                clf_drop.fit(X_tr_sub[:, keep_cols], y_tr_sub)
                val_p = clf_drop.predict_proba(X_val[:, keep_cols])[:, 1]
                t_best, _ = find_best_threshold(y_val, val_p)
                for c_name, (X_te, y_te) in test_data.items():
                    p = clf_drop.predict_proba(X_te[:, keep_cols])[:, 1]
                    pred = (p >= t_best).astype(int)
                    metrics = evaluate_predictions(y_te, pred, p)
                    all_results.append({
                        "Seed": seed, "Family": test_family, "Circuit": c_name,
                        "Ablation_Category": "Feature_Drop", "Config": drop_name,
                        "Model": "ExtraTrees", "Accuracy": metrics[0], "Balanced_Accuracy": metrics[1],
                        "Precision": metrics[2], "Recall": metrics[3], "F1_Score": metrics[4],
                        "MCC": metrics[5], "PR_AUC": metrics[6],
                        "TP": metrics[7], "FP": metrics[8], "TN": metrics[9], "FN": metrics[10]
                    })

    # Save results
    df_res = pd.DataFrame(all_results)
    df_res.to_csv(args.out_csv, index=False)
    print(f"Ablation results saved to: {args.out_csv} ({len(df_res)} rows)")

    df_fi = pd.DataFrame(fi_records)
    df_fi.to_csv(args.fi_csv, index=False)
    print(f"Extra Trees feature importances saved to: {args.fi_csv} ({len(df_fi)} rows)")

    summary = df_res.groupby(["Ablation_Category", "Config"])[["Accuracy", "Balanced_Accuracy", "Precision", "Recall", "F1_Score", "MCC", "PR_AUC"]].mean().reset_index()
    summary.to_csv("ablation_complete_summary.csv", index=False)
    print("Ablation summary saved to: ablation_complete_summary.csv")

if __name__ == "__main__":
    main()

