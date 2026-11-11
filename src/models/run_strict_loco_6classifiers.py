#!/usr/bin/env python3
"""
run_strict_loco_6classifiers.py
==============================================================================
Strict Cross-Family Leave-One-Circuit-Out (LOCO) Evaluation for all 6 Classifiers
Seeds: 42, 123, 2024, 7, 999
Target Families: 13 benchmark families

Classifiers:
  1. GNN (solely): GNN's own classification head (argmax)
  2. SVM: Linear kernel with balanced class weights (full_23)
  3. XGBoost: scale_pos_weight (full_23)
  4. CatBoost: balanced class weights (full_23)
  5. Random Forest: 50 trees, no depth limit, balanced (full_23)
  6. Extra Trees: 50 trees, no depth limit, balanced (full_23)

Trojan-Free Test Circuits:
  - RS232_normal (46 gates, 0 Trojans)
  - wb_conmax-T300 (7,608 gates, 0 Trojans)
  - AES_normal (2,973 gates, 0 Trojans)

Data Preprocessing / Task A1 Resolution:
  - Excludes the 22 inverted RS232 netlists (T1000-T2000 synthesized variants)
    caused by cell-name RTL-to-gate diffing artifacts.
  - Retains the 15 clean, verified RS232 circuits (14 Trojans + RS232_normal).
  - Can optionally include legacy 107 circuits via --include_inverted_rs232.

Outputs:
  - One CSV per classifier: strict_loco_<model>.csv
  - Master raw CSV: strict_loco_all_6classifiers_raw.csv
  - Statistical summary CSV: strict_loco_summary.csv
  - Per-gate scores: gate_scores/gate_scores_seed_{seed}_{model}_{circuit}.csv
==============================================================================
"""

import os
import glob
import time
import random
import argparse
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, JumpingKnowledge

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, average_precision_score, confusion_matrix,
    fbeta_score
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not installed. Run `pip install catboost` to enable CatBoost.")

# Set seed for reproducible results
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Optimal threshold finder on validation split using F2 score
def find_best_threshold(y_val_true, y_val_proba):
    if len(np.unique(y_val_true)) < 2:
        return 0.5, 0.0
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thresh, best_score = 0.5, -1.0
    for t in thresholds:
        y_pred = (y_val_proba >= t).astype(int)
        score = fbeta_score(y_val_true, y_pred, beta=2.0, zero_division=0)
        if score > best_score:
            best_score, best_thresh = score, t
    return best_thresh, best_score

# 22 inverted RS232 variants identified in Task A1
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

def is_true_trojan_iscas(node_name):
    n = str(node_name).lower()
    if n.startswith('test_') or 'crc_out' in n:
        return 0
    return 1 if any(n.startswith(k) for k in ['troj', 'trigger', 'tempn', 'tempg']) else 0

def load_datasets_from_folder(folder_path, exclude_inverted_rs232=True):
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
                # Task A1: Strictly set f2 = 0.0 across all circuits to eliminate golden-twin leakage
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
                # Task A1: Strictly set f2 = 0.0 across all circuits to eliminate golden-twin leakage
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
        # Classification head: projects concatenated [orig_x (8) + out_15d (15)] -> 2 classes
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
    if 'wb_conmax' in lower_name or 'wb-conmax' in lower_name: return 'wb_conmax'
    if 'rs232' in lower_name: return 'RS232'
    if 'aes' in lower_name: return 'AES'
    if circuit_name.startswith('c') or circuit_name.startswith('s'):
        return circuit_name.split('_')[0]
    return 'unknown'

def main():
    parser = argparse.ArgumentParser(description="Strict LOCO 6-Classifier Evaluation")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024, 7, 999], help="Random seeds")
    parser.add_argument("--epochs", type=int, default=40, help="GNN training epochs")
    parser.add_argument("--out_dir", type=str, default=".", help="Output directory")
    parser.add_argument("--include_inverted_rs232", action="store_true", help="Include the 22 inverted RS232 netlists (default: False)")
    parser.add_argument("--save_gate_scores", action="store_true", default=True, help="Save per-gate scores")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"=== Starting Strict Cross-Family LOCO Evaluation ===")
    print(f"Compute Device: {device}")
    print(f"Seeds: {args.seeds}")
    print(f"Exclude Inverted RS232: {not args.include_inverted_rs232}")
    
    # Target Google Drive if available
    if os.path.exists("/content/drive/MyDrive"):
        out_dir = os.path.join("/content/drive/MyDrive", "TrojanNet_Strict_LOCO")
        os.makedirs(out_dir, exist_ok=True)
        print(f"Detected Google Drive! Results saved permanently to: {out_dir}")
    else:
        out_dir = args.out_dir
        os.makedirs(out_dir, exist_ok=True)
        print(f"Results saved locally to: {os.path.abspath(out_dir)}")
        
    gate_scores_dir = os.path.join(out_dir, "gate_scores")
    os.makedirs(gate_scores_dir, exist_ok=True)
    
    raw_master_path = os.path.join(out_dir, "strict_loco_all_6classifiers_raw.csv")
    summary_path = os.path.join(out_dir, "strict_loco_summary.csv")
    
    # Load all dataset folders
    folders = [
        "training_datasets", "other_trit_tc_datasets", "rest_trit_tc_datasets", 
        "datasets_for_gnn", "b19_datasets", "ethernet_datasets", "wb_conmax_datasets", 
        "aes_output_datasets", "rs232_output_datasets"
    ]
    
    print("\nLoading benchmark graphs...")
    all_graphs = []
    for f in folders:
        if os.path.exists(f):
            all_graphs.extend(load_datasets_from_folder(f, exclude_inverted_rs232=not args.include_inverted_rs232))
            
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
    
    print(f"\nLoaded {len(all_graphs)} total unique benchmark circuits across {len(family_dict)} families:")
    for fam in target_families:
        c_names = [g.circuit_name for g in family_dict[fam]]
        clean_c = [c for c in c_names if 'normal' in c.lower() or 't300' in c.lower()]
        print(f"  - {fam:16s}: {len(c_names):2d} circuits (Trojan-free: {clean_c})")
        
    # Checkpoint resumption
    all_results = []
    completed_runs = set()
    if os.path.exists(raw_master_path):
        try:
            df_prev = pd.read_csv(raw_master_path)
            all_results = df_prev.to_dict('records')
            for _, r in df_prev.iterrows():
                completed_runs.add((int(r['Seed']), str(r['Family']), str(r['Model']), str(r['Circuit'])))
            print(f"\n[Checkpoint] Found existing results in {raw_master_path} with {len(all_results)} records. Resuming...")
        except Exception as e:
            print(f"Could not read previous checkpoint: {e}")
            
    total_start = time.time()
    
    for seed in args.seeds:
        print(f"\n{'='*75}")
        print(f"STARTING STRICT LOCO EVALUATION — SEED: {seed} ({args.seeds.index(seed)+1}/{len(args.seeds)})")
        print(f"{'='*75}")
        set_seed(seed)
        
        for test_family in target_families:
            if test_family not in family_dict or len(family_dict[test_family]) == 0:
                continue
                
            loco_test_graphs = family_dict[test_family]
            
            # Check if all models for this fold and seed are already completed
            models_to_run = ['GNN_solely', 'SVM', 'XGBoost', 'RandomForest', 'ExtraTrees']
            if CATBOOST_AVAILABLE:
                models_to_run.append('CatBoost')
                
            needed = False
            for m in models_to_run:
                for cg in loco_test_graphs:
                    if (seed, test_family, m, cg.circuit_name) not in completed_runs:
                        needed = True
                        break
                if needed: break
                
            if not needed:
                print(f"[Skip] Seed {seed}, Family {test_family} already fully completed.")
                continue
                
            print(f"\n>>> Fold: Test on {test_family} ({len(loco_test_graphs)} circuits) | Seed {seed}")
            
            # Form training pool: All other families completely held out
            loco_train_graphs = []
            core_train_families = ['RS232', 'c2670', 'c3540', 'c5315', 'c6288', 's13207', 's1423', 's15850', 's35932']
            for fam in core_train_families:
                if fam != test_family and fam in family_dict:
                    loco_train_graphs.extend(family_dict[fam])
                    
            if len(loco_train_graphs) == 0:
                continue
                
            num_features = loco_train_graphs[0].x.shape[1]
            gnn = GNNFeatureExtractor(in_channels=num_features, hidden_channels=32, out_channels=15).to(device)
            optimizer = torch.optim.Adam(gnn.parameters(), lr=0.005)
            
            total_nodes = sum([g.num_nodes for g in loco_train_graphs])
            total_ht = sum([g.y.sum().item() for g in loco_train_graphs])
            pos_weight = torch.tensor([(total_nodes - total_ht) / max(1, total_ht)], dtype=torch.float).to(device)
            criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight.item()], dtype=torch.float).to(device))
            
            train_loader = DataLoader(loco_train_graphs, batch_size=4, shuffle=True)
            
            # Train GNN encoder
            gnn.train()
            for epoch in range(args.epochs):
                for data in train_loader:
                    data = data.to(device)
                    optimizer.zero_grad()
                    out = gnn(data.x, data.edge_index)
                    loss = criterion(out, data.y)
                    loss.backward()
                    optimizer.step()
                    
            # 1. Evaluate GNN (solely) directly on test circuits using GNN head argmax
            gnn.eval()
            for c_graph in loco_test_graphs:
                c_name = c_graph.circuit_name
                run_key = (seed, test_family, 'GNN_solely', c_name)
                if run_key not in completed_runs:
                    with torch.no_grad():
                        g_dev = c_graph.clone().to(device)
                        logits = gnn(g_dev.x, g_dev.edge_index)
                        preds = logits.argmax(dim=1).cpu().numpy()
                        proba = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                        y_true = g_dev.y.cpu().numpy()
                        
                    acc = accuracy_score(y_true, preds)
                    bal_acc = balanced_accuracy_score(y_true, preds)
                    prec = precision_score(y_true, preds, zero_division=0)
                    rec = recall_score(y_true, preds, zero_division=0)
                    f1 = f1_score(y_true, preds, zero_division=0)
                    mcc = matthews_corrcoef(y_true, preds)
                    try:
                        pr_auc = average_precision_score(y_true, proba)
                    except Exception:
                        pr_auc = 0.0
                    cm = confusion_matrix(y_true, preds, labels=[0, 1])
                    tn, fp, fn, tp = cm.ravel()
                    
                    rec_entry = {
                        "Seed": seed, "Family": test_family, "Circuit": c_name,
                        "Feature_Set": "gnn_head", "Model": "GNN_solely",
                        "Accuracy": acc, "Balanced_Accuracy": bal_acc,
                        "Precision": prec, "Recall": rec, "F1_Score": f1,
                        "MCC": mcc, "PR_AUC": pr_auc,
                        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)
                    }
                    all_results.append(rec_entry)
                    completed_runs.add(run_key)
                    
                    if args.save_gate_scores:
                        node_names = getattr(c_graph, 'node_names', [str(i) for i in range(len(y_true))])
                        df_gates = pd.DataFrame({
                            'Node': node_names, 'y_true': y_true, 'score': proba, 'pred': preds
                        })
                        gate_file = os.path.join(gate_scores_dir, f"gate_scores_s{seed}_GNN_solely_{c_name}.csv")
                        df_gates.to_csv(gate_file, index=False)
                        
            # 2. Extract 23-D embeddings for downstream tree & SVM models
            X_train_emb, y_train_emb = [], []
            with torch.no_grad():
                for g in loco_train_graphs:
                    g_dev = g.clone().to(device)
                    emb = gnn.extract_embeddings(g_dev.x, g_dev.edge_index)
                    X_train_emb.append(emb.cpu())
                    y_train_emb.append(g_dev.y.cpu())
            X_train_emb = torch.cat(X_train_emb).numpy()
            y_train_emb = torch.cat(y_train_emb).numpy()
            
            test_embeddings = {}
            with torch.no_grad():
                for c_graph in loco_test_graphs:
                    g_dev = c_graph.clone().to(device)
                    X_test = gnn.extract_embeddings(g_dev.x, g_dev.edge_index).cpu().numpy()
                    y_test = g_dev.y.cpu().numpy()
                    node_names = getattr(c_graph, 'node_names', [str(i) for i in range(len(y_test))])
                    test_embeddings[c_graph.circuit_name] = (X_test, y_test, node_names)
                    
            # 85/15 validation split on training embeddings for threshold optimization
            np.random.seed(seed)
            random.seed(seed)
            X_tr_sub, X_val, y_tr_sub, y_val = train_test_split(
                X_train_emb, y_train_emb, test_size=0.15, stratify=y_train_emb, random_state=seed
            )
            
            # Initialize downstream models
            downstream_models = {
                "SVM": CalibratedClassifierCV(
                    estimator=LinearSVC(class_weight='balanced', random_state=seed, max_iter=2500),
                    cv=3
                ),
                "XGBoost": XGBClassifier(
                    scale_pos_weight=pos_weight.item(), eval_metric='logloss',
                    random_state=seed, n_jobs=-1
                ),
                "RandomForest": RandomForestClassifier(
                    n_estimators=50, max_depth=None, class_weight="balanced",
                    n_jobs=-1, random_state=seed
                ),
                "ExtraTrees": ExtraTreesClassifier(
                    n_estimators=50, max_depth=None, class_weight="balanced",
                    n_jobs=-1, random_state=seed
                )
            }
            if CATBOOST_AVAILABLE:
                downstream_models["CatBoost"] = CatBoostClassifier(
                    auto_class_weights='Balanced', verbose=0, random_seed=seed, thread_count=-1
                )
                
            for m_name, model in downstream_models.items():
                # Check if all circuits for this model are already done
                circs_needed = [cg.circuit_name for cg in loco_test_graphs if (seed, test_family, m_name, cg.circuit_name) not in completed_runs]
                if not circs_needed:
                    continue
                    
                model.fit(X_tr_sub, y_tr_sub)
                
                # Validation threshold tuning
                val_proba = model.predict_proba(X_val)[:, 1]
                best_thresh, _ = find_best_threshold(y_val, val_proba)
                
                for c_name in circs_needed:
                    X_test, y_test, node_names = test_embeddings[c_name]
                    test_proba = model.predict_proba(X_test)[:, 1]
                    m_preds = (test_proba >= best_thresh).astype(int)
                    
                    acc = accuracy_score(y_test, m_preds)
                    bal_acc = balanced_accuracy_score(y_test, m_preds)
                    prec = precision_score(y_test, m_preds, zero_division=0)
                    rec = recall_score(y_test, m_preds, zero_division=0)
                    f1 = f1_score(y_test, m_preds, zero_division=0)
                    mcc = matthews_corrcoef(y_test, m_preds)
                    try:
                        pr_auc = average_precision_score(y_test, test_proba)
                    except Exception:
                        pr_auc = 0.0
                    cm = confusion_matrix(y_test, m_preds, labels=[0, 1])
                    tn, fp, fn, tp = cm.ravel()
                    
                    rec_entry = {
                        "Seed": seed, "Family": test_family, "Circuit": c_name,
                        "Feature_Set": "full_23", "Model": m_name,
                        "Accuracy": acc, "Balanced_Accuracy": bal_acc,
                        "Precision": prec, "Recall": rec, "F1_Score": f1,
                        "MCC": mcc, "PR_AUC": pr_auc,
                        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)
                    }
                    all_results.append(rec_entry)
                    completed_runs.add((seed, test_family, m_name, c_name))
                    
                    if args.save_gate_scores:
                        df_gates = pd.DataFrame({
                            'Node': node_names, 'y_true': y_test, 'score': test_proba, 'pred': m_preds
                        })
                        gate_file = os.path.join(gate_scores_dir, f"gate_scores_s{seed}_{m_name}_{c_name}.csv")
                        df_gates.to_csv(gate_file, index=False)
                        
            # Periodically checkpoint
            df_curr = pd.DataFrame(all_results)
            df_curr.to_csv(raw_master_path, index=False)
            
    print(f"\n{'='*75}")
    print(f"EVALUATION COMPLETED IN {(time.time() - total_start):.2f} SECONDS!")
    print(f"{'='*75}")
    
    # Export individual CSVs per classifier
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(raw_master_path, index=False)
    print(f"Master CSV saved: {raw_master_path}")
    
    for m in df_all['Model'].unique():
        df_m = df_all[df_all['Model'] == m]
        m_filename = os.path.join(out_dir, f"strict_loco_{m.lower()}.csv")
        df_m.to_csv(m_filename, index=False)
        print(f"  --> Saved individual CSV: {m_filename} ({len(df_m)} rows)")
        
    # Generate Summary Aggregation Table
    summary_records = []
    for m in df_all['Model'].unique():
        df_m = df_all[df_all['Model'] == m]
        fam_grouped = df_m.groupby(['Family', 'Seed'])[['Accuracy', 'Balanced_Accuracy', 'Precision', 'Recall', 'F1_Score', 'MCC', 'PR_AUC']].mean().reset_index()
        fam_stats = fam_grouped.groupby('Family')[['Accuracy', 'Balanced_Accuracy', 'Precision', 'Recall', 'F1_Score', 'MCC', 'PR_AUC']].agg(['mean', 'std'])
        
        for fam in fam_stats.index:
            summary_records.append({
                "Model": m, "Family": fam,
                "Accuracy": f"{fam_stats.loc[fam, ('Accuracy', 'mean')]:.4f} ± {fam_stats.loc[fam, ('Accuracy', 'std')]:.4f}",
                "Balanced_Accuracy": f"{fam_stats.loc[fam, ('Balanced_Accuracy', 'mean')]:.4f} ± {fam_stats.loc[fam, ('Balanced_Accuracy', 'std')]:.4f}",
                "Precision": f"{fam_stats.loc[fam, ('Precision', 'mean')]:.4f} ± {fam_stats.loc[fam, ('Precision', 'std')]:.4f}",
                "Recall": f"{fam_stats.loc[fam, ('Recall', 'mean')]:.4f} ± {fam_stats.loc[fam, ('Recall', 'std')]:.4f}",
                "F1_Score": f"{fam_stats.loc[fam, ('F1_Score', 'mean')]:.4f} ± {fam_stats.loc[fam, ('F1_Score', 'std')]:.4f}",
                "MCC": f"{fam_stats.loc[fam, ('MCC', 'mean')]:.4f} ± {fam_stats.loc[fam, ('MCC', 'std')]:.4f}",
                "PR_AUC": f"{fam_stats.loc[fam, ('PR_AUC', 'mean')]:.4f} ± {fam_stats.loc[fam, ('PR_AUC', 'std')]:.4f}"
            })
            
        overall_grouped = fam_grouped.groupby('Seed')[['Accuracy', 'Balanced_Accuracy', 'Precision', 'Recall', 'F1_Score', 'MCC', 'PR_AUC']].mean().reset_index()
        om = overall_grouped.mean()
        ostd = overall_grouped.std()
        summary_records.append({
            "Model": m, "Family": "OVERALL_MACRO",
            "Accuracy": f"{om['Accuracy']:.4f} ± {ostd['Accuracy']:.4f}",
            "Balanced_Accuracy": f"{om['Balanced_Accuracy']:.4f} ± {ostd['Balanced_Accuracy']:.4f}",
            "Precision": f"{om['Precision']:.4f} ± {ostd['Precision']:.4f}",
            "Recall": f"{om['Recall']:.4f} ± {ostd['Recall']:.4f}",
            "F1_Score": f"{om['F1_Score']:.4f} ± {ostd['F1_Score']:.4f}",
            "MCC": f"{om['MCC']:.4f} ± {ostd['MCC']:.4f}",
            "PR_AUC": f"{om['PR_AUC']:.4f} ± {ostd['PR_AUC']:.4f}"
        })
        
    df_sum = pd.DataFrame(summary_records)
    df_sum.to_csv(summary_path, index=False)
    print(f"\nSummary table saved to: {summary_path}")
    
    # Compress gate scores for convenient export
    if args.save_gate_scores and os.path.exists(gate_scores_dir):
        zip_gate_path = os.path.join(out_dir, "gate_scores.zip")
        print(f"\nCompressing per-gate score files to: {zip_gate_path}...")
        import zipfile
        with zipfile.ZipFile(zip_gate_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            for g_file in glob.glob(os.path.join(gate_scores_dir, "*.csv")):
                zf.write(g_file, os.path.basename(g_file))
        print(f"Gate scores compressed successfully ({os.path.getsize(zip_gate_path)/(1024*1024):.2f} MB)!")

if __name__ == "__main__":
    main()
