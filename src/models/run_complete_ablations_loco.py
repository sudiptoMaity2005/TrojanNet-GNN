import os
import glob
import time
import random
import argparse
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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from collections import defaultdict

INVERTED_RS232_VARIANTS = {
    'RS232-T100', 'RS232-T200', 'RS232-T300', 'RS232-T400', 'RS232-T500',
    'RS232-T600', 'RS232-T700', 'RS232-T800', 'RS232-T900', 'RS232-T901'
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
            # Note: Do not attach python list node_names to Data to avoid DataLoader collation conflicts
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

def train_gnn_model(train_graphs, device, in_dim, use_jk=True, use_residual=True, weighted_ce=True, epochs=40):
    gnn = ConfigurableGNN(in_dim, 32, 15, use_jk=use_jk, use_residual=use_residual).to(device)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=0.005)

    tot_nodes = sum(g.num_nodes for g in train_graphs)
    tot_ht = sum(g.y.sum().item() for g in train_graphs)
    pos_wt = (tot_nodes - tot_ht) / max(1, tot_ht)

    if weighted_ce:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_wt], dtype=torch.float).to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    loader = DataLoader(train_graphs, batch_size=4, shuffle=True, exclude_keys=['node_names', 'circuit_name'])
    gnn.train()
    for _ in range(epochs):
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(gnn(batch.x, batch.edge_index), batch.y)
            loss.backward()
            optimizer.step()
    return gnn

def extract_features(gnn, graphs, device):
    gnn.eval()
    X_list, y_list = [], []
    with torch.no_grad():
        for g in graphs:
            g_dev = g.clone().to(device)
            X_list.append(gnn.extract_embeddings(g_dev.x, g_dev.edge_index).cpu())
            y_list.append(g_dev.y.cpu())
    X = torch.cat(X_list).numpy()
    y = torch.cat(y_list).numpy()
    return X, y

def main():
    parser = argparse.ArgumentParser(description="Task B2 Complete Reviewer Ablations (Strict LOCO)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=40)
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
        if fam != 'unknown':
            family_dict[fam].append(g)

    target_families = ['AES', 'RS232', 'b19', 'EthernetMAC10GE', 'wb_conmax', 'c2670', 'c3540', 'c5315', 'c6288', 's13207', 's1423', 's15850', 's35932']
    core_train_families = ['RS232', 'c2670', 'c3540', 'c5315', 'c6288', 's13207', 's1423', 's15850', 's35932']
    seeds = [42, 123, 2024, 7, 999]

    feature_names = [f"f{i+1}" for i in range(8)] + [f"emb_{j+1}" for j in range(15)]
    all_results = []
    fi_records = []
    completed_runs = set()

    # Checkpoint resumption
    if os.path.exists(args.out_csv):
        try:
            df_prev = pd.read_csv(args.out_csv)
            all_results = df_prev.to_dict('records')
            for _, r in df_prev.iterrows():
                completed_runs.add((int(r['Seed']), str(r['Family']), str(r['Config']), str(r['Circuit'])))
            print(f"[Checkpoint] Resuming from {args.out_csv} with {len(all_results)} existing records.")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")

    for seed in seeds:
        print(f"\n{'='*75}")
        print(f"TASK B2 ABLATION — SEED {seed} ({seeds.index(seed)+1}/{len(seeds)})")
        print(f"{'='*75}")
        set_seed(seed)

        for test_family in target_families:
            if test_family not in family_dict or len(family_dict[test_family]) == 0:
                continue

            test_graphs = family_dict[test_family]
            train_graphs = []
            for fam in core_train_families:
                if fam != test_family and fam in family_dict:
                    train_graphs.extend(family_dict[fam])

            if len(train_graphs) == 0:
                continue

            print(f"\n>>> Fold: Test on {test_family} ({len(test_graphs)} circuits) | Seed {seed}")
            in_dim = train_graphs[0].x.shape[1]

            # -------------------------------------------------------------
            # Standard GNN: used for Representation, Weights, Depth, Drops
            # -------------------------------------------------------------
            gnn_std = train_gnn_model(train_graphs, device, in_dim, use_jk=True, use_residual=True, weighted_ce=True, epochs=args.epochs)
            X_tr, y_tr = extract_features(gnn_std, train_graphs, device)

            test_data_std = {}
            with torch.no_grad():
                for g in test_graphs:
                    g_dev = g.clone().to(device)
                    emb = gnn_std.extract_embeddings(g_dev.x, g_dev.edge_index).cpu().numpy()
                    test_data_std[g.circuit_name] = (emb, g.y.numpy())

            X_tr_sub, X_val, y_tr_sub, y_val = train_test_split(
                X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=seed
            )

            # 1. Feature Representation Ablations: raw_8, emb_15, hybrid
            repr_configs = {
                "raw_8": list(range(8)),
                "emb_15": list(range(8, 23)),
                "hybrid": list(range(23))
            }
            for rep_name, idxs in repr_configs.items():
                needed = any((seed, test_family, rep_name, cg.circuit_name) not in completed_runs for cg in test_graphs)
                if not needed:
                    continue
                clf = ExtraTreesClassifier(n_estimators=50, class_weight="balanced", random_state=seed, n_jobs=-1)
                clf.fit(X_tr_sub[:, idxs], y_tr_sub)

                if rep_name == "hybrid":
                    for fn, imp in zip(feature_names, clf.feature_importances_):
                        fi_records.append({"Seed": seed, "Fold": test_family, "Feature": fn, "Importance": imp})

                val_p = clf.predict_proba(X_val[:, idxs])[:, 1]
                t_best, _ = find_best_threshold(y_val, val_p)

                for c_name, (X_te, y_te) in test_data_std.items():
                    run_key = (seed, test_family, rep_name, c_name)
                    if run_key in completed_runs:
                        continue
                    p = clf.predict_proba(X_te[:, idxs])[:, 1]
                    pred = (p >= t_best).astype(int)
                    m = evaluate_predictions(y_te, pred, p)
                    all_results.append({
                        "Seed": seed, "Family": test_family, "Circuit": c_name,
                        "Ablation_Category": "Representation", "Config": rep_name,
                        "Model": "ExtraTrees", "Accuracy": m[0], "Balanced_Accuracy": m[1],
                        "Precision": m[2], "Recall": m[3], "F1_Score": m[4], "MCC": m[5], "PR_AUC": m[6],
                        "TP": m[7], "FP": m[8], "TN": m[9], "FN": m[10]
                    })
                    completed_runs.add(run_key)

            # 2. Tree Weight Ablation: no_tree_weights
            needed = any((seed, test_family, "no_tree_weights", cg.circuit_name) not in completed_runs for cg in test_graphs)
            if needed:
                clf_unweighted = ExtraTreesClassifier(n_estimators=50, class_weight=None, random_state=seed, n_jobs=-1)
                clf_unweighted.fit(X_tr_sub, y_tr_sub)
                val_p = clf_unweighted.predict_proba(X_val)[:, 1]
                t_best, _ = find_best_threshold(y_val, val_p)
                for c_name, (X_te, y_te) in test_data_std.items():
                    run_key = (seed, test_family, "no_tree_weights", c_name)
                    if run_key in completed_runs:
                        continue
                    p = clf_unweighted.predict_proba(X_te)[:, 1]
                    pred = (p >= t_best).astype(int)
                    m = evaluate_predictions(y_te, pred, p)
                    all_results.append({
                        "Seed": seed, "Family": test_family, "Circuit": c_name,
                        "Ablation_Category": "Weighting", "Config": "no_tree_weights",
                        "Model": "ExtraTrees", "Accuracy": m[0], "Balanced_Accuracy": m[1],
                        "Precision": m[2], "Recall": m[3], "F1_Score": m[4], "MCC": m[5], "PR_AUC": m[6],
                        "TP": m[7], "FP": m[8], "TN": m[9], "FN": m[10]
                    })
                    completed_runs.add(run_key)

            # 3. Tree Depth Ablations: depth_1, depth_2, depth_4, depth_5
            for depth in [1, 2, 4, 5]:
                cfg_name = f"depth_{depth}"
                needed = any((seed, test_family, cfg_name, cg.circuit_name) not in completed_runs for cg in test_graphs)
                if not needed:
                    continue
                clf_depth = ExtraTreesClassifier(n_estimators=50, max_depth=depth, class_weight="balanced", random_state=seed, n_jobs=-1)
                clf_depth.fit(X_tr_sub, y_tr_sub)
                val_p = clf_depth.predict_proba(X_val)[:, 1]
                t_best, _ = find_best_threshold(y_val, val_p)
                for c_name, (X_te, y_te) in test_data_std.items():
                    run_key = (seed, test_family, cfg_name, c_name)
                    if run_key in completed_runs:
                        continue
                    p = clf_depth.predict_proba(X_te)[:, 1]
                    pred = (p >= t_best).astype(int)
                    m = evaluate_predictions(y_te, pred, p)
                    all_results.append({
                        "Seed": seed, "Family": test_family, "Circuit": c_name,
                        "Ablation_Category": "Tree_Depth", "Config": cfg_name,
                        "Model": "ExtraTrees", "Accuracy": m[0], "Balanced_Accuracy": m[1],
                        "Precision": m[2], "Recall": m[3], "F1_Score": m[4], "MCC": m[5], "PR_AUC": m[6],
                        "TP": m[7], "FP": m[8], "TN": m[9], "FN": m[10]
                    })
                    completed_runs.add(run_key)

            # 4. Feature Drop & Correlated Group Ablations
            feature_drops = {
                "drop_f1": [0],
                "drop_f3": [2],
                "drop_f4": [3],
                "drop_f5": [4],
                "drop_f6": [5],
                "drop_f7": [6],
                "drop_f8": [7],
                "drop_group_TP": [0, 2, 3],
                "drop_group_Degree": [6, 7],
                "drop_group_Complexity": [4, 5]
            }
            for drop_name, drop_cols in feature_drops.items():
                needed = any((seed, test_family, drop_name, cg.circuit_name) not in completed_runs for cg in test_graphs)
                if not needed:
                    continue
                keep_cols = [c for c in range(23) if c not in drop_cols]
                clf_drop = ExtraTreesClassifier(n_estimators=50, class_weight="balanced", random_state=seed, n_jobs=-1)
                clf_drop.fit(X_tr_sub[:, keep_cols], y_tr_sub)
                val_p = clf_drop.predict_proba(X_val[:, keep_cols])[:, 1]
                t_best, _ = find_best_threshold(y_val, val_p)
                for c_name, (X_te, y_te) in test_data_std.items():
                    run_key = (seed, test_family, drop_name, c_name)
                    if run_key in completed_runs:
                        continue
                    p = clf_drop.predict_proba(X_te[:, keep_cols])[:, 1]
                    pred = (p >= t_best).astype(int)
                    m = evaluate_predictions(y_te, pred, p)
                    all_results.append({
                        "Seed": seed, "Family": test_family, "Circuit": c_name,
                        "Ablation_Category": "Feature_Drop", "Config": drop_name,
                        "Model": "ExtraTrees", "Accuracy": m[0], "Balanced_Accuracy": m[1],
                        "Precision": m[2], "Recall": m[3], "F1_Score": m[4], "MCC": m[5], "PR_AUC": m[6],
                        "TP": m[7], "FP": m[8], "TN": m[9], "FN": m[10]
                    })
                    completed_runs.add(run_key)

            # -------------------------------------------------------------
            # 5. Architecture & Loss Ablations: no_jk, no_residual, plain_cross_entropy
            # -------------------------------------------------------------
            arch_variants = [
                ("Architecture", "no_jk", False, True, True),
                ("Architecture", "no_residual", True, False, True),
                ("Loss_Function", "plain_cross_entropy", True, True, False)
            ]
            for cat_name, cfg_name, u_jk, u_res, w_ce in arch_variants:
                needed = any((seed, test_family, cfg_name, cg.circuit_name) not in completed_runs for cg in test_graphs)
                if not needed:
                    continue
                gnn_var = train_gnn_model(train_graphs, device, in_dim, use_jk=u_jk, use_residual=u_res, weighted_ce=w_ce, epochs=args.epochs)
                X_tr_v, y_tr_v = extract_features(gnn_var, train_graphs, device)

                X_tr_sub_v, X_val_v, y_tr_sub_v, y_val_v = train_test_split(
                    X_tr_v, y_tr_v, test_size=0.15, stratify=y_tr_v, random_state=seed
                )

                clf_var = ExtraTreesClassifier(n_estimators=50, class_weight="balanced", random_state=seed, n_jobs=-1)
                clf_var.fit(X_tr_sub_v, y_tr_sub_v)
                val_p = clf_var.predict_proba(X_val_v)[:, 1]
                t_best, _ = find_best_threshold(y_val_v, val_p)

                with torch.no_grad():
                    for cg in test_graphs:
                        run_key = (seed, test_family, cfg_name, cg.circuit_name)
                        if run_key in completed_runs:
                            continue
                        g_dev = cg.clone().to(device)
                        emb_test = gnn_var.extract_embeddings(g_dev.x, g_dev.edge_index).cpu().numpy()
                        p = clf_var.predict_proba(emb_test)[:, 1]
                        pred = (p >= t_best).astype(int)
                        m = evaluate_predictions(cg.y.numpy(), pred, p)
                        all_results.append({
                            "Seed": seed, "Family": test_family, "Circuit": cg.circuit_name,
                            "Ablation_Category": cat_name, "Config": cfg_name,
                            "Model": "ExtraTrees", "Accuracy": m[0], "Balanced_Accuracy": m[1],
                            "Precision": m[2], "Recall": m[3], "F1_Score": m[4], "MCC": m[5], "PR_AUC": m[6],
                            "TP": m[7], "FP": m[8], "TN": m[9], "FN": m[10]
                        })
                        completed_runs.add(run_key)

            # Auto-save checkpoint after each fold
            pd.DataFrame(all_results).to_csv(args.out_csv, index=False)

    # Final summary export
    df_res = pd.DataFrame(all_results)
    df_res.to_csv(args.out_csv, index=False)
    print(f"\n[DONE] Master Ablation CSV saved: {args.out_csv} ({len(df_res)} rows)")

    if fi_records:
        df_fi = pd.DataFrame(fi_records)
        df_fi.to_csv(args.fi_csv, index=False)
        print(f"[DONE] Feature importances saved: {args.fi_csv} ({len(df_fi)} rows)")

    summary = df_res.groupby(["Ablation_Category", "Config"])[["Accuracy", "Balanced_Accuracy", "Precision", "Recall", "F1_Score", "MCC", "PR_AUC"]].mean().reset_index()
    summary.to_csv("ablation_complete_summary.csv", index=False)
    print(f"[DONE] Summary table saved: ablation_complete_summary.csv")

if __name__ == "__main__":
    main()
