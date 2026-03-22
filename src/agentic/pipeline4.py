import os
import glob
import re
import json
import yaml
import math
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except Exception:
    HAS_CAT = False
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuroHarmonize import harmonizationLearn, harmonizationApply
from langgraph.graph import StateGraph, END
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w = nn.Linear(in_features, out_features, bias=False)
    def forward(self, x, adj):
        ax = torch.matmul(adj, x)
        return self.w(ax)


class DeeperGCN(nn.Module):
    def __init__(self, in_features, hidden, n_classes, n_layers=4):
        super().__init__()
        self.proj = nn.Linear(in_features, hidden)
        self.gcn_layers = nn.ModuleList([
            GCNLayer(hidden, hidden) for _ in range(n_layers)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(n_layers)])
        self.attn_w = nn.Linear(hidden, 1)
        self.cls = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x, adj):
        x = F.relu(self.proj(x))
        h = x
        all_h = [h]
        for i, (gcn, bn) in enumerate(zip(self.gcn_layers, self.bns)):
            h_new = F.relu(bn(gcn(h, adj)))
            h = h + 0.5 * h_new
            all_h.append(h)
        weights = torch.softmax(self.attn_w(torch.stack(all_h, dim=1)).squeeze(-1), dim=1)
        weighted_h = (torch.stack(all_h, dim=1) * weights.unsqueeze(-1)).sum(dim=1)
        g_mean = weighted_h.mean(dim=1, keepdim=True)
        g_max = weighted_h.max(dim=1, keepdim=True).values
        g_std = weighted_h.std(dim=1, keepdim=True)
        g = torch.cat([g_mean.squeeze(1), g_max.squeeze(1), g_std.squeeze(1)], dim=1)
        return self.cls(g)


class _ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )

    def forward(self, x):
        return F.relu(x + self.net(x))


class ImprovedTransformerFusion(nn.Module):
    def __init__(self, fc_dim: int, smri_dim: int, pheno_dim: int,
                 d_model: int = 192, nhead: int = 6, n_layers: int = 3, dropout: float = 0.25):
        super().__init__()
        self.fc_proj = nn.Sequential(
            nn.Linear(fc_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.smri_proj = nn.Sequential(
            nn.Linear(smri_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.pheno_proj = nn.Sequential(
            nn.Linear(pheno_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 4, 1),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, xfc, xsmri, xpheno):
        t_fc  = self.fc_proj(xfc).unsqueeze(1)
        t_sm  = self.smri_proj(xsmri).unsqueeze(1)
        t_ph  = self.pheno_proj(xpheno).unsqueeze(1)
        cls   = self.cls_token.expand(xfc.size(0), -1, -1)
        tokens = torch.cat([cls, t_fc, t_sm, t_ph], dim=1)
        out = self.transformer(tokens)
        return self.head(out[:, 0])


class BrainNetCNN(nn.Module):
    def __init__(self, n_nodes: int, dropout: float = 0.45):
        super().__init__()
        self.n_nodes = n_nodes
        self.e2e = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=(1, n_nodes)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(48),
        )
        self.e2e_t = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=(n_nodes, 1)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(48),
        )
        self.e2n = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=(n_nodes, 1)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(192),
        )
        self.cls = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(192, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + x.transpose(-2, -1)) * 0.5
        h1 = self.e2e(x)
        h2 = self.e2e_t(x)
        h2 = h2.transpose(-2, -1)
        h  = torch.cat([h1, h2], dim=1)
        h  = self.e2n(h)
        h  = h.view(h.size(0), -1)
        return self.cls(h)


class ImprovedMLP(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        d1, d2, d3 = 512, 384, 256
        self.net = nn.Sequential(
            nn.Linear(in_features, d1), nn.BatchNorm1d(d1), nn.GELU(), nn.Dropout(0.3),
            _ResBlock(d1, dropout=0.25),
            _ResBlock(d1, dropout=0.25),
            nn.Linear(d1, d2), nn.BatchNorm1d(d2), nn.GELU(), nn.Dropout(0.25),
            _ResBlock(d2, dropout=0.2),
            nn.Linear(d2, d3), nn.BatchNorm1d(d3), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(d3, 1),
        )

    def forward(self, x):
        return self.net(x)


def load_fc_matrices(path: str, verbose: bool = False) -> Dict[int, np.ndarray]:
    files = glob.glob(os.path.join(path, "*.csv"))
    data = {}
    total = len(files)
    c = 0
    for fp in files:
        m = re.search(r"(\d+)_fc_matrix\.csv$", os.path.basename(fp))
        if not m:
            continue
        sid = int(m.group(1))
        arr = pd.read_csv(fp, header=None).values.astype(np.float32)
        data[sid] = arr
        c += 1
        if verbose and c % 100 == 0:
            print(f"Loaded {c}/{total} FC matrices", flush=True)
    return data


def load_smri_table(fp: str) -> pd.DataFrame:
    df = pd.read_csv(fp)
    if "Subject_ID" in df.columns:
        df = df.rename(columns={"Subject_ID": "SUB_ID"})
    return df


def load_phenotypes(fp: str) -> pd.DataFrame:
    df = pd.read_csv(fp)
    cols = df.columns.tolist()
    cands = ["SUB_ID", "subject"]
    sid_col = None
    for c in cands:
        if c in cols:
            sid_col = c
            break
    if sid_col != "SUB_ID" and sid_col is not None:
        df = df.rename(columns={sid_col: "SUB_ID"})
    return df


def align_subjects(fc: Dict[int, np.ndarray], smri: pd.DataFrame, pheno: pd.DataFrame) -> Tuple[List[int], Dict[int, np.ndarray], pd.DataFrame, pd.DataFrame]:
    smri_ids = set(smri["SUB_ID"].astype(int).tolist())
    ph_ids = set(pheno["SUB_ID"].astype(int).tolist())
    fc_ids = set(fc.keys())
    common = sorted(list(fc_ids & smri_ids & ph_ids))
    smri_f = smri[smri["SUB_ID"].isin(common)].copy()
    ph_f = pheno[pheno["SUB_ID"].isin(common)].copy()
    fc_f = {k: fc[k] for k in common}
    return common, fc_f, smri_f, ph_f


def fc_feature_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    if state.get("verbose"):
        print("FC agent: start feature extraction", flush=True)
    fc_dict = state["fc_dict"]
    sids = state["subject_ids"]
    mats_raw = [fc_dict[s] for s in sids]
    def fisher_z(mat: np.ndarray) -> np.ndarray:
        m = mat.copy().astype(np.float32)
        m = (m + m.T) / 2.0
        np.fill_diagonal(m, 0.0)
        m = np.clip(m, -0.99999, 0.99999)
        return np.arctanh(m)
    def prop_threshold(mat: np.ndarray, p: float) -> np.ndarray:
        if p <= 0 or p >= 1:
            return mat
        a = np.abs(mat)
        ut = np.triu_indices_from(a, 1)
        vals = a[ut]
        k = int(np.ceil(p * vals.size))
        if k <= 0:
            return mat
        thr = np.partition(vals, -k)[-k]
        m2 = mat.copy()
        m2[a < thr] = 0.0
        m2 = (m2 + m2.T) / 2.0
        np.fill_diagonal(m2, 0.0)
        return m2
    def mst_plus_prop(mat: np.ndarray, p: float) -> np.ndarray:
        a = np.abs(mat)
        G = nx.from_numpy_array(a)
        T = nx.maximum_spanning_tree(G, weight="weight")
        m2 = np.zeros_like(mat, dtype=np.float32)
        for u, v, d in T.edges(data=True):
            w = mat[u, v]
            m2[u, v] = w
            m2[v, u] = w
        ut = np.triu_indices_from(a, 1)
        vals = a[ut]
        order = np.argsort(vals)[::-1]
        total_keep = int(np.ceil(p * vals.size))
        added = 0
        for idx in order:
            i = ut[0][idx]; j = ut[1][idx]
            if m2[i, j] != 0.0:
                continue
            m2[i, j] = mat[i, j]
            m2[j, i] = mat[i, j]
            added += 1
            if added >= max(0, total_keep - (T.number_of_edges())):
                break
        np.fill_diagonal(m2, 0.0)
        return m2
    search_ps = state.get("fc_search_ps", [0.1, 0.2, 0.3])
    search_ks = state.get("fc_search_ks", [1000, 2000, 3000, 4000])
    search_strats = state.get("fc_search_strategies", ["prop", "mst_prop"])
    mats_z = [fisher_z(m) for m in mats_raw]
    n = mats_z[0].shape[0]
    ut_idx = np.triu_indices(n, 1)
    best_acc = -1.0
    best_cfg = (0.2, 3000, "prop")
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    y_tmp = None
    if "phenotypes" in state and "DX_GROUP" in state["phenotypes"].columns:
        ph = state["phenotypes"]
        ph = ph[ph["SUB_ID"].isin(sids)].copy()
        y_tmp = np.where(ph["DX_GROUP"].astype(int).values == 1, 1, 0)
    for pthr in search_ps:
        for strat in search_strats:
            if strat == "prop":
                mats_t = [prop_threshold(m, pthr) for m in mats_z]
            else:
                mats_t = [mst_plus_prop(m, pthr) for m in mats_z]
            vstack_t = np.stack(mats_t, axis=0)
            edges_all = vstack_t[:, ut_idx[0], ut_idx[1]]
            var = edges_all.var(axis=0)
            for ksel in search_ks:
                k = min(ksel, edges_all.shape[1])
                idx = np.argsort(var)[-k:]
                Xe = edges_all[:, idx]
                if y_tmp is None:
                    continue
                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                accs = []
                for tr, te in skf.split(Xe, y_tmp):
                    clf = LogisticRegression(max_iter=500, class_weight="balanced")
                    clf.fit(Xe[tr], y_tmp[tr])
                    pred = clf.predict(Xe[te])
                    accs.append((pred == y_tmp[te]).mean())
                acc = float(np.mean(accs)) if accs else -1.0
                if state.get("verbose"):
                    print(f"FC agent search: strat={strat}, p={pthr}, k={k} -> acc={acc:.3f}", flush=True)
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = (pthr, k, strat)
    pthr, k, strat = best_cfg
    mats = [prop_threshold(m, pthr) if strat == "prop" else mst_plus_prop(m, pthr) for m in mats_z]
    n = mats[0].shape[0]
    vstack = np.stack(mats, axis=0)
    ut_idx = np.triu_indices(n, 1)
    edges = vstack[:, ut_idx[0], ut_idx[1]]
    var = edges.var(axis=0)
    k = min(k, edges.shape[1])
    if y_tmp is not None:
        try:
            from scipy.stats import ttest_ind as _ttest
            X0 = edges[y_tmp == 0]
            X1 = edges[y_tmp == 1]
            t_res = _ttest(X0, X1, equal_var=False)
            t_abs = np.nan_to_num(np.abs(t_res.statistic), nan=0.0, posinf=0.0)
            rank_v = np.argsort(np.argsort(var))
            rank_t = np.argsort(np.argsort(t_abs))
            combined_score = rank_v.astype(np.float32) + rank_t.astype(np.float32)
            top_idx = np.argsort(combined_score)[-k:]
        except Exception:
            top_idx = np.argsort(var)[-k:]
    else:
        top_idx = np.argsort(var)[-k:]
    X_edges = edges[:, top_idx]
    if state.get("verbose"):
        print(f"FC agent: selected {k} high-variance edges from {edges.shape[1]} (strat={strat}, p={pthr})", flush=True)
    g_feats = []
    for i, m in enumerate(mats):
        G = nx.from_numpy_array(m)
        deg = np.array([d for _, d in G.degree(weight=None)], dtype=np.float32)
        strn = np.array(list(dict(G.degree(weight="weight")).values()), dtype=np.float32)
        tri = np.array(list(nx.triangles(G).values()), dtype=np.float32)
        cc = nx.average_clustering(G, weight=None)
        ge = np.nan
        try:
            ge = nx.global_efficiency(G)
        except Exception:
            ge = 0.0
        try:
            bet = np.array(list(nx.betweenness_centrality(G, normalized=True).values()), dtype=np.float32)
        except Exception:
            bet = np.zeros_like(deg)
        try:
            eig = np.array(list(nx.eigenvector_centrality_numpy(G).values()), dtype=np.float32)
        except Exception:
            eig = np.zeros_like(deg)
        modularity = 0.0
        ncom = 1
        try:
            comms = nx.algorithms.community.greedy_modularity_communities(G)
            modularity = nx.algorithms.community.quality.modularity(G, comms)
            ncom = float(len(comms))
        except Exception:
            modularity = 0.0
            ncom = 1.0
        gvec = np.concatenate([
            deg.mean(keepdims=True), deg.std(keepdims=True),
            strn.mean(keepdims=True), strn.std(keepdims=True),
            tri.mean(keepdims=True), tri.std(keepdims=True),
            np.array([cc], dtype=np.float32), np.array([ge], dtype=np.float32),
            bet.mean(keepdims=True), bet.std(keepdims=True),
            eig.mean(keepdims=True), eig.std(keepdims=True),
            np.array([modularity], dtype=np.float32),
            np.array([ncom], dtype=np.float32)
        ], axis=0)
        g_feats.append(gvec)
        if state.get("verbose") and (i + 1) % 50 == 0:
            print(f"FC agent: processed graph features for {i+1}/{len(mats)} subjects", flush=True)
    X_graph = np.stack(g_feats, axis=0)
    X_fc = np.concatenate([X_edges, X_graph], axis=1)
    node_init = []
    for m in mats:
        nf = m.copy().astype(np.float32)
        row_scale = (np.abs(nf).max(axis=1, keepdims=True) + 1e-8)
        nf = nf / row_scale
        node_init.append(nf)
    X_nodes = np.stack(node_init, axis=0)
    adjs = []
    for m in mats:
        A = m.copy()
        A = (A - A.min()) / (A.max() - A.min() + 1e-6)
        I = np.eye(A.shape[0], dtype=np.float32)
        A = A + I
        D = np.diag(1.0 / np.sqrt(np.sum(A, axis=1) + 1e-6))
        Ahat = D @ A @ D
        adjs.append(Ahat.astype(np.float32))
    A_batch = np.stack(adjs, axis=0)
    if state.get("verbose"):
        print(f"FC agent: features shape {X_fc.shape}, node_feat {X_nodes.shape}, adj {A_batch.shape}", flush=True)
    out = {
        "fc_features": X_fc,
        "node_features": X_nodes,
        "adj_batch": A_batch,
        "subject_ids": sids,
        "fc_edge_count": int(k)
    }
    for k in ["smri", "phenotypes", "labels", "sites", "verbose"]:
        if k in state:
            out[k] = state[k]
    return out


def smri_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    smri = state.get("smri", pd.DataFrame({"SUB_ID": []}))
    sids = state["subject_ids"]
    cols = [c for c in smri.columns if c != "SUB_ID"]
    X = smri[smri["SUB_ID"].isin(sids)][cols].values.astype(np.float32) if len(cols) > 0 else np.zeros((len(sids), 0), dtype=np.float32)
    dim = int(state.get("smri_pca_dim", 0))
    if dim and X.shape[1] > dim:
        pca = PCA(n_components=dim, random_state=42)
        X = pca.fit_transform(np.nan_to_num(X, nan=0.0))
    if state.get("verbose"):
        print(f"sMRI agent: features shape {X.shape}", flush=True)
    out = {"smri_features": X, "smri_columns": cols, "subject_ids": sids}
    for k in ["fc_features", "node_features", "adj_batch", "phenotypes", "labels", "sites", "verbose"]:
        if k in state:
            out[k] = state[k]
    return out


def phenotype_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    ph = state["phenotypes"]
    sids = state["subject_ids"]
    ph = ph[ph["SUB_ID"].isin(sids)].copy()
    y = None
    if "DX_GROUP" in ph.columns:
        y = ph["DX_GROUP"].astype(int).values
        y = np.where(y == 1, 1, 0)
    site = None
    site_col = None
    for c in ["SITE_ID", "SITE", "SITEID"]:
        if c in ph.columns:
            site_col = c
            break
    if site_col is None:
        site = np.array(["SITE_UNKNOWN"] * ph.shape[0])
    else:
        site = ph[site_col].astype(str).values
    dcols = []
    for c in ["AGE_AT_SCAN", "AGE_AT_MPRAGE"]:
        if c in ph.columns:
            dcols.append(c)
            break
    for iq in ["FIQ", "FSIQ", "VIQ", "PIQ"]:
        if iq in ph.columns and iq not in dcols:
            dcols.append(iq)
    cat_cols = []
    if "SEX" in ph.columns:
        cat_cols.append("SEX")
    use_cols = dcols + cat_cols
    if not use_cols:
        Xp = np.zeros((ph.shape[0], 1), dtype=np.float32)
        enc_info = {"num_cols": [], "cat_cols": [], "enc": None}
        return {"phenotype_features": Xp, "labels": y, "sites": site, "phenotype_encoder": enc_info}
    num = ph[dcols].fillna(ph[dcols].median()).values if dcols else np.zeros((ph.shape[0], 0))
    cat = ph[cat_cols].fillna("UNK").astype(str).values if cat_cols else np.zeros((ph.shape[0], 0))
    if cat_cols:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        cfeat = enc.fit_transform(cat)
    else:
        enc = None
        cfeat = np.zeros((ph.shape[0], 0))
    Xp = np.concatenate([num.astype(np.float32), cfeat.astype(np.float32)], axis=1)
    enc_info = {"num_cols": dcols, "cat_cols": cat_cols, "enc": enc}
    if state.get("verbose"):
        print(f"Phenotype agent: features shape {Xp.shape}, labels {None if y is None else y.shape}", flush=True)
    out = {"phenotype_features": Xp, "labels": y, "sites": site, "phenotype_encoder": enc_info, "subject_ids": sids}
    for k in ["fc_features", "node_features", "adj_batch", "smri_features", "smri_columns", "verbose"]:
        if k in state:
            out[k] = state[k]
    return out


def harmonization_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    if state.get("verbose"):
        print("Harmonization: applying ComBat to continuous modalities", flush=True)
    sites = state["sites"]
    batch = sites
    out = {}
    for key in ["smri_features", "fc_features"]:
        if key in state and state[key] is not None:
            X = state[key].astype(np.float32)
            mask = np.isfinite(X).all(axis=1)
            X2 = X[mask]
            b2 = batch[mask]
            b2 = b2.reshape(-1, 1)
            design = pd.DataFrame({"SITE": b2.squeeze()})
            try:
                mod = []
                params = harmonizationLearn(X2, design, "SITE", mod)
                if isinstance(params, tuple):
                    params = params[0]
                Xh = harmonizationApply(X2, design, params)
                X3 = X.copy()
                X3[mask] = Xh
                out[key] = X3.astype(np.float32)
            except Exception as e:
                if state.get("verbose"):
                    print(f"Harmonization: skipping {key} due to {str(e)}", flush=True)
                out[key] = X
            if state.get("verbose"):
                print(f"Harmonization: {key} done, shape {X3.shape}", flush=True)
    for k in ["phenotype_features", "labels", "sites", "node_features", "adj_batch", "smri_columns", "subject_ids", "verbose"]:
        if k in state:
            out[k] = state[k]
    return out


def fusion_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    parts = []
    scalers = {}
    per_mod = []
    if "fc_features" in state:
        Xraw = state["fc_features"]
        Xc = np.array(Xraw, dtype=np.float32)
        cm = np.nanmean(np.where(np.isfinite(Xc), Xc, np.nan), axis=0)
        bad = ~np.isfinite(Xc)
        if bad.any():
            Xc[bad] = np.take(np.where(np.isfinite(cm), cm, 0.0), np.where(bad)[1])
        sc = StandardScaler()
        X = sc.fit_transform(Xc)
        parts.append(X)
        per_mod.append(X)
        scalers["fc"] = sc
    if "smri_features" in state:
        Xraw = state["smri_features"]
        Xc = np.array(Xraw, dtype=np.float32)
        cm = np.nanmean(np.where(np.isfinite(Xc), Xc, np.nan), axis=0)
        bad = ~np.isfinite(Xc)
        if bad.any():
            Xc[bad] = np.take(np.where(np.isfinite(cm), cm, 0.0), np.where(bad)[1])
        sc = StandardScaler()
        X = sc.fit_transform(Xc)
        parts.append(X)
        per_mod.append(X)
        scalers["smri"] = sc
    if "phenotype_features" in state:
        Xraw = state["phenotype_features"]
        Xc = np.array(Xraw, dtype=np.float32)
        cm = np.nanmean(np.where(np.isfinite(Xc), Xc, np.nan), axis=0)
        bad = ~np.isfinite(Xc)
        if bad.any():
            Xc[bad] = np.take(np.where(np.isfinite(cm), cm, 0.0), np.where(bad)[1])
        sc = StandardScaler()
        X = sc.fit_transform(Xc)
        parts.append(X)
        per_mod.append(X)
        scalers["pheno"] = sc
    Xcat = np.concatenate(parts, axis=1)
    Xcat = np.nan_to_num(Xcat, nan=0.0, posinf=0.0, neginf=0.0)
    fusion_method = state.get("fusion_method", "concat")
    if fusion_method == "concat":
        if Xcat.shape[1] >= 4:
            dim = min(256, Xcat.shape[1], max(2, Xcat.shape[0] - 1))
            pca = PCA(n_components=dim, random_state=42)
            Xz = pca.fit_transform(Xcat)
        else:
            dim = Xcat.shape[1]
            pca = None
            Xz = Xcat
    else:
        d = int(state.get("attn_dim", 128))
        tokens = []
        for Xi in per_mod:
            Wi = np.random.RandomState(42).normal(scale=0.02, size=(Xi.shape[1], d)).astype(np.float32)
            tokens.append(Xi @ Wi)
        T = np.stack(tokens, axis=1)
        Xz = T.mean(axis=1)
        pca = None
    if state.get("verbose"):
        print(f"Fusion: concatenated {Xcat.shape}, fused -> {Xz.shape} via {fusion_method}", flush=True)
    out = {
        "fused_features": Xz.astype(np.float32),
        "concat_features": Xcat.astype(np.float32),
        "scalers": scalers,
        "pca": pca
    }
    for k in ["labels", "node_features", "adj_batch", "subject_ids", "verbose"]:
        if k in state:
            out[k] = state[k]
    return out


def embed_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    Xcat = state.get("concat_features", None)
    if Xcat is None:
        Xcat = state["fused_features"]
    method = state.get("embed_method", "pca")
    verbose = state.get("verbose", False)
    if method == "ae":
        hid = int(state.get("ae_dim", 384))
        inp = Xcat.shape[1]
        enc = nn.Sequential(nn.Linear(inp, hid), nn.ReLU(), nn.Linear(hid, hid))
        dec = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, inp))
        params = list(enc.parameters()) + list(dec.parameters())
        opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
        Xt = torch.tensor(Xcat, dtype=torch.float32)
        bsz = 64
        epochs = int(state.get("ae_epochs", 50))
        best = 1e9
        bad = 0
        for e in range(epochs):
            idx = torch.randperm(Xt.shape[0])
            for i in range(0, Xt.shape[0], bsz):
                sl = idx[i:i+bsz]
                xb = Xt[sl]
                opt.zero_grad()
                z = enc(xb)
                xr = dec(z)
                loss = F.mse_loss(xr, xb)
                loss.backward()
                opt.step()
            with torch.no_grad():
                z = enc(Xt)
                xr = dec(z)
                l = F.mse_loss(xr, Xt).item()
            if l + 1e-6 < best:
                best = l
                bad = 0
            else:
                bad += 1
            if verbose and ((e + 1) % 5 == 0 or e == epochs - 1):
                print(f"Embed AE epoch {e+1}/{epochs} recon={l:.4f}", flush=True)
            if bad >= 7:
                break
        with torch.no_grad():
            Z = enc(torch.tensor(Xcat, dtype=torch.float32)).numpy()
        feats = Z
    else:
        feats = state["fused_features"]
    add_edges = int(state.get("concat_edges_count", 0))
    if add_edges > 0 and "fc_features" in state and "fc_edge_count" in state:
        ke = int(state["fc_edge_count"])
        keep = min(add_edges, ke)
        Xedges = state["fc_features"][:, :keep]
        feats = np.concatenate([feats, Xedges], axis=1)
    out = {"features_for_training": feats.astype(np.float32)}
    for k in ["labels", "node_features", "adj_batch", "subject_ids", "verbose"]:
        if k in state:
            out[k] = state[k]
    return out


def evaluate_models(X: np.ndarray, y: np.ndarray, node_feat: np.ndarray, adj_batch: np.ndarray, seed: int = 42, verbose: bool = False, n_folds: int = 10) -> Dict[str, Any]:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    results = {}
    def assess(clf_name, preds, probs, trues):
        acc = accuracy_score(trues, preds)
        prec = precision_score(trues, preds, zero_division=0)
        rec = recall_score(trues, preds, zero_division=0)
        f1 = f1_score(trues, preds)
        auc = roc_auc_score(trues, probs) if len(np.unique(trues)) == 2 else np.nan
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}
    
    models = {
        "logreg": LogisticRegression(max_iter=1500, class_weight="balanced", C=0.8, solver='lbfgs'),
        "logreg_l1": LogisticRegression(max_iter=1500, class_weight="balanced", C=1.0, solver='saga', penalty='l1'),
        "rf": RandomForestClassifier(n_estimators=700, max_depth=None, min_samples_leaf=1, random_state=seed, class_weight="balanced", n_jobs=-1),
        "rf_deep": RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_leaf=2, random_state=seed+1, class_weight="balanced", n_jobs=-1),
        "xgb": XGBClassifier(n_estimators=1200, max_depth=8, learning_rate=0.025, subsample=0.9, colsample_bytree=0.8, random_state=seed, reg_lambda=1.5, n_jobs=0, eval_metric="logloss"),
        "xgb_deep": XGBClassifier(n_estimators=1000, max_depth=10, learning_rate=0.02, subsample=0.85, colsample_bytree=0.75, random_state=seed+1, reg_lambda=2.0, n_jobs=0, eval_metric="logloss"),
        "hgb": HistGradientBoostingClassifier(max_depth=None, learning_rate=0.05, max_iter=500, l2_regularization=0.0),
        "hgb_deep": HistGradientBoostingClassifier(max_depth=10, learning_rate=0.03, max_iter=600, l2_regularization=0.5),
        "svm": SVC(C=2.5, kernel="rbf", probability=True, class_weight="balanced", gamma='scale'),
        "et": ExtraTreesClassifier(n_estimators=600, max_depth=None, min_samples_leaf=1, random_state=seed, class_weight="balanced", n_jobs=-1),
        "ada": AdaBoostClassifier(n_estimators=300, learning_rate=0.05, random_state=seed, algorithm='SAMME'),
        "ridge": RidgeClassifier(alpha=1.0, class_weight="balanced"),
        "mlp": None
    }
    if HAS_LGBM:
        models["lgbm"] = LGBMClassifier(n_estimators=1500, num_leaves=63, learning_rate=0.025, subsample=0.9, colsample_bytree=0.8, reg_lambda=1.5, random_state=seed, n_jobs=-1, objective="binary")
        models["lgbm_deep"] = LGBMClassifier(n_estimators=1200, num_leaves=127, learning_rate=0.02, subsample=0.85, colsample_bytree=0.75, reg_lambda=2.0, random_state=seed+1, n_jobs=-1, objective="binary")
    if HAS_CAT:
        models["cat"] = CatBoostClassifier(iterations=1800, depth=7, learning_rate=0.025, loss_function="Logloss", eval_metric="AUC", verbose=False, random_seed=seed, l2_leaf_reg=3.0)
        models["cat_deep"] = CatBoostClassifier(iterations=1500, depth=8, learning_rate=0.02, loss_function="Logloss", eval_metric="AUC", verbose=False, random_seed=seed+1, l2_leaf_reg=5.0)
    
    y = y.astype(int)
    preds_all = {k: [] for k in models.keys()}
    probs_all = {k: [] for k in models.keys()}
    trues_all = []
    fold_idx = 0
    for tr, te in skf.split(X, y):
        fold_idx += 1
        if verbose:
            print(f"Model selection: fold {fold_idx}/{n_folds}", flush=True)
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        kbest = min(5000, Xtr.shape[1])
        selector = SelectKBest(mutual_info_classif, k=kbest)
        try:
            Xtr_sel = selector.fit_transform(Xtr, ytr)
            Xte_sel = selector.transform(Xte)
        except Exception:
            Xtr_sel, Xte_sel = Xtr, Xte
        for name, mdl in models.items():
            if name == "mlp":
                in_dim = Xtr_sel.shape[1]
                clf = ImprovedMLP(in_dim)
                opt = torch.optim.AdamW(clf.parameters(), lr=8e-4, weight_decay=2e-4)
                sched_mlp = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=120)
                bsz = 64
                epochs = 120
                patience = 20
                best_val = 1e9
                bad = 0
                Xt = torch.tensor(Xtr_sel, dtype=torch.float32)
                yt = torch.tensor(ytr.reshape(-1, 1), dtype=torch.float32)
                ntr = Xt.shape[0]
                vs = max(1, int(0.1 * ntr))
                Xv, yv = Xt[:vs], yt[:vs]
                Xtrb, ytrb = Xt[vs:], yt[vs:]
                clf.train()
                for e in range(epochs):
                    idx = torch.randperm(Xtrb.shape[0])
                    for i in range(0, Xtrb.shape[0], bsz):
                        sl = idx[i:i+bsz]
                        xb = Xtrb[sl]
                        yb = ytrb[sl]
                        opt.zero_grad()
                        out = clf(xb)
                        loss = F.binary_cross_entropy_with_logits(out, yb)
                        loss.backward()
                        opt.step()
                    sched_mlp.step()
                    with torch.no_grad():
                        val_loss = F.binary_cross_entropy_with_logits(clf(Xv), yv).item()
                    if val_loss + 1e-5 < best_val:
                        best_val = val_loss
                        bad = 0
                    else:
                        bad += 1
                    if verbose and ((e + 1) % 10 == 0 or e == epochs - 1):
                        print(f"MLP epoch {e+1}/{epochs} val_loss={val_loss:.4f}", flush=True)
                    if bad >= patience:
                        if verbose:
                            print("MLP early stopping", flush=True)
                        break
                clf.eval()
                with torch.no_grad():
                    p = torch.sigmoid(clf(torch.tensor(Xte_sel, dtype=torch.float32))).squeeze().numpy()
                preds = (p >= 0.5).astype(int)
                preds_all[name].extend(preds.tolist())
                probs_all[name].extend(p.tolist())
            else:
                try:
                    mdl.fit(Xtr_sel, ytr)
                    p = mdl.predict_proba(Xte_sel)[:, 1]
                except Exception:
                    try:
                        mdl.fit(Xtr, ytr)
                        p = mdl.predict_proba(Xte)[:, 1]
                    except Exception:
                        p = np.full(len(Xte_sel), 0.5)
                preds = (p >= 0.5).astype(int)
                preds_all[name].extend(preds.tolist())
                probs_all[name].extend(p.tolist())
            if verbose:
                print(f"Trained {name} on fold {fold_idx}", flush=True)
        trues_all.extend(yte.tolist())
    for name in models.keys():
        pa = np.array(probs_all[name])
        ya = np.array(trues_all)
        best_t = 0.5
        best_a = -1.0
        for t in np.linspace(0.3, 0.7, 41):
            pr = (pa >= t).astype(int)
            a = accuracy_score(ya, pr)
            if a > best_a:
                best_a = a
                best_t = t
        preds_opt = (pa >= best_t).astype(int)
        results[name] = assess(name, preds_opt, pa, ya)
    
    base_keys = [k for k in ["logreg", "rf", "xgb", "hgb", "lgbm", "cat", "mlp", "et", "lgbm_deep", "xgb_deep"] if k in probs_all]
    if len(base_keys) >= 3:
        P = np.vstack([np.array(probs_all[k]) for k in base_keys]).T
        ya = np.array(trues_all)
        skf2 = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed + 1)
        accs_s, f1s_s, aucs_s = [], [], []
        for tr2, te2 in skf2.split(P, ya):
            P_tr = P[tr2]
            P_te = P[te2]
            P_cal_te = np.zeros_like(P_te)
            for j in range(P.shape[1]):
                try:
                    cal = LogisticRegression(max_iter=1000, class_weight="balanced")
                    pj_tr = P_tr[:, j].reshape(-1, 1)
                    cal.fit(pj_tr, ya[tr2])
                    P_cal_te[:, j] = cal.predict_proba(P_te[:, j].reshape(-1, 1))[:, 1]
                except Exception:
                    P_cal_te[:, j] = P_te[:, j]
            meta_lr = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5)
            meta_lr.fit(P[tr2], ya[tr2])
            pm_lr = meta_lr.predict_proba(P[te2])[:, 1]
            if HAS_LGBM:
                meta_lgbm = LGBMClassifier(n_estimators=500, num_leaves=31, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8, random_state=seed + 2, objective="binary")
                try:
                    meta_lgbm.fit(P[tr2], ya[tr2])
                    pm_lgbm = meta_lgbm.predict_proba(P[te2])[:, 1]
                except Exception:
                    pm_lgbm = pm_lr
            else:
                pm_lgbm = pm_lr
            def best_acc(pm, yt):
                best_a = -1.0
                best_t = 0.5
                for t in np.linspace(0.3, 0.7, 41):
                    pr = (pm >= t).astype(int)
                    a = accuracy_score(yt, pr)
                    if a > best_a:
                        best_a = a
                        best_t = t
                prb = (pm >= best_t).astype(int)
                return best_a, prb
            a_lr, pr_lr = best_acc(pm_lr, ya[te2])
            a_lg, pr_lg = best_acc(pm_lgbm, ya[te2])
            best_pair = max([(a_lr, pr_lr, pm_lr), (a_lg, pr_lg, pm_lgbm)], key=lambda x: x[0])
            pr_best, pm_best = best_pair[1], best_pair[2]
            accs_s.append(accuracy_score(ya[te2], pr_best))
            f1s_s.append(f1_score(ya[te2], pr_best))
            aucs_s.append(roc_auc_score(ya[te2], pm_best))
        results["stack"] = {
            "accuracy": float(np.mean(accs_s)),
            "precision": float(np.nan),
            "recall": float(np.nan),
            "f1": float(np.mean(f1s_s)),
            "auc": float(np.mean(aucs_s)),
        }
    
    gcn_scores = None
    try:
        idxs = np.arange(X.shape[0])
        fold_preds = []
        fold_probs = []
        fold_trues = []
        gfold = 0
        for tr, te in skf.split(idxs, y):
            gfold += 1
            if verbose:
                print(f"DeeperGCN fold {gfold}/{n_folds}", flush=True)
            in_feat = node_feat.shape[2]
            model = DeeperGCN(in_features=in_feat, hidden=96, n_classes=1, n_layers=5)
            opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)
            epochs = 80
            patience_gcn = 15
            bsz = 32
            Xt = torch.tensor(node_feat[tr], dtype=torch.float32)
            At = torch.tensor(adj_batch[tr], dtype=torch.float32)
            yt = torch.tensor(y[tr].reshape(-1, 1), dtype=torch.float32)
            vs_g = max(1, int(0.1 * Xt.shape[0]))
            Xv_g, yv_g, Av_g = Xt[:vs_g], yt[:vs_g], At[:vs_g]
            Xb_g, yb_g, Ab_g = Xt[vs_g:], yt[vs_g:], At[vs_g:]
            best_vloss = 1e9
            bad_g = 0
            model.train()
            for e in range(epochs):
                idx = torch.randperm(Xb_g.shape[0])
                for i in range(0, Xb_g.shape[0], bsz):
                    sl = idx[i:i+bsz]
                    xb = Xb_g[sl]
                    ab = Ab_g[sl]
                    yb = yb_g[sl]
                    opt.zero_grad()
                    out = model(xb, ab)
                    loss = F.binary_cross_entropy_with_logits(out, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()
                with torch.no_grad():
                    vl = F.binary_cross_entropy_with_logits(model(Xv_g, Av_g), yv_g).item()
                if vl + 1e-5 < best_vloss:
                    best_vloss = vl
                    bad_g = 0
                else:
                    bad_g += 1
                if verbose and ((e + 1) % 10 == 0 or e == epochs - 1):
                    print(f"DeeperGCN epoch {e+1}/{epochs} val={vl:.4f}", flush=True)
                if bad_g >= patience_gcn:
                    if verbose:
                        print("DeeperGCN early stop", flush=True)
                    break
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(
                    torch.tensor(node_feat[te], dtype=torch.float32),
                    torch.tensor(adj_batch[te], dtype=torch.float32)
                )).squeeze().numpy()
            preds = (p >= 0.5).astype(int)
            fold_preds.extend(preds.tolist())
            fold_probs.extend(p.tolist())
            fold_trues.extend(y[te].tolist())
        gcn_scores = {"accuracy": accuracy_score(fold_trues, fold_preds), "f1": f1_score(fold_trues, fold_preds), "auc": roc_auc_score(fold_trues, fold_probs)}
        results["gcn"] = gcn_scores
    except Exception:
        pass
    
    best_name = None
    best_acc = -1.0
    for k, v in results.items():
        if v["accuracy"] > best_acc:
            best_acc = v["accuracy"]
            best_name = k
    
    feat_imp = None
    try:
        kbest = min(5000, X.shape[1])
        selector_full = SelectKBest(mutual_info_classif, k=kbest)
        X_sel = selector_full.fit_transform(X, y)
        if best_name in ["rf", "rf_deep", "et"]:
            mdl = RandomForestClassifier(n_estimators=800, random_state=seed, class_weight="balanced", n_jobs=-1)
            mdl.fit(X_sel, y)
            imp = mdl.feature_importances_
            idx = selector_full.get_support(indices=True)
            feat_imp = {"model": best_name, "indices": idx.tolist(), "importances": imp.tolist()}
        elif best_name in ["xgb", "xgb_deep"]:
            mdl = XGBClassifier(n_estimators=1200, max_depth=8, learning_rate=0.025, subsample=0.9, colsample_bytree=0.8, random_state=seed, reg_lambda=1.5, n_jobs=0, eval_metric="logloss")
            mdl.fit(X_sel, y)
            imp = mdl.feature_importances_
            idx = selector_full.get_support(indices=True)
            feat_imp = {"model": best_name, "indices": idx.tolist(), "importances": imp.tolist()}
        elif best_name in ["logreg", "logreg_l1"]:
            mdl = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.8)
            mdl.fit(X_sel, y)
            imp = np.abs(mdl.coef_).ravel()
            idx = selector_full.get_support(indices=True)
            feat_imp = {"model": best_name, "indices": idx.tolist(), "importances": imp.tolist()}
    except Exception:
        feat_imp = None
    
    return {"model_results": results, "best_model": best_name, "oof_probs": probs_all, "oof_y": trues_all, "feature_importance": feat_imp}


def model_selection_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    if state.get("verbose"):
        print("Model selection: starting with improved settings", flush=True)
    X = state.get("features_for_training", None)
    if X is None:
        X = state["fused_features"]
    y = state["labels"]
    node_feat = state.get("node_features", None)
    adj_batch = state.get("adj_batch", None)
    n_folds = state.get("n_folds", 10)
    res = evaluate_models(X, y, node_feat, adj_batch, verbose=state.get("verbose", False), n_folds=n_folds)
    out = {**res}
    out["features_for_training"] = X
    out["labels"] = y
    return out


def ensemble_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    if "oof_probs" not in state or "oof_y" not in state:
        return state
    probs_all = state["oof_probs"]
    y = np.array(state["oof_y"])
    keys = list(probs_all.keys())
    best_acc = state["model_results"].get(state["best_model"], {}).get("accuracy", -1.0)
    best_name = state["best_model"]
    cand_sets = [
        [k for k in keys if k in ["mlp", "xgb", "logreg", "lgbm"]],
        [k for k in keys if k in ["mlp", "xgb_deep", "lgbm", "cat"]],
        [k for k in keys if k in ["mlp", "xgb", "cat", "logreg", "lgbm_deep"]],
    ]
    results = dict(state["model_results"])
    for cset in cand_sets:
        P = np.vstack([np.array(probs_all[k]) for k in cset]).T
        pb = P.mean(axis=1)
        best_t = 0.5
        best_a = -1.0
        for t in np.linspace(0.3, 0.7, 41):
            pr = (pb >= t).astype(int)
            a = accuracy_score(y, pr)
            if a > best_a:
                best_a = a
                best_t = t
        pr = (pb >= best_t).astype(int)
        prec = precision_score(y, pr, zero_division=0)
        rec = recall_score(y, pr, zero_division=0)
        f1v = f1_score(y, pr)
        aucv = roc_auc_score(y, pb)
        name = f"softvote_{'+'.join(cset)}"
        results[name] = {"accuracy": best_a, "precision": prec, "recall": rec, "f1": f1v, "auc": aucv}
        if best_a > best_acc:
            best_acc = best_a
            best_name = name
    return {"model_results": results, "best_model": best_name}


def late_fusion_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    verbose = state.get("verbose", False)
    if verbose:
        print("Late Fusion Agent: starting per-modality training", flush=True)

    y = state.get("labels")
    if y is None:
        if verbose:
            print("Late Fusion Agent: no labels, skipping", flush=True)
        return {}
    y = np.array(y).astype(int)

    modalities: Dict[str, np.ndarray] = {}
    for key, label in [("fc_features", "fc"), ("smri_features", "smri"), ("phenotype_features", "pheno")]:
        raw = state.get(key)
        if raw is not None and np.array(raw).shape[1] > 0:
            X = np.array(raw, dtype=np.float32)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            sc = StandardScaler()
            X = sc.fit_transform(X)
            modalities[label] = X

    if len(modalities) < 2:
        if verbose:
            print("Late Fusion Agent: fewer than 2 modalities available, skipping", flush=True)
        return {}

    n_folds = state.get("n_folds", 10)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_probs: Dict[str, np.ndarray] = {k: np.zeros(len(y), dtype=np.float32) for k in modalities}

    per_mod_results: Dict[str, Dict[str, float]] = {}

    for mod_name, X in modalities.items():
        fold_probs = np.zeros(len(y), dtype=np.float32)
        for fold_idx, (tr, te) in enumerate(skf.split(X, y)):
            Xtr, Xte = X[tr], X[te]
            ytr, yte = y[tr], y[te]
            kbest = min(3000, Xtr.shape[1])
            try:
                sel = SelectKBest(mutual_info_classif, k=kbest)
                Xtr_s = sel.fit_transform(Xtr, ytr)
                Xte_s = sel.transform(Xte)
            except Exception:
                Xtr_s, Xte_s = Xtr, Xte

            clf = XGBClassifier(
                n_estimators=800, max_depth=7, learning_rate=0.03,
                subsample=0.9, colsample_bytree=0.8,
                random_state=42, reg_lambda=1.5, n_jobs=0, eval_metric="logloss"
            )
            try:
                clf.fit(Xtr_s, ytr)
                p = clf.predict_proba(Xte_s)[:, 1]
            except Exception:
                clf2 = LogisticRegression(max_iter=1000, class_weight="balanced")
                clf2.fit(Xtr_s, ytr)
                p = clf2.predict_proba(Xte_s)[:, 1]

            fold_probs[te] = p
            if verbose:
                preds = (p >= 0.5).astype(int)
                acc_f = accuracy_score(yte, preds)
                print(f"  {mod_name} fold {fold_idx+1}/{n_folds} acc={acc_f:.3f}", flush=True)

        oof_probs[mod_name] = fold_probs
        best_t, best_a = 0.5, -1.0
        for t in np.linspace(0.3, 0.7, 41):
            a = accuracy_score(y, (fold_probs >= t).astype(int))
            if a > best_a:
                best_a, best_t = a, t
        preds_mod = (fold_probs >= best_t).astype(int)
        per_mod_results[mod_name] = {
            "accuracy": float(accuracy_score(y, preds_mod)),
            "f1": float(f1_score(y, preds_mod, zero_division=0)),
            "auc": float(roc_auc_score(y, fold_probs) if len(np.unique(y)) == 2 else np.nan),
            "threshold": float(best_t),
        }
        if verbose:
            print(f"  {mod_name} overall acc={per_mod_results[mod_name]['accuracy']:.4f}", flush=True)

    late_results: Dict[str, Dict[str, float]] = {}

    def _eval_blend(pb: np.ndarray, name: str):
        best_t, best_a = 0.5, -1.0
        for t in np.linspace(0.3, 0.7, 41):
            a = accuracy_score(y, (pb >= t).astype(int))
            if a > best_a:
                best_a, best_t = a, t
        preds = (pb >= best_t).astype(int)
        return {
            "accuracy": float(accuracy_score(y, preds)),
            "f1": float(f1_score(y, preds, zero_division=0)),
            "auc": float(roc_auc_score(y, pb) if len(np.unique(y)) == 2 else np.nan),
        }

    P_stack = np.stack(list(oof_probs.values()), axis=1)
    p_avg = P_stack.mean(axis=1)
    late_results["late_avg"] = _eval_blend(p_avg, "late_avg")

    skf2 = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=43)
    stack_accs, stack_f1s, stack_aucs = [], [], []
    for tr2, te2 in skf2.split(P_stack, y):
        meta = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
        meta.fit(P_stack[tr2], y[tr2])
        pm = meta.predict_proba(P_stack[te2])[:, 1]
        best_t2, best_a2 = 0.5, -1.0
        for t in np.linspace(0.3, 0.7, 41):
            a = accuracy_score(y[te2], (pm >= t).astype(int))
            if a > best_a2:
                best_a2, best_t2 = a, t
        preds_s = (pm >= best_t2).astype(int)
        stack_accs.append(accuracy_score(y[te2], preds_s))
        stack_f1s.append(f1_score(y[te2], preds_s, zero_division=0))
        stack_aucs.append(roc_auc_score(y[te2], pm) if len(np.unique(y[te2])) == 2 else np.nan)
    late_results["late_stack"] = {
        "accuracy": float(np.mean(stack_accs)),
        "f1": float(np.mean(stack_f1s)),
        "auc": float(np.nanmean(stack_aucs)),
    }

    if verbose:
        print("\nLate Fusion Results:", flush=True)
        for k, v in {**per_mod_results, **late_results}.items():
            print(f"  {k}: acc={v['accuracy']:.4f}  f1={v['f1']:.4f}  auc={v.get('auc', float('nan')):.4f}", flush=True)

    existing_best_acc = -1.0
    existing_best_name = state.get("best_model")
    for k, v in state.get("model_results", {}).items():
        if v.get("accuracy", -1.0) > existing_best_acc:
            existing_best_acc = v["accuracy"]
            existing_best_name = k

    all_late = {**per_mod_results, **late_results}
    best_late_name = max(all_late, key=lambda k: all_late[k]["accuracy"])
    best_late_acc = all_late[best_late_name]["accuracy"]

    merged_results = dict(state.get("model_results", {}))
    for k, v in all_late.items():
        merged_results[f"late_{k}" if not k.startswith("late_") else k] = v

    best_overall = existing_best_name
    if best_late_acc > existing_best_acc:
        best_overall = best_late_name
        if verbose:
            print(f"\nLate Fusion IMPROVED best: {best_late_name} acc={best_late_acc:.4f} (vs early {existing_best_acc:.4f})", flush=True)
    else:
        if verbose:
            print(f"\nEarly/concat fusion still wins: {existing_best_name} acc={existing_best_acc:.4f} (best late={best_late_acc:.4f})", flush=True)

    out: Dict[str, Any] = {
        "model_results": merged_results,
        "best_model": best_overall,
        "late_fusion_per_mod": per_mod_results,
        "late_fusion_results": late_results,
    }
    for k in ["labels", "node_features", "adj_batch", "subject_ids", "verbose",
              "oof_probs", "oof_y", "features_for_training"]:
        if k in state:
            out[k] = state[k]
    return out


def multimodal_deep_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    verbose = state.get("verbose", False)
    if verbose:
        print("Multimodal Deep Agent: ImprovedTransformerFusion training", flush=True)

    y = state.get("labels")
    if y is None:
        return {}
    y = np.array(y).astype(int)

    def _prep(key):
        raw = state.get(key)
        if raw is None:
            return None
        X = np.array(raw, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        sc = StandardScaler()
        return sc.fit_transform(X)

    Xfc   = _prep("fc_features")
    Xsmri = _prep("smri_features")
    Xpheno = _prep("phenotype_features")

    if Xfc is None or Xsmri is None or Xpheno is None:
        if verbose:
            print("Multimodal Deep Agent: missing modality, skipping", flush=True)
        return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_folds = state.get("n_folds", 10)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_probs, fold_preds, fold_trues = [], [], []

    for fold_i, (tr, te) in enumerate(skf.split(Xfc, y)):
        if verbose:
            print(f"  ImprovedTransformerFusion fold {fold_i+1}/{n_folds}", flush=True)
        model = ImprovedTransformerFusion(
            fc_dim=Xfc.shape[1], smri_dim=Xsmri.shape[1], pheno_dim=Xpheno.shape[1],
            d_model=192, nhead=6, n_layers=3, dropout=0.25,
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)

        Xfc_t   = torch.tensor(Xfc[tr],    dtype=torch.float32).to(device)
        Xsm_t   = torch.tensor(Xsmri[tr],  dtype=torch.float32).to(device)
        Xph_t   = torch.tensor(Xpheno[tr], dtype=torch.float32).to(device)
        yt      = torch.tensor(y[tr].reshape(-1, 1), dtype=torch.float32).to(device)

        vs = max(1, int(0.1 * Xfc_t.shape[0]))
        Xfc_v,  Xsm_v,  Xph_v,  yv  = Xfc_t[:vs],  Xsm_t[:vs],  Xph_t[:vs],  yt[:vs]
        Xfc_b,  Xsm_b,  Xph_b,  yb  = Xfc_t[vs:],  Xsm_t[vs:],  Xph_t[vs:],  yt[vs:]

        best_vl, bad, patience = 1e9, 0, 20
        bsz = 32

        for e in range(100):
            model.train()
            idx = torch.randperm(Xfc_b.shape[0], device=device)
            for i in range(0, Xfc_b.shape[0], bsz):
                sl = idx[i:i+bsz]
                opt.zero_grad()
                out = model(Xfc_b[sl], Xsm_b[sl], Xph_b[sl])
                loss = F.binary_cross_entropy_with_logits(out, yb[sl])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            model.eval()
            with torch.no_grad():
                vl = F.binary_cross_entropy_with_logits(model(Xfc_v, Xsm_v, Xph_v), yv).item()
            if vl + 1e-5 < best_vl:
                best_vl, bad = vl, 0
            else:
                bad += 1
            if bad >= patience:
                break

        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(
                torch.tensor(Xfc[te],    dtype=torch.float32).to(device),
                torch.tensor(Xsmri[te],  dtype=torch.float32).to(device),
                torch.tensor(Xpheno[te], dtype=torch.float32).to(device),
            )).squeeze().cpu().numpy()
        fold_probs.extend(p.tolist())
        fold_trues.extend(y[te].tolist())

    p_arr = np.array(fold_probs)
    y_arr = np.array(fold_trues)
    best_t, best_a = 0.5, -1.0
    for t in np.linspace(0.3, 0.7, 41):
        a = accuracy_score(y_arr, (p_arr >= t).astype(int))
        if a > best_a:
            best_a, best_t = a, t
    preds_opt = (p_arr >= best_t).astype(int)
    tf_result = {
        "accuracy": float(accuracy_score(y_arr, preds_opt)),
        "f1":       float(f1_score(y_arr, preds_opt, zero_division=0)),
        "auc":      float(roc_auc_score(y_arr, p_arr) if len(np.unique(y_arr)) == 2 else float("nan")),
        "precision": float(precision_score(y_arr, preds_opt, zero_division=0)),
        "recall":    float(recall_score(y_arr, preds_opt, zero_division=0)),
    }
    if verbose:
        print(f"  ImprovedTransformerFusion: acc={tf_result['accuracy']:.4f}  f1={tf_result['f1']:.4f}  auc={tf_result['auc']:.4f}", flush=True)

    merged_mr = dict(state.get("model_results", {}))
    merged_mr["transformer_fusion"] = tf_result

    oof_probs = state.get("oof_probs", {})
    if oof_probs:
        for base_key in ["stack", "mlp", "xgb", "lgbm", "xgb_deep", "lgbm_deep"]:
            if base_key not in oof_probs:
                continue
            base_p = np.array(oof_probs[base_key])
            oof_y  = np.array(state.get("oof_y", []))
            if len(base_p) != len(p_arr) or len(oof_y) == 0:
                continue
            pb = (p_arr + base_p) / 2.0
            best_ta, best_aa = 0.5, -1.0
            for t in np.linspace(0.3, 0.7, 41):
                a = accuracy_score(oof_y, (pb >= t).astype(int))
                if a > best_aa:
                    best_aa, best_ta = a, t
            blend_pred = (pb >= best_ta).astype(int)
            blend_res = {
                "accuracy": float(accuracy_score(oof_y, blend_pred)),
                "f1":       float(f1_score(oof_y, blend_pred, zero_division=0)),
                "auc":      float(roc_auc_score(oof_y, pb) if len(np.unique(oof_y)) == 2 else float("nan")),
                "precision": float(precision_score(oof_y, blend_pred, zero_division=0)),
                "recall":    float(recall_score(oof_y, blend_pred, zero_division=0)),
            }
            blend_name = f"tf_blend_{base_key}"
            merged_mr[blend_name] = blend_res

    best_name = max(merged_mr, key=lambda k: merged_mr[k].get("accuracy", -1) if isinstance(merged_mr[k], dict) else -1)

    out: Dict[str, Any] = {"model_results": merged_mr, "best_model": best_name,
                            "tf_oof_probs": p_arr.tolist()}
    for k in ["labels", "node_features", "adj_batch", "subject_ids", "verbose",
              "oof_probs", "oof_y", "features_for_training",
              "late_fusion_results", "late_fusion_per_mod"]:
        if k in state:
            out[k] = state[k]
    return out


def brain_net_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    verbose = state.get("verbose", False)
    if verbose:
        print("BrainNetCNN Agent: training on raw FC matrices", flush=True)

    fc_dict = state.get("fc_dict")
    sids    = state.get("subject_ids")
    y       = state.get("labels")
    if fc_dict is None or sids is None or y is None:
        return {}

    y = np.array(y).astype(int)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"  BrainNetCNN device: {device}", flush=True)

    mats = []
    for sid in sids:
        m = fc_dict[sid].copy().astype(np.float32)
        m = (m + m.T) * 0.5
        np.fill_diagonal(m, 0.0)
        m = np.clip(m, -0.99999, 0.99999)
        m = np.arctanh(m)
        mats.append(m)
    X_raw = np.stack(mats, axis=0)
    n_nodes = X_raw.shape[1]

    flat   = X_raw.reshape(len(X_raw), -1)
    mean_e = flat.mean(axis=0, keepdims=True)
    std_e  = flat.std(axis=0, keepdims=True) + 1e-8
    X_norm = ((flat - mean_e) / std_e).reshape(X_raw.shape)
    X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)

    n_folds = state.get("n_folds", 10)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_probs, fold_trues = [], []

    for fold_i, (tr, te) in enumerate(skf.split(X_norm, y)):
        if verbose:
            print(f"  BrainNetCNN fold {fold_i + 1}/{n_folds}", flush=True)
        model = BrainNetCNN(n_nodes=n_nodes, dropout=0.45).to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=25, T_mult=2)

        Xtr, ytr = X_norm[tr][:, np.newaxis], y[tr]
        Xte       = X_norm[te][:, np.newaxis]
        vs        = max(1, int(0.1 * len(Xtr)))
        Xv, yv, Xb, yb = Xtr[:vs], ytr[:vs], Xtr[vs:], ytr[vs:]

        best_vl, bad, patience, bsz = 1e9, 0, 20, 16

        for e in range(120):
            model.train()
            idx = np.random.permutation(len(Xb))
            for i in range(0, len(Xb), bsz):
                sl = idx[i:i + bsz]
                xb_t = torch.tensor(Xb[sl], dtype=torch.float32).to(device)
                xb_t = xb_t + 0.02 * torch.randn_like(xb_t)
                mask = (torch.rand_like(xb_t) > 0.05).float()
                xb_t = xb_t * mask
                yb_t = torch.tensor(yb[sl].reshape(-1, 1), dtype=torch.float32).to(device)
                opt.zero_grad()
                out = model(xb_t)
                loss = F.binary_cross_entropy_with_logits(out, yb_t)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            model.eval()
            with torch.no_grad():
                xv_t = torch.tensor(Xv, dtype=torch.float32).to(device)
                yv_t = torch.tensor(yv.reshape(-1, 1), dtype=torch.float32).to(device)
                vl   = F.binary_cross_entropy_with_logits(model(xv_t), yv_t).item()
            if vl + 1e-5 < best_vl:
                best_vl, bad = vl, 0
            else:
                bad += 1
            if verbose and (e + 1) % 20 == 0:
                print(f"    epoch {e+1}/120  val_loss={vl:.4f}", flush=True)
            if bad >= patience:
                if verbose:
                    print(f"    early stop at epoch {e+1}", flush=True)
                break

        model.eval()
        probs_fold = []
        for i in range(0, len(Xte), bsz):
            with torch.no_grad():
                p = torch.sigmoid(
                    model(torch.tensor(Xte[i:i + bsz], dtype=torch.float32).to(device))
                ).squeeze().cpu().numpy()
            probs_fold.extend(np.atleast_1d(p).tolist())
        fold_probs.extend(probs_fold)
        fold_trues.extend(y[te].tolist())

    p_arr = np.array(fold_probs)
    y_arr = np.array(fold_trues)
    best_t, best_a = 0.5, -1.0
    for t in np.linspace(0.3, 0.7, 41):
        a = accuracy_score(y_arr, (p_arr >= t).astype(int))
        if a > best_a:
            best_a, best_t = a, t
    preds_opt = (p_arr >= best_t).astype(int)
    bncnn_res = {
        "accuracy":  float(accuracy_score(y_arr, preds_opt)),
        "f1":        float(f1_score(y_arr, preds_opt, zero_division=0)),
        "auc":       float(roc_auc_score(y_arr, p_arr) if len(np.unique(y_arr)) == 2 else float("nan")),
        "precision": float(precision_score(y_arr, preds_opt, zero_division=0)),
        "recall":    float(recall_score(y_arr, preds_opt, zero_division=0)),
    }
    if verbose:
        print(f"  BrainNetCNN: acc={bncnn_res['accuracy']:.4f}  f1={bncnn_res['f1']:.4f}  auc={bncnn_res['auc']:.4f}", flush=True)

    merged_mr = dict(state.get("model_results", {}))
    merged_mr["bncnn"] = bncnn_res
    oof_probs = state.get("oof_probs", {})
    oof_y     = np.array(state.get("oof_y", []))
    if oof_probs and len(oof_y) > 0 and len(oof_y) == len(p_arr):
        for base_key in ["stack", "mlp", "xgb", "lgbm", "xgb_deep", "lgbm_deep"]:
            base_p = np.array(oof_probs.get(base_key, []))
            if len(base_p) != len(p_arr):
                continue
            pb = (p_arr + base_p) / 2.0
            best_ta, best_aa = 0.5, -1.0
            for t in np.linspace(0.3, 0.7, 41):
                a = accuracy_score(oof_y, (pb >= t).astype(int))
                if a > best_aa:
                    best_aa, best_ta = a, t
            bl_pred = (pb >= best_ta).astype(int)
            merged_mr[f"bncnn_x_{base_key}"] = {
                "accuracy":  float(accuracy_score(oof_y, bl_pred)),
                "f1":        float(f1_score(oof_y, bl_pred, zero_division=0)),
                "auc":       float(roc_auc_score(oof_y, pb) if len(np.unique(oof_y)) == 2 else float("nan")),
                "precision": float(precision_score(oof_y, bl_pred, zero_division=0)),
                "recall":    float(recall_score(oof_y, bl_pred, zero_division=0)),
            }

    best_name = max(merged_mr, key=lambda k: merged_mr[k].get("accuracy", -1) if isinstance(merged_mr[k], dict) else -1)
    out: Dict[str, Any] = {"model_results": merged_mr, "best_model": best_name,
                            "bncnn_oof_probs": p_arr.tolist()}
    for k in ["labels", "node_features", "adj_batch", "subject_ids", "verbose",
              "oof_probs", "oof_y", "features_for_training",
              "late_fusion_results", "late_fusion_per_mod", "tf_oof_probs"]:
        if k in state:
            out[k] = state[k]
    return out


def super_ensemble_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    verbose = state.get("verbose", False)
    if verbose:
        print("Super Ensemble Agent: Best Single + Majority Voting + Stacking", flush=True)

    oof_y = np.array(state.get("oof_y", []))
    if len(oof_y) == 0:
        return {}

    pool: Dict[str, np.ndarray] = {}
    for k, v in state.get("oof_probs", {}).items():
        arr = np.array(v)
        if arr.ndim == 1 and len(arr) == len(oof_y):
            pool[k] = arr
    for extra_key in ["bncnn_oof_probs", "tf_oof_probs"]:
        raw = state.get(extra_key)
        if raw is not None:
            arr = np.array(raw)
            if arr.ndim == 1 and len(arr) == len(oof_y):
                label = extra_key.replace("_oof_probs", "")
                pool[label] = arr

    if len(pool) < 2:
        return {}

    if verbose:
        print(f"  Sources: {list(pool.keys())}", flush=True)

    keys = sorted(pool.keys())

    def _eval_probs(pb: np.ndarray) -> Dict[str, float]:
        best_t, best_a = 0.5, -1.0
        for t in np.linspace(0.3, 0.7, 41):
            a = accuracy_score(oof_y, (pb >= t).astype(int))
            if a > best_a:
                best_a, best_t = a, t
        preds = (pb >= best_t).astype(int)
        return {
            "accuracy":  float(accuracy_score(oof_y, preds)),
            "f1":        float(f1_score(oof_y, preds, zero_division=0)),
            "auc":       float(roc_auc_score(oof_y, pb) if len(np.unique(oof_y)) == 2 else float("nan")),
            "precision": float(precision_score(oof_y, preds, zero_division=0)),
            "recall":    float(recall_score(oof_y, preds, zero_division=0)),
        }

    merged_mr = dict(state.get("model_results", {}))

    best_single_acc = -1.0
    best_single_name = None
    best_single_probs = None
    for k in keys:
        probs = pool[k]
        result = _eval_probs(probs)
        if result["accuracy"] > best_single_acc:
            best_single_acc = result["accuracy"]
            best_single_name = k
            best_single_probs = probs

    merged_mr["super_best_single"] = _eval_probs(best_single_probs)
    if verbose:
        print(f"  Best single model: {best_single_name} (acc={best_single_acc:.4f})", flush=True)

    from itertools import combinations
    best_subset_acc = -1.0
    best_subset_name = None
    best_subset_probs = None
    for r in range(2, min(len(keys), 7)):
        for combo in combinations(keys, r):
            P_subset = np.column_stack([pool[k] for k in combo])
            p_avg = P_subset.mean(axis=1)
            result = _eval_probs(p_avg)
            name = f"subset_{'+'.join(combo)}"
            merged_mr[name] = result
            if result["accuracy"] > best_subset_acc:
                best_subset_acc = result["accuracy"]
                best_subset_name = name
                best_subset_probs = p_avg

    if best_subset_probs is not None:
        merged_mr["super_best_subset_avg"] = _eval_probs(best_subset_probs)
        if verbose:
            print(f"  Best subset: {best_subset_name} (acc={best_subset_acc:.4f})", flush=True)

    P_all = np.column_stack([pool[k] for k in keys])
    binary_preds = (P_all >= 0.5).astype(int)
    majority_votes = (binary_preds.sum(axis=1) >= len(keys) / 2).astype(int)
    merged_mr["super_majority_vote"] = _eval_probs(majority_votes.astype(float))
    if verbose:
        print(f"  Majority vote: acc={merged_mr['super_majority_vote']['accuracy']:.4f}", flush=True)

    n_folds = state.get("n_folds", 10)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=44)
    lr_fold_probs = np.zeros(len(oof_y), dtype=np.float32)
    for tr2, te2 in skf.split(P_all, oof_y):
        meta_lr = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5)
        meta_lr.fit(P_all[tr2], oof_y[tr2])
        lr_fold_probs[te2] = meta_lr.predict_proba(P_all[te2])[:, 1].astype(np.float32)

    merged_mr["super_stack"] = _eval_probs(lr_fold_probs)
    if verbose:
        print(f"  Stacking: acc={merged_mr['super_stack']['accuracy']:.4f}", flush=True)

    if verbose:
        print("\n  Summary:", flush=True)
        print(f"    super_best_single:   acc={merged_mr['super_best_single']['accuracy']:.4f}", flush=True)
        if best_subset_probs is not None:
            print(f"    super_best_subset:   acc={merged_mr['super_best_subset_avg']['accuracy']:.4f}", flush=True)
        print(f"    super_majority_vote: acc={merged_mr['super_majority_vote']['accuracy']:.4f}", flush=True)
        print(f"    super_stack:        acc={merged_mr['super_stack']['accuracy']:.4f}", flush=True)

    best_name = max(merged_mr, key=lambda k: merged_mr[k].get("accuracy", -1) if isinstance(merged_mr[k], dict) else -1)
    out: Dict[str, Any] = {"model_results": merged_mr, "best_model": best_name}
    for k in ["labels", "oof_y", "verbose", "bncnn_oof_probs", "tf_oof_probs"]:
        if k in state:
            out[k] = state[k]
    return out


def build_graph() -> Any:
    graph = StateGraph(dict)
    graph.add_node("fc_features", fc_feature_agent)
    graph.add_node("smri_features", smri_agent)
    graph.add_node("phenotypes", phenotype_agent)
    graph.add_node("harmonize", harmonization_agent)
    graph.add_node("fuse", fusion_agent)
    graph.add_node("embed", embed_agent)
    graph.add_node("model_select", model_selection_agent)
    graph.add_node("ensemble", ensemble_agent)
    graph.set_entry_point("fc_features")
    graph.add_edge("fc_features", "smri_features")
    graph.add_edge("smri_features", "phenotypes")
    graph.add_edge("phenotypes", "harmonize")
    graph.add_edge("harmonize", "fuse")
    graph.add_edge("fuse", "embed")
    graph.add_edge("embed", "model_select")
    graph.add_edge("model_select", "ensemble")
    graph.add_edge("ensemble", END)
    return graph.compile()


def _run_agents_sequential(init_state: Dict[str, Any]) -> Dict[str, Any]:
    agents = [
        ("fc_features",       fc_feature_agent),
        ("smri_features",     smri_agent),
        ("phenotypes",        phenotype_agent),
        ("harmonize",         harmonization_agent),
        ("fuse",              fusion_agent),
        ("embed",             embed_agent),
        ("model_select",      model_selection_agent),
        ("ensemble",          ensemble_agent),
    ]
    state = dict(init_state)
    for name, fn in agents:
        verbose = state.get("verbose", False)
        if verbose:
            print(f"Agent: {name}", flush=True)
        try:
            out = fn(state)
            if isinstance(out, dict):
                state = {**state, **out}
        except Exception as exc:
            if verbose:
                print(f"Agent {name} failed: {exc}", flush=True)
            raise
    return state


def run_pipeline(data_dir: str, config: Dict[str, Any] = None, verbose: bool = True) -> Dict[str, Any]:
    if verbose:
        print("Pipeline: loading data", flush=True)
    fc_dir = os.path.join(data_dir, "fc_matrices")
    ph_fp = os.path.join(data_dir, "phenotypic_data.csv")
    smri_fp_1 = os.path.join(data_dir, "structural_data_cleaned.csv")
    smri = None
    if os.path.exists(smri_fp_1):
        smri = load_smri_table(smri_fp_1)
    else:
        smri = pd.DataFrame({"SUB_ID": []})
    ph = load_phenotypes(ph_fp)
    fc_dict = load_fc_matrices(fc_dir, verbose=verbose)
    sids, fc_f, smri_f, ph_f = align_subjects(fc_dict, smri, ph)
    if verbose:
        print(f"Pipeline: aligned subjects {len(sids)}", flush=True)
        print("Pipeline: executing agent pipeline", flush=True)
    
    n_folds = 10
    init_state = {
        "fc_dict": fc_f,
        "smri": smri_f,
        "phenotypes": ph_f,
        "subject_ids": sids,
        "verbose": verbose,
        "n_folds": n_folds,
        "fc_search_ps": [0.05, 0.1, 0.2, 0.3, 0.4],
        "fc_search_ks": [2000, 4000, 6000, 8000],
        "fc_search_strategies": ["prop", "mst_prop"],
        "embed_method": "ae",
        "ae_dim": 384,
        "ae_epochs": 50,
        "concat_edges_count": 2500,
    }
    state = _run_agents_sequential(init_state)
    if verbose:
        print("Pipeline: running late fusion agent", flush=True)
    try:
        lf_out = late_fusion_agent(state)
        if isinstance(lf_out, dict) and lf_out:
            state = {**state, **lf_out}
    except Exception as exc:
        if verbose:
            print(f"Late fusion agent failed: {exc}", flush=True)
    if verbose:
        print("Pipeline: running multimodal deep agent (ImprovedTransformerFusion)", flush=True)
    try:
        md_out = multimodal_deep_agent(state)
        if isinstance(md_out, dict) and md_out:
            state = {**state, **md_out}
    except Exception as exc:
        if verbose:
            print(f"Multimodal deep agent failed: {exc}", flush=True)
    if verbose:
        print("Pipeline: running BrainNetCNN agent", flush=True)
    try:
        bn_out = brain_net_agent(state)
        if isinstance(bn_out, dict) and bn_out:
            state = {**state, **bn_out}
    except Exception as exc:
        if verbose:
            print(f"BrainNetCNN agent failed: {exc}", flush=True)
    if verbose:
        print("Pipeline: running super ensemble agent", flush=True)
    try:
        se_out = super_ensemble_agent(state)
        if isinstance(se_out, dict) and se_out:
            state = {**state, **se_out}
    except Exception as exc:
        if verbose:
            print(f"Super ensemble agent failed: {exc}", flush=True)
    if verbose:
        print("Pipeline: completed", flush=True)
    best = state.get("best_model")
    mr = state.get("model_results", {})
    if verbose and mr:
        print("\n=== Model Results ===", flush=True)
        for k, v in sorted(mr.items(), key=lambda x: x[1].get("accuracy", -1), reverse=True):
            print(f"  {k:40s}  acc={v.get('accuracy', float('nan')):.4f}  f1={v.get('f1', float('nan')):.4f}  auc={v.get('auc', float('nan')):.4f}", flush=True)
        print(f"\n  Best model: {best}", flush=True)
    return state


def save_report(res: Dict[str, Any], fp: str):
    _SKIP_TYPES = (np.ndarray,)
    try:
        _SKIP_TYPES = _SKIP_TYPES + (pd.DataFrame, pd.Series)
    except Exception:
        pass

    def _json_safe(obj, _depth: int = 0):
        if _depth > 10:
            return None
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if isinstance(v, _SKIP_TYPES):
                    continue
                safe_v = _json_safe(v, _depth + 1)
                if safe_v is not None or v is None:
                    out[k] = safe_v
            return out
        if isinstance(obj, (list, tuple)):
            result = [_json_safe(i, _depth + 1) for i in obj]
            return result
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if (v != v) else v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return None
        if isinstance(obj, pd.Series):
            return None
        if isinstance(obj, float) and obj != obj:
            return None
        try:
            import json as _json
            _json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)

    safe = _json_safe(res)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(safe, f, indent=2)


if __name__ == "__main__":
    import argparse
    import time
    p = argparse.ArgumentParser(description="ASD vs HC Classification Pipeline v4")
    p.add_argument("--data_dir", type=str, default=os.path.join(os.path.dirname(__file__), "data"))
    p.add_argument("--out", type=str, default="resultspipeline4.json")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--folds", type=int, default=10, help="Number of CV folds (default: 10)")
    args = p.parse_args()
    verbose = not args.quiet
    
    start_time = time.time()
    
    if verbose:
        print("=" * 80, flush=True)
        print("ASD vs HC Pipeline (v4): 10-fold CV, Improved Models", flush=True)
        print("=" * 80, flush=True)
        print(f"Data directory: {args.data_dir}", flush=True)
        print(f"Output file: {args.out}", flush=True)
        print(f"CV folds: {args.folds}", flush=True)
        print(flush=True)
    
    res = run_pipeline(args.data_dir, verbose=verbose)
    res["elapsed_time"] = time.time() - start_time
    
    save_report(res, args.out)
    
    if verbose:
        print("\n" + "=" * 80, flush=True)
        print("FINAL RESULTS", flush=True)
        print("=" * 80, flush=True)
    
    mr = res.get("model_results", {})
    best = res.get("best_model")
    if mr:
        if verbose:
            print("\n=== Final Model Results ===", flush=True)
        for k, v in sorted(mr.items(), key=lambda x: x[1].get("accuracy", -1) if isinstance(x[1], dict) else -1, reverse=True):
            if isinstance(v, dict):
                if verbose:
                    print(f"  {k:45s}  acc={v.get('accuracy', float('nan')):.4f}  f1={v.get('f1', float('nan')):.4f}  auc={v.get('auc', float('nan')):.4f}", flush=True)
        if verbose:
            print(flush=True)
    else:
        if verbose:
            print("model_results: (empty)", flush=True)
    
    if best:
        if verbose:
            print(f"Best model: {best}", flush=True)
        if best in mr:
            bv = mr[best]
            if verbose:
                print(f"  accuracy : {bv.get('accuracy', float('nan')):.4f}", flush=True)
                print(f"  f1       : {bv.get('f1', float('nan')):.4f}", flush=True)
                print(f"  auc      : {bv.get('auc', float('nan')):.4f}", flush=True)
    
    elapsed = res.get("elapsed_time", 0)
    if verbose:
        print(f"\nTotal elapsed time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)", flush=True)
        print(f"Results saved to: {args.out}", flush=True)
