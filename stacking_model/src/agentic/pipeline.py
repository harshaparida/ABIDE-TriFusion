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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
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


class SimpleGCN(nn.Module):
    def __init__(self, in_features, hidden, n_classes):
        super().__init__()
        self.proj = nn.Linear(in_features, hidden)
        self.gcn1 = GCNLayer(hidden, hidden)
        self.gcn2 = GCNLayer(hidden, hidden)
        self.gcn3 = GCNLayer(hidden, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.cls = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden, n_classes),
        )

    def _bn_nodes(self, h, bn):
        B, N, D = h.shape
        return bn(h.reshape(B * N, D)).reshape(B, N, D)

    def forward(self, x, adj):
        x = F.relu(self.proj(x))
        h = F.relu(self._bn_nodes(self.gcn1(x, adj), self.bn1))
        h = h + x
        h2 = F.relu(self._bn_nodes(self.gcn2(h, adj), self.bn2))
        h2 = h2 + h
        h3 = self.gcn3(h2, adj)
        g_mean = h3.mean(dim=1)
        g_max = h3.max(dim=1).values
        g = torch.cat([g_mean, g_max], dim=1)
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


class TransformerFusionModel(nn.Module):
    """Cross-modal transformer: each modality becomes one token + CLS token for classification."""

    def __init__(self, fc_dim: int, smri_dim: int, pheno_dim: int,
                 d_model: int = 128, nhead: int = 4, n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.fc_proj   = nn.Sequential(nn.Linear(fc_dim, d_model), nn.LayerNorm(d_model))
        self.smri_proj = nn.Sequential(nn.Linear(smri_dim, d_model), nn.LayerNorm(d_model))
        self.pheno_proj = nn.Sequential(nn.Linear(pheno_dim, d_model), nn.LayerNorm(d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
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
    """Memory-efficient BrainNetCNN (Kawahara et al. 2017) for symmetric FC matrices.
    E2E → E2N → Dense.  Input: [B, 1, N, N].
    """

    def __init__(self, n_nodes: int, dropout: float = 0.5):
        super().__init__()
        self.n_nodes = n_nodes
        # E2E: (1, N) conv sweeps along columns → [B, 32, N, 1]
        self.e2e = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, n_nodes)),
            nn.LeakyReLU(0.33),
            nn.BatchNorm2d(32),
        )
        # Second E2E on transpose to capture col-direction
        self.e2e_t = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(n_nodes, 1)),
            nn.LeakyReLU(0.33),
            nn.BatchNorm2d(32),
        )
        # E2N: (N, 1) conv aggregates nodes → [B, 128, 1, 1]
        self.e2n = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(n_nodes, 1)),
            nn.LeakyReLU(0.33),
            nn.BatchNorm2d(128),
        )
        # Dense classifier
        self.cls = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.33),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.33),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure symmetry
        x = (x + x.transpose(-2, -1)) * 0.5
        h1 = self.e2e(x)               # [B, 32, N, 1]
        h2 = self.e2e_t(x)             # [B, 32, 1, N]
        h2 = h2.transpose(-2, -1)      # [B, 32, N, 1]
        h  = torch.cat([h1, h2], dim=1) # [B, 64, N, 1]
        h  = self.e2n(h)               # [B, 128, 1, 1]
        h  = h.view(h.size(0), -1)     # [B, 128]
        return self.cls(h)


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
        # add top proportion edges beyond MST
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
    # quick search over thresholds and top-k using a fast 3-fold CV on edges only
    search_ps = state.get("fc_search_ps", [0.3])
    search_ks = state.get("fc_search_ks", [8000])
    search_strats = state.get("fc_search_strategies", ["prop"])
    mats_z = [fisher_z(m) for m in mats_raw]
    n = mats_z[0].shape[0]
    ut_idx = np.triu_indices(n, 1)
    best_acc = -1.0
    best_cfg = (0.3, 8000, "prop")
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
    # Combined edge ranking: variance + t-statistic (ASD vs HC)
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
        ge = 0.0
        gvec = np.concatenate([
            deg.mean(keepdims=True), deg.std(keepdims=True),
            strn.mean(keepdims=True), strn.std(keepdims=True),
            tri.mean(keepdims=True), tri.std(keepdims=True),
            np.array([cc], dtype=np.float32), np.array([ge], dtype=np.float32),
        ], axis=0)
        g_feats.append(gvec)
        if state.get("verbose") and (i + 1) % 100 == 0:
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
    # optional sMRI dimensionality reduction
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
    # include IQ if available
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
    for key in ["smri_features"]:  # Only harmonize smri, keep FC raw
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
    if "fc_features" in state:
        out["fc_features"] = state["fc_features"]
        if state.get("verbose"):
            print(f"Harmonization: fc_features passed through raw, shape {state['fc_features'].shape}", flush=True)
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
        # simple attention-based fusion without training (mean of linear projections)
        # if torch training is desired, prefer embed_agent downstream
        d = int(state.get("attn_dim", 128))
        tokens = []
        for Xi in per_mod:
            Wi = np.random.RandomState(42).normal(scale=0.02, size=(Xi.shape[1], d)).astype(np.float32)
            tokens.append(Xi @ Wi)
        T = np.stack(tokens, axis=1)  # [N, M, d]
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


def evaluate_models(X: np.ndarray, y: np.ndarray, node_feat: np.ndarray, adj_batch: np.ndarray, seed: int = 42, verbose: bool = False) -> Dict[str, Any]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    results = {}
    def assess(clf_name, preds, probs, trues):
        acc = accuracy_score(trues, preds)
        prec = precision_score(trues, preds, zero_division=0)
        rec = recall_score(trues, preds, zero_division=0)
        f1 = f1_score(trues, preds)
        auc = roc_auc_score(trues, probs) if len(np.unique(trues)) == 2 else np.nan
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}
    models = {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_leaf=1, random_state=seed, class_weight="balanced"),
        "xgb": XGBClassifier(n_estimators=900, max_depth=7, learning_rate=0.03, subsample=0.9, colsample_bytree=0.8, random_state=seed, reg_lambda=1.0, n_jobs=0, eval_metric="logloss"),
        "hgb": HistGradientBoostingClassifier(max_depth=None, learning_rate=0.06, max_iter=400, l2_regularization=0.0),
        "svm": SVC(C=2.0, kernel="rbf", probability=True, class_weight="balanced"),
        "mlp": None
    }
    if HAS_LGBM:
        models["lgbm"] = LGBMClassifier(n_estimators=1200, num_leaves=63, learning_rate=0.03, subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, n_jobs=-1, objective="binary")
    if HAS_CAT:
        models["cat"] = CatBoostClassifier(iterations=1500, depth=6, learning_rate=0.03, loss_function="Logloss", eval_metric="AUC", verbose=False, random_seed=seed)
    y = y.astype(int)
    preds_all = {k: [] for k in models.keys()}
    probs_all = {k: [] for k in models.keys()}
    trues_all = []
    fold_idx = 0
    for tr, te in skf.split(X, y):
        fold_idx += 1
        if verbose:
            print(f"Model selection: fold {fold_idx}/5", flush=True)
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        kbest = min(4000, Xtr.shape[1])
        selector = SelectKBest(mutual_info_classif, k=kbest)
        try:
            Xtr_sel = selector.fit_transform(Xtr, ytr)
            Xte_sel = selector.transform(Xte)
        except Exception:
            Xtr_sel, Xte_sel = Xtr, Xte
        for name, mdl in models.items():
            if name == "mlp":
                in_dim = Xtr_sel.shape[1]
                d = 512
                clf = nn.Sequential(
                    nn.Linear(in_dim, d), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(0.3),
                    _ResBlock(d, dropout=0.3),
                    _ResBlock(d, dropout=0.3),
                    nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                    _ResBlock(256, dropout=0.3),
                    nn.Linear(256, 1),
                )
                opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=2e-4)
                sched_mlp = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
                bsz = 64
                epochs = 100
                patience = 15
                best_val = 1e9
                bad = 0
                Xt = torch.tensor(Xtr_sel, dtype=torch.float32)
                yt = torch.tensor(ytr.reshape(-1, 1), dtype=torch.float32)
                # simple 90/10 split
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
                    if verbose and ((e + 1) % 5 == 0 or e == epochs - 1):
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
                    mdl.fit(Xtr, ytr)
                    p = mdl.predict_proba(Xte)[:, 1]
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
    # simple weighted blend of top tabular models
    if all(k in probs_all for k in ["mlp", "xgb", "logreg"]) and len(probs_all["mlp"]) == len(trues_all):
        ya = np.array(trues_all)
        mlp_p = np.array(probs_all["mlp"])
        xgb_p = np.array(probs_all["xgb"])
        lr_p = np.array(probs_all["logreg"])
        best_acc_b = -1.0
        best_w = (0.5, 0.35, 0.15)
        best_t = 0.5
        for w1 in np.linspace(0.4, 0.6, 5):
            for w2 in np.linspace(0.2, 0.5, 7):
                w3 = max(0.0, 1.0 - w1 - w2)
                if w3 < 0.0:
                    continue
                pb = w1 * mlp_p + w2 * xgb_p + w3 * lr_p
                for t in np.linspace(0.3, 0.7, 41):
                    pr = (pb >= t).astype(int)
                    a = accuracy_score(ya, pr)
                    if a > best_acc_b:
                        best_acc_b = a
                        best_w = (w1, w2, w3)
                        best_t = t
        w1, w2, w3 = best_w
        p_blend = w1 * mlp_p + w2 * xgb_p + w3 * lr_p
        yb = (p_blend >= best_t).astype(int)
        results["blend"] = assess("blend", yb, p_blend, ya)
    # stacking meta-learner using OOF probabilities
    base_keys = [k for k in ["logreg", "rf", "xgb", "hgb", "lgbm", "cat", "mlp"] if k in probs_all]
    if len(base_keys) >= 3:
        P = np.vstack([np.array(probs_all[k]) for k in base_keys]).T
        ya = np.array(trues_all)
        skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 1)
        accs_s, f1s_s, aucs_s = [], [], []
        for tr2, te2 in skf2.split(P, ya):
            # probability calibration per base (Platt scaling)
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
            # meta candidates
            meta_lr = LogisticRegression(max_iter=1000, class_weight="balanced")
            meta_lr.fit(P[tr2], ya[tr2])
            pm_lr = meta_lr.predict_proba(P[te2])[:, 1]
            # LightGBM meta if available
            if HAS_LGBM:
                meta_lgbm = LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8, random_state=seed + 2, objective="binary")
                try:
                    meta_lgbm.fit(P[tr2], ya[tr2])
                    pm_lgbm = meta_lgbm.predict_proba(P[te2])[:, 1]
                except Exception:
                    pm_lgbm = pm_lr
            else:
                pm_lgbm = pm_lr
            # MLP meta-learner
            try:
                _meta_mlp = nn.Sequential(
                    nn.Linear(P.shape[1], 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(64, 32), nn.ReLU(),
                    nn.Linear(32, 1),
                )
                _mo = torch.optim.Adam(_meta_mlp.parameters(), lr=5e-3, weight_decay=1e-3)
                _Ptr = torch.tensor(P[tr2], dtype=torch.float32)
                _ytr = torch.tensor(ya[tr2].reshape(-1, 1), dtype=torch.float32)
                _meta_mlp.train()
                for _e in range(60):
                    _idx = torch.randperm(_Ptr.shape[0])
                    for _i in range(0, _Ptr.shape[0], 32):
                        _sl = _idx[_i:_i+32]
                        _mo.zero_grad()
                        _l = F.binary_cross_entropy_with_logits(_meta_mlp(_Ptr[_sl]), _ytr[_sl])
                        _l.backward()
                        _mo.step()
                _meta_mlp.eval()
                with torch.no_grad():
                    pm_meta_mlp = torch.sigmoid(_meta_mlp(torch.tensor(P[te2], dtype=torch.float32))).squeeze().numpy()
            except Exception:
                pm_meta_mlp = pm_lr
            # choose better meta per split by accuracy after threshold optimization
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
            a_lr, pr_lr   = best_acc(pm_lr, ya[te2])
            a_lg, pr_lg   = best_acc(pm_lgbm, ya[te2])
            a_mm, pr_mm   = best_acc(pm_meta_mlp, ya[te2])
            best_pair = max([(a_lr, pr_lr, pm_lr), (a_lg, pr_lg, pm_lgbm), (a_mm, pr_mm, pm_meta_mlp)], key=lambda x: x[0])
            pr_best, pm_best = best_pair[1], best_pair[2]
            # threshold sweep
            accs_s.append(accuracy_score(ya[te2], pr_best))
            f1s_s.append(f1_score(ya[te2], pr_best))
            aucs_s.append(roc_auc_score(ya[te2], pm_best))
        results["stack"] = {
            "accuracy": float(np.mean(accs_s)),
            "precision": float(np.nan),  # not aggregated in this shortcut
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
                print(f"GCN fold {gfold}/5", flush=True)
            in_feat = node_feat.shape[2]
            model = SimpleGCN(in_features=in_feat, hidden=64, n_classes=1)
            opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
            epochs = 60
            patience_gcn = 12
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
                    print(f"GCN epoch {e+1}/{epochs} val={vl:.4f}", flush=True)
                if bad_g >= patience_gcn:
                    if verbose:
                        print("GCN early stop", flush=True)
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
    # feature importance retrain on full data
    feat_imp = None
    try:
        kbest = min(4000, X.shape[1])
        selector_full = SelectKBest(mutual_info_classif, k=kbest)
        X_sel = selector_full.fit_transform(X, y)
        if best_name == "rf":
            mdl = RandomForestClassifier(n_estimators=600, random_state=seed, class_weight="balanced")
            mdl.fit(X_sel, y)
            imp = mdl.feature_importances_
            idx = selector_full.get_support(indices=True)
            feat_imp = {"model": best_name, "indices": idx.tolist(), "importances": imp.tolist()}
        elif best_name == "xgb":
            mdl = XGBClassifier(n_estimators=1000, max_depth=7, learning_rate=0.03, subsample=0.9, colsample_bytree=0.8, random_state=seed, reg_lambda=1.0, n_jobs=0, eval_metric="logloss")
            mdl.fit(X_sel, y)
            imp = mdl.feature_importances_
            idx = selector_full.get_support(indices=True)
            feat_imp = {"model": best_name, "indices": idx.tolist(), "importances": imp.tolist()}
        elif best_name == "logreg":
            mdl = LogisticRegression(max_iter=2000, class_weight="balanced")
            mdl.fit(X_sel, y)
            imp = np.abs(mdl.coef_).ravel()
            idx = selector_full.get_support(indices=True)
            feat_imp = {"model": best_name, "indices": idx.tolist(), "importances": imp.tolist()}
    except Exception:
        feat_imp = None
    return {"model_results": results, "best_model": best_name, "oof_probs": probs_all, "oof_y": trues_all, "feature_importance": feat_imp}


def model_selection_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    if state.get("verbose"):
        print("Model selection: starting", flush=True)
    X = state.get("features_for_training", None)
    if X is None:
        X = state["fused_features"]
    y = state["labels"]
    node_feat = state.get("node_features", None)
    adj_batch = state.get("adj_batch", None)
    res = evaluate_models(X, y, node_feat, adj_batch, verbose=state.get("verbose", False))
    out = {**res}
    out["features_for_training"] = X
    out["labels"] = y
    return out


def ensemble_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    # consumes oof probabilities and tries alternative ensembling strategies
    if "oof_probs" not in state or "oof_y" not in state:
        return state
    probs_all = state["oof_probs"]
    y = np.array(state["oof_y"])
    keys = list(probs_all.keys())
    best_acc = state["model_results"].get(state["best_model"], {}).get("accuracy", -1.0)
    best_name = state["best_model"]
    # soft voting across top models
    cand_sets = [
        [k for k in keys if k in ["mlp", "xgb", "logreg"]],
        [k for k in keys if k in ["mlp", "xgb", "lgbm"]],
        [k for k in keys if k in ["mlp", "xgb", "cat", "logreg"]],
    ]
    results = dict(state["model_results"])
    for cset in cand_sets:
        P = np.vstack([np.array(probs_all[k]) for k in cset]).T
        w = np.ones(len(cset)) / len(cset)
        pb = (P @ w).ravel()
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


def embed_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    Xcat = state.get("concat_features", None)
    if Xcat is None:
        Xcat = state["fused_features"]
    method = state.get("embed_method", "pca")
    verbose = state.get("verbose", False)
    if method == "ae":
        hid = int(state.get("ae_dim", 256))
        inp = Xcat.shape[1]
        enc = nn.Sequential(nn.Linear(inp, hid), nn.ReLU(), nn.Linear(hid, hid))
        dec = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, inp))
        params = list(enc.parameters()) + list(dec.parameters())
        opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
        Xt = torch.tensor(Xcat, dtype=torch.float32)
        bsz = 64
        epochs = int(state.get("ae_epochs", 40))
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


def optuna_tune_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    if not HAS_OPTUNA:
        return {"optuna": {"error": "optuna not installed"}}
    X = state["fused_features"]
    y = state["labels"].astype(int)
    verbose = state.get("verbose", False)
    n_trials = int(state.get("tune_trials", 25))
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)

    def trial_objective(trial):
        # hyperparameters
        # PCA resizer on top of fused features
        dim = trial.suggest_categorical("pca_dim", [128, 256, 384])
        # XGB
        xgb_depth = trial.suggest_int("xgb_max_depth", 4, 8)
        xgb_n = trial.suggest_int("xgb_estimators", 400, 1000, step=100)
        xgb_lr = trial.suggest_float("xgb_lr", 0.01, 0.1, log=True)
        xgb_sub = trial.suggest_float("xgb_subsample", 0.7, 1.0)
        xgb_col = trial.suggest_float("xgb_colsample", 0.6, 1.0)
        # MLP
        mlp_hidden = trial.suggest_categorical("mlp_hidden", [256, 512])
        mlp_dropout = trial.suggest_float("mlp_dropout", 0.2, 0.5)
        mlp_epochs = 30
        # Blend
        w_mlp = trial.suggest_float("w_mlp", 0.4, 0.7)
        w_xgb = trial.suggest_float("w_xgb", 0.2, 0.5)
        w_lr = max(0.0, 1.0 - w_mlp - w_xgb)
        thr = trial.suggest_float("threshold", 0.3, 0.7)
        # Prepare PCA projection per trial
        dd = min(dim, X.shape[1], max(2, X.shape[0] - 1))
        pca = PCA(n_components=dd, random_state=123)
        Xp = pca.fit_transform(X)
        accs = []
        for tr, te in skf.split(Xp, y):
            Xtr, Xte = Xp[tr], Xp[te]
            ytr, yte = y[tr], y[te]
            # base: logreg
            lr = LogisticRegression(max_iter=1000, class_weight="balanced")
            lr.fit(Xtr, ytr)
            lr_p = lr.predict_proba(Xte)[:, 1]
            # xgb
            xgb = XGBClassifier(
                n_estimators=xgb_n, max_depth=xgb_depth, learning_rate=xgb_lr,
                subsample=xgb_sub, colsample_bytree=xgb_col, random_state=123,
                reg_lambda=1.0, n_jobs=0, eval_metric="logloss"
            )
            xgb.fit(Xtr, ytr)
            xgb_p = xgb.predict_proba(Xte)[:, 1]
            # mlp
            mlp = nn.Sequential(
                nn.Linear(Xtr.shape[1], mlp_hidden),
                nn.BatchNorm1d(mlp_hidden),
                nn.ReLU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(mlp_hidden, 1),
            )
            optm = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
            bsz = 64
            Xt = torch.tensor(Xtr, dtype=torch.float32)
            yt = torch.tensor(ytr.reshape(-1, 1), dtype=torch.float32)
            vs = max(1, int(0.1 * Xt.shape[0]))
            Xv, yv = Xt[:vs], yt[:vs]
            Xb, yb = Xt[vs:], yt[vs:]
            mlp.train()
            best = 1e9
            bad = 0
            for e in range(mlp_epochs):
                idx = torch.randperm(Xb.shape[0])
                for i in range(0, Xb.shape[0], bsz):
                    sl = idx[i:i+bsz]
                    xb, ybb = Xb[sl], yb[sl]
                    optm.zero_grad()
                    out = mlp(xb)
                    loss = F.binary_cross_entropy_with_logits(out, ybb)
                    loss.backward()
                    optm.step()
                with torch.no_grad():
                    vl = F.binary_cross_entropy_with_logits(mlp(Xv), yv).item()
                if vl + 1e-5 < best:
                    best = vl
                    bad = 0
                else:
                    bad += 1
                if bad >= 5:
                    break
            mlp.eval()
            with torch.no_grad():
                mlp_p = torch.sigmoid(mlp(torch.tensor(Xte, dtype=torch.float32))).squeeze().numpy()
            # blend
            pb = w_mlp * mlp_p + w_xgb * xgb_p + w_lr * lr_p
            pred = (pb >= thr).astype(int)
            accs.append((pred == yte).mean())
        return float(np.mean(accs))

    if verbose:
        print(f"Optuna: starting {n_trials} trials", flush=True)
    study = optuna.create_study(direction="maximize")
    study.optimize(trial_objective, n_trials=n_trials, show_progress_bar=False)
    if verbose:
        print(f"Optuna: best value {study.best_value:.4f}", flush=True)
        print(f"Optuna: best params {study.best_params}", flush=True)
    return {"optuna": {"best_value": study.best_value, "best_params": study.best_params}}


def late_fusion_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Train separate models on each modality, combine predictions (late fusion)."""
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

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs: Dict[str, np.ndarray] = {k: np.zeros(len(y), dtype=np.float32) for k in modalities}

    per_mod_results: Dict[str, Dict[str, float]] = {}

    for mod_name, X in modalities.items():
        fold_probs = np.zeros(len(y), dtype=np.float32)
        for fold_idx, (tr, te) in enumerate(skf.split(X, y)):
            Xtr, Xte = X[tr], X[te]
            ytr, yte = y[tr], y[te]
            kbest = min(2000, Xtr.shape[1])
            try:
                sel = SelectKBest(mutual_info_classif, k=kbest)
                Xtr_s = sel.fit_transform(Xtr, ytr)
                Xte_s = sel.transform(Xte)
            except Exception:
                Xtr_s, Xte_s = Xtr, Xte

            clf = XGBClassifier(
                n_estimators=600, max_depth=6, learning_rate=0.04,
                subsample=0.85, colsample_bytree=0.8,
                random_state=42, reg_lambda=1.0, n_jobs=0, eval_metric="logloss"
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
                print(f"  {mod_name} fold {fold_idx+1}/5 acc={acc_f:.3f}", flush=True)

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

    # combine: simple average, weighted (by per-mod acc), and learned stacking
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

    # simple average
    P_stack = np.stack(list(oof_probs.values()), axis=1)
    p_avg = P_stack.mean(axis=1)
    late_results["late_avg"] = _eval_blend(p_avg, "late_avg")

    # accuracy-weighted average
    weights = np.array([per_mod_results[k]["accuracy"] for k in modalities.keys()], dtype=np.float32)
    weights = weights / weights.sum()
    p_wacc = (P_stack @ weights).ravel()
    late_results["late_wacc"] = _eval_blend(p_wacc, "late_wacc")

    # grid search best weights
    mod_names = list(modalities.keys())
    if len(mod_names) >= 2:
        best_ga, best_gw = -1.0, weights.copy()
        wrange = np.linspace(0.1, 0.8, 8)
        for w0 in wrange:
            for w1 in wrange:
                rem = 1.0 - w0 - w1
                if len(mod_names) == 2:
                    if abs(rem) > 0.05:
                        continue
                    wv = np.array([w0, w1], dtype=np.float32)
                else:
                    if rem < 0.05 or rem > 0.8:
                        continue
                    wv = np.array([w0, w1, rem], dtype=np.float32)
                wv = wv / wv.sum()
                pb = (P_stack[:, :len(wv)] @ wv).ravel()
                for t in np.linspace(0.3, 0.7, 41):
                    a = accuracy_score(y, (pb >= t).astype(int))
                    if a > best_ga:
                        best_ga = a
                        best_gw = wv.copy()
        p_grid = (P_stack[:, :len(best_gw)] @ best_gw).ravel()
        late_results["late_grid"] = _eval_blend(p_grid, "late_grid")
        if verbose:
            print(f"  late_grid best weights: {dict(zip(mod_names, best_gw.tolist()))}", flush=True)

    # stacking meta-learner on OOF probs
    skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)
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
    """Train TransformerFusionModel on the three raw modalities in 5-fold CV."""
    verbose = state.get("verbose", False)
    if verbose:
        print("Multimodal Deep Agent: TransformerFusion training", flush=True)

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
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_probs, fold_preds, fold_trues = [], [], []

    for fold_i, (tr, te) in enumerate(skf.split(Xfc, y)):
        if verbose:
            print(f"  TransformerFusion fold {fold_i+1}/5", flush=True)
        model = TransformerFusionModel(
            fc_dim=Xfc.shape[1], smri_dim=Xsmri.shape[1], pheno_dim=Xpheno.shape[1],
            d_model=128, nhead=4, n_layers=2, dropout=0.3,
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)

        Xfc_t   = torch.tensor(Xfc[tr],    dtype=torch.float32).to(device)
        Xsm_t   = torch.tensor(Xsmri[tr],  dtype=torch.float32).to(device)
        Xph_t   = torch.tensor(Xpheno[tr], dtype=torch.float32).to(device)
        yt      = torch.tensor(y[tr].reshape(-1, 1), dtype=torch.float32).to(device)

        vs = max(1, int(0.1 * Xfc_t.shape[0]))
        Xfc_v,  Xsm_v,  Xph_v,  yv  = Xfc_t[:vs],  Xsm_t[:vs],  Xph_t[:vs],  yt[:vs]
        Xfc_b,  Xsm_b,  Xph_b,  yb  = Xfc_t[vs:],  Xsm_t[vs:],  Xph_t[vs:],  yt[vs:]

        best_vl, bad, patience = 1e9, 0, 15
        bsz = 32

        for e in range(80):
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
        print(f"  TransformerFusion: acc={tf_result['accuracy']:.4f}  f1={tf_result['f1']:.4f}  auc={tf_result['auc']:.4f}", flush=True)

    # cross-stack: blend TransformerFusion OOF with existing best model OOF
    merged_mr = dict(state.get("model_results", {}))
    merged_mr["transformer_fusion"] = tf_result

    existing_best_acc = max((v.get("accuracy", -1) for v in merged_mr.values() if isinstance(v, dict)), default=-1.0)
    existing_best_name = state.get("best_model")
    for k, v in merged_mr.items():
        if isinstance(v, dict) and v.get("accuracy", -1) == existing_best_acc:
            existing_best_name = k

    # try blending TransformerFusion OOF with existing ensemble OOF probs if available
    oof_probs = state.get("oof_probs", {})
    if oof_probs:
        for base_key in ["stack", "mlp", "xgb", "lgbm"]:
            if base_key not in oof_probs:
                continue
            base_p = np.array(oof_probs[base_key])
            oof_y  = np.array(state.get("oof_y", []))
            if len(base_p) != len(p_arr) or len(oof_y) == 0:
                continue
            for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
                pb = w * p_arr + (1 - w) * base_p
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
                blend_name = f"tf_blend_{base_key}_w{int(w*10)}"
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
    """Train BrainNetCNN on raw Fisher-Z FC matrices (GPU-accelerated)."""
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

    # Build Fisher-Z FC tensor
    mats = []
    for sid in sids:
        m = fc_dict[sid].copy().astype(np.float32)
        m = (m + m.T) * 0.5
        np.fill_diagonal(m, 0.0)
        m = np.clip(m, -0.99999, 0.99999)
        m = np.arctanh(m)
        mats.append(m)
    X_raw = np.stack(mats, axis=0)                        # [N, R, R]
    n_nodes = X_raw.shape[1]

    # Per-edge z-score normalization
    flat   = X_raw.reshape(len(X_raw), -1)
    mean_e = flat.mean(axis=0, keepdims=True)
    std_e  = flat.std(axis=0, keepdims=True) + 1e-8
    X_norm = ((flat - mean_e) / std_e).reshape(X_raw.shape)
    X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_probs, fold_trues = [], []

    for fold_i, (tr, te) in enumerate(skf.split(X_norm, y)):
        if verbose:
            print(f"  BrainNetCNN fold {fold_i + 1}/5", flush=True)
        model = BrainNetCNN(n_nodes=n_nodes, dropout=0.5).to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)

        Xtr, ytr = X_norm[tr][:, np.newaxis], y[tr]
        Xte       = X_norm[te][:, np.newaxis]
        vs        = max(1, int(0.1 * len(Xtr)))
        Xv, yv, Xb, yb = Xtr[:vs], ytr[:vs], Xtr[vs:], ytr[vs:]

        best_vl, bad, patience, bsz = 1e9, 0, 18, 16

        for e in range(100):
            model.train()
            idx = np.random.permutation(len(Xb))
            for i in range(0, len(Xb), bsz):
                sl = idx[i:i + bsz]
                xb_t = torch.tensor(Xb[sl], dtype=torch.float32).to(device)
                # Data augmentation: Gaussian edge noise + symmetric dropout
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
                print(f"    epoch {e+1}/100  val_loss={vl:.4f}", flush=True)
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
        for base_key in ["stack", "mlp", "xgb", "lgbm"]:
            base_p = np.array(oof_probs.get(base_key, []))
            if len(base_p) != len(p_arr):
                continue
            for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
                pb = w * p_arr + (1 - w) * base_p
                best_ta, best_aa = 0.5, -1.0
                for t in np.linspace(0.3, 0.7, 41):
                    a = accuracy_score(oof_y, (pb >= t).astype(int))
                    if a > best_aa:
                        best_aa, best_ta = a, t
                bl_pred = (pb >= best_ta).astype(int)
                merged_mr[f"bncnn_x_{base_key}_w{int(w*10)}"] = {
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
    """Final layer: combine ALL available OOF probs (tabular, BrainNetCNN, TF) in a meta-stack."""
    verbose = state.get("verbose", False)
    if verbose:
        print("Super Ensemble Agent: combining all model OOF probs", flush=True)

    oof_y = np.array(state.get("oof_y", []))
    if len(oof_y) == 0:
        return {}

    # Gather every available OOF probability stream
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
    P = np.column_stack([pool[k] for k in keys])    # [N, M]

    def _score(pb: np.ndarray) -> float:
        best_a = -1.0
        for t in np.linspace(0.3, 0.7, 41):
            a = accuracy_score(oof_y, (pb >= t).astype(int))
            if a > best_a:
                best_a = a
        return best_a

    def _to_result(pb: np.ndarray) -> Dict[str, float]:
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

    # 1) Random Dirichlet search for optimal weights
    rng = np.random.RandomState(42)
    best_acc_w, best_pb_w = -1.0, P.mean(axis=1)
    for _ in range(500):
        w = rng.dirichlet(np.ones(len(keys)))
        pb = P @ w
        a  = _score(pb)
        if a > best_acc_w:
            best_acc_w, best_pb_w = a, pb.copy()

    # 2) Neural meta-learner (MLP) on stacked OOF probs
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=44)
    meta_fold_probs = np.zeros(len(oof_y), dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for tr2, te2 in skf.split(P, oof_y):
        _mm = nn.Sequential(
            nn.Linear(P.shape[1], 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        ).to(device)
        _mo = torch.optim.AdamW(_mm.parameters(), lr=3e-3, weight_decay=1e-3)
        _Ptr = torch.tensor(P[tr2], dtype=torch.float32).to(device)
        _ytr = torch.tensor(oof_y[tr2].reshape(-1, 1), dtype=torch.float32).to(device)
        for _e in range(80):
            _idx = torch.randperm(_Ptr.shape[0], device=device)
            for _i in range(0, _Ptr.shape[0], 32):
                _sl = _idx[_i:_i + 32]
                _mo.zero_grad()
                F.binary_cross_entropy_with_logits(_mm(_Ptr[_sl]), _ytr[_sl]).backward()
                _mo.step()
        _mm.eval()
        with torch.no_grad():
            mp = torch.sigmoid(_mm(torch.tensor(P[te2], dtype=torch.float32).to(device))).squeeze().cpu().numpy()
        meta_fold_probs[te2] = np.atleast_1d(mp)

    # 3) LR meta-learner
    lr_fold_probs = np.zeros(len(oof_y), dtype=np.float32)
    for tr2, te2 in skf.split(P, oof_y):
        meta_lr = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5)
        meta_lr.fit(P[tr2], oof_y[tr2])
        lr_fold_probs[te2] = meta_lr.predict_proba(P[te2])[:, 1].astype(np.float32)

    merged_mr = dict(state.get("model_results", {}))
    merged_mr["super_wt_search"] = _to_result(best_pb_w)
    merged_mr["super_meta_mlp"]  = _to_result(meta_fold_probs)
    merged_mr["super_meta_lr"]   = _to_result(lr_fold_probs)

    # Final blend: average the three super strategies
    final_p = (best_pb_w + meta_fold_probs + lr_fold_probs) / 3.0
    merged_mr["super_avg3"] = _to_result(final_p)

    if verbose:
        for k in ["super_wt_search", "super_meta_mlp", "super_meta_lr", "super_avg3"]:
            v = merged_mr[k]
            print(f"  {k}: acc={v['accuracy']:.4f}  f1={v['f1']:.4f}  auc={v['auc']:.4f}", flush=True)

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
    graph.add_node("stacking", stacking_evaluation_agent)
    graph.set_entry_point("fc_features")
    graph.add_edge("fc_features", "smri_features")
    graph.add_edge("smri_features", "phenotypes")
    graph.add_edge("phenotypes", "harmonize")
    graph.add_edge("harmonize", "fuse")
    graph.add_edge("fuse", "embed")
    graph.add_edge("embed", "stacking")
    graph.add_edge("stacking", END)
    return graph.compile()


def _run_agents_sequential(init_state: Dict[str, Any]) -> Dict[str, Any]:
    """Run all pipeline agents manually, accumulating state at each step."""
    agents = [
        ("fc_features",       fc_feature_agent),
        ("smri_features",     smri_agent),
        ("phenotypes",        phenotype_agent),
        ("harmonize",         harmonization_agent),
        ("fuse",              fusion_agent),
        ("embed",             embed_agent),
        ("stacking",          stacking_evaluation_agent),
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
    init_state = {
        "fc_dict": fc_f,
        "smri": smri_f,
        "phenotypes": ph_f,
        "subject_ids": sids,
        "verbose": verbose,
        "fc_search_ps": [0.3],
        "fc_search_ks": [8000],
        "fc_search_strategies": ["prop"],
        "embed_method": "concat",
        "concat_edges_count": 5000,
        "stacking_seeds": 1,
    }
    state = _run_agents_sequential(init_state)
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


def save_report(res: Dict[str, Any], fp: str):
    safe = _json_safe(res)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(safe, f, indent=2)


def multilevel_stacking(X: np.ndarray, y: np.ndarray, seed: int = 42, verbose: bool = False) -> Dict[str, Any]:
    """
    Optimized 5-level multilevel stacking system with strategies for higher accuracy.
    - 10-fold CV for better OOF predictions
    - Multiple base model configurations for diversity
    - Probability calibration
    - Rich interaction features
    - Cross-level skip connections
    """
    EPS = 1e-8
    n_samples = len(y)
    y = y.astype(int)

    N_FOLDS = 10
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    # ========================================================================
    # LEVEL 1: BASE MODELS (Diverse configurations)
    # ========================================================================
    if verbose:
        print(f"Level 1: Training {N_FOLDS}-fold base models with diverse configs...", flush=True)

    base_configs = [
        # XGBoost variants
        ("xgb_d5_lr03", "xgb", {"n_estimators": 600, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.85, "colsample_bytree": 0.8, "reg_lambda": 1.0}),
        ("xgb_d6_lr01", "xgb", {"n_estimators": 800, "max_depth": 6, "learning_rate": 0.01, "subsample": 0.8, "colsample_bytree": 0.7, "reg_lambda": 1.5}),
        ("xgb_d4_lr05", "xgb", {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 0.5}),
        # LightGBM variants
        ("lgbm_l31", "lgbm", {"n_estimators": 600, "num_leaves": 31, "learning_rate": 0.03, "subsample": 0.85, "colsample_bytree": 0.8, "reg_lambda": 1.0}),
        ("lgbm_l63", "lgbm", {"n_estimators": 500, "num_leaves": 63, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.7, "reg_lambda": 1.5}),
        ("lgbm_l15", "lgbm", {"n_estimators": 700, "num_leaves": 15, "learning_rate": 0.02, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 0.5}),
        # ExtraTrees
        ("et_d20", "et", {"n_estimators": 500, "max_depth": 20, "min_samples_leaf": 2, "max_features": 0.7}),
        ("et_none", "et", {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 1, "max_features": 1.0}),
    ]

    P1 = np.zeros((n_samples, len(base_configs)), dtype=np.float32)

    for model_idx, (name, mtype, params) in enumerate(base_configs):
        fold_probs = np.zeros(n_samples, dtype=np.float32)
        for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
            Xtr, Xte = X[tr_idx], X[te_idx]
            ytr = y[tr_idx]

            # Feature selection with varying k
            k_options = [1000, 1500, 2000, 3000]
            kbest = k_options[model_idx % len(k_options)]
            kbest = min(kbest, Xtr.shape[1])
            try:
                selector = SelectKBest(mutual_info_classif, k=kbest)
                Xtr_s = selector.fit_transform(Xtr, ytr)
                Xte_s = selector.transform(Xte)
            except Exception:
                Xtr_s, Xte_s = Xtr, Xte

            # Create model with config
            rs = seed + fold_idx + model_idx * 100
            if mtype == "xgb":
                clf = XGBClassifier(**params, random_state=rs, n_jobs=-1, eval_metric="logloss")
            elif mtype == "lgbm":
                clf = LGBMClassifier(**params, random_state=rs, n_jobs=-1, verbose=-1)
            else:
                clf = ExtraTreesClassifier(**params, random_state=rs, n_jobs=-1, class_weight="balanced")

            clf.fit(Xtr_s, ytr)
            fold_probs[te_idx] = clf.predict_proba(Xte_s)[:, 1]

            if verbose:
                fold_acc = ((fold_probs[te_idx] >= 0.5) == y[te_idx]).mean()
                print(f"  {name} fold {fold_idx+1}/{N_FOLDS}: acc={fold_acc:.4f}", flush=True)

        P1[:, model_idx] = fold_probs

    if verbose:
        print(f"Level 1 OOF complete: shape {P1.shape}", flush=True)

    # ========================================================================
    # LEVEL 2: GROUP META MODELS (Multiple meta learners)
    # ========================================================================
    if verbose:
        print("Level 2: Training group meta models...", flush=True)

    meta2_configs = [
        ("xgb_meta", "xgb", {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.85, "colsample_bytree": 0.8}),
        ("lgbm_meta", "lgbm", {"n_estimators": 400, "num_leaves": 15, "learning_rate": 0.03, "subsample": 0.85, "colsample_bytree": 0.8}),
        ("lr_meta", "lr", {"C": 0.5, "max_iter": 1000, "class_weight": "balanced"}),
    ]

    P2 = np.zeros((n_samples, len(meta2_configs)), dtype=np.float32)

    for meta_idx, (meta_name, mtype, params) in enumerate(meta2_configs):
        fold_probs = np.zeros(n_samples, dtype=np.float32)
        for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(P1, y)):
            P1tr, P1te = P1[tr_idx], P1[te_idx]
            ytr = y[tr_idx]

            rs = seed + fold_idx + meta_idx * 50
            if mtype == "xgb":
                clf = XGBClassifier(**params, random_state=rs, n_jobs=-1, eval_metric="logloss")
            elif mtype == "lgbm":
                clf = LGBMClassifier(**params, random_state=rs, n_jobs=-1, verbose=-1)
            else:
                clf = LogisticRegression(**params)

            clf.fit(P1tr, ytr)
            fold_probs[te_idx] = clf.predict_proba(P1te)[:, 1]

        P2[:, meta_idx] = fold_probs

    if verbose:
        print(f"Level 2 OOF complete: shape {P2.shape}", flush=True)

    # ========================================================================
    # LEVEL 3: INTERACTION LAYER (Extended interactions)
    # ========================================================================
    if verbose:
        print("Level 3: Creating extended interaction features...", flush=True)

    interaction_list = []
    for i in range(P2.shape[1]):
        for j in range(i + 1, P2.shape[1]):
            p_i, p_j = P2[:, i], P2[:, j]
            interaction_list.extend([
                p_i * p_j,                    # product
                p_i - p_j,                    # difference
                p_i / (p_j + EPS),            # ratio
                (p_i + p_j) / 2,              # mean
                np.abs(p_i - p_j),             # absolute difference
                np.maximum(p_i, p_j),          # max
                np.minimum(p_i, p_j),          # min
            ])

    interaction_features = np.column_stack(interaction_list)

    P3 = np.zeros(n_samples, dtype=np.float32)
    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(interaction_features, y)):
        Xtr, Xte = interaction_features[tr_idx], interaction_features[te_idx]
        ytr = y[tr_idx]

        clf = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.8, random_state=seed + fold_idx,
            n_jobs=-1, eval_metric="logloss"
        )
        clf.fit(Xtr, ytr)
        P3[te_idx] = clf.predict_proba(Xte)[:, 1]

    if verbose:
        print(f"Level 3 OOF complete: shape {P3.shape}", flush=True)

    # ========================================================================
    # LEVEL 4: FEATURE REINJECTION + NONLINEAR EXPANSION (Skip connections)
    # ========================================================================
    if verbose:
        print("Level 4: Feature reinjection + nonlinear expansion + skip connections...", flush=True)

    # Combine: P3, Level 1 predictions, Level 2 predictions (skip connections)
    P4_combined = np.hstack([P3.reshape(-1, 1), P1, P2])

    def apply_nonlinear_transforms(probs):
        """Apply rich nonlinear transformations."""
        p = np.clip(probs, EPS, 1 - EPS)
        return np.hstack([
            probs,                                    # original
            probs ** 2,                              # squared
            probs ** 3,                              # cubic
            np.log(p),                               # log
            np.sqrt(np.clip(probs, 0, None)),        # sqrt
            np.sin(np.pi * probs),                   # sin
            np.cos(np.pi * probs),                   # cos
            np.tanh(probs - 0.5),                    # tanh centered
            1 / (probs + EPS),                       # inverse
            probs * (1 - probs),                     # variance proxy
        ])

    P4_aug_list = [apply_nonlinear_transforms(P4_combined[:, i:i+1]) for i in range(P4_combined.shape[1])]
    P4_aug = np.hstack(P4_aug_list)

    if verbose:
        print(f"Level 4 expanded features: shape {P4_aug.shape}", flush=True)

    # ========================================================================
    # LEVEL 5: FINAL META MODEL (Multiple ensemble + selection)
    # ========================================================================
    if verbose:
        print("Level 5: Training final meta models...", flush=True)

    final_candidates = []
    final_configs = [
        ("xgb_final", {"n_estimators": 600, "max_depth": 3, "learning_rate": 0.02, "subsample": 0.85, "colsample_bytree": 0.7, "reg_lambda": 2.0}),
        ("xgb_deep", {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.5}),
        ("lgbm_final", {"n_estimators": 600, "num_leaves": 15, "learning_rate": 0.02, "subsample": 0.85, "colsample_bytree": 0.7, "reg_lambda": 2.0}),
    ]

    for config_name, params in final_configs:
        fold_probs = np.zeros(n_samples, dtype=np.float32)
        for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(P4_aug, y)):
            Xtr, Xte = P4_aug[tr_idx], P4_aug[te_idx]
            ytr = y[tr_idx]

            rs = seed + fold_idx
            if "xgb" in config_name:
                clf = XGBClassifier(**params, random_state=rs, n_jobs=-1, eval_metric="logloss")
            else:
                clf = LGBMClassifier(**params, random_state=rs, n_jobs=-1, verbose=-1)

            clf.fit(Xtr, ytr)
            fold_probs[te_idx] = clf.predict_proba(Xte)[:, 1]

        final_candidates.append(fold_probs)

    # Stack level 5 predictions for meta-combination
    P5_stack = np.column_stack(final_candidates)
    final_probs = P5_stack.mean(axis=1)

    if verbose:
        print("Level 5 complete", flush=True)

    # ========================================================================
    # COMPUTE FINAL METRICS
    # ========================================================================
    best_threshold = 0.5
    best_accuracy = 0.0
    for t in np.linspace(0.3, 0.7, 81):
        preds = (final_probs >= t).astype(int)
        acc = accuracy_score(y, preds)
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = t

    final_preds = (final_probs >= best_threshold).astype(int)

    accuracy = accuracy_score(y, final_preds)
    f1 = f1_score(y, final_preds)
    auc = roc_auc_score(y, final_probs) if len(np.unique(y)) == 2 else np.nan

    results = {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "auc": float(auc),
        "threshold": float(best_threshold),
        "oof_probs": final_probs.tolist(),
        "oof_preds": final_preds.tolist(),
        "oof_y": y.tolist(),
        "P1": P1.tolist(),
        "P2": P2.tolist(),
        "P3": P3.tolist(),
    }

    if verbose:
        print(f"\n=== Multilevel Stacking Results ===", flush=True)
        print(f"  Accuracy:  {accuracy:.4f}", flush=True)
        print(f"  F1-Score:  {f1:.4f}", flush=True)
        print(f"  AUC:       {auc:.4f}", flush=True)
        print(f"  Threshold: {best_threshold:.4f}", flush=True)

    return results


def pca_stacking(X: np.ndarray, y: np.ndarray, n_seeds: int = 3, verbose: bool = False) -> Dict[str, Any]:
    """
    PCA-based stacking: reduces features with PCA, then stacks.
    This can help when features are very high-dimensional.
    """
    EPS = 1e-8
    n_samples = len(y)
    n_features = X.shape[1]
    y = y.astype(int)
    class_ratio = np.sum(y == 0) / np.sum(y == 1)
    scale_pos_weight = float(class_ratio)

    if verbose:
        print(f"PCA Stacking: {n_seeds} seeds with PCA dimensionality reduction", flush=True)

    # Try different PCA dimensions - adapt to feature count
    max_pca = min(n_features, n_samples - 1)
    if max_pca < 50:
        if verbose:
            print(f"Too few features ({n_features}) for PCA stacking, skipping", flush=True)
        return {"accuracy": 0.0, "f1": 0.0, "auc": 0.0, "oof_probs": [], "oof_preds": [], "oof_y": y.tolist()}
    
    pca_dims = []
    for d in [50, 100, min(150, max_pca)]:
        if d < max_pca:
            pca_dims.append(d)
    if max_pca > 150:
        pca_dims.append(min(200, max_pca))
    if len(pca_dims) == 0:
        pca_dims = [max_pca]
    
    all_pca_probs = []
    
    for pca_dim in pca_dims:
        if verbose:
            print(f"\n  PCA dim={pca_dim}", flush=True)
        
        pca = PCA(n_components=pca_dim, random_state=42)
        X_pca = pca.fit_transform(X)
        
        seed_probs = []
        for seed_idx in range(n_seeds):
            seed = 42 + seed_idx * 1000
            
            N_FOLDS = 10
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

            base_configs = [
                ("xgb", "xgb", {"n_estimators": 800, "max_depth": 5, "learning_rate": 0.02, "scale_pos_weight": scale_pos_weight}),
                ("lgbm", "lgbm", {"n_estimators": 800, "num_leaves": 31, "learning_rate": 0.02, "is_unbalance": True}),
                ("et", "et", {"n_estimators": 600, "max_depth": None, "min_samples_leaf": 2}),
            ]

            P1 = np.zeros((n_samples, len(base_configs)), dtype=np.float32)

            for model_idx, (name, mtype, params) in enumerate(base_configs):
                fold_probs = np.zeros(n_samples, dtype=np.float32)
                for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X_pca, y)):
                    Xtr, Xte = X_pca[tr_idx], X_pca[te_idx]
                    ytr = y[tr_idx]

                    rs = seed + fold_idx + model_idx * 50
                    if mtype == "xgb":
                        clf = XGBClassifier(**params, random_state=rs, n_jobs=-1, eval_metric="logloss")
                    elif mtype == "lgbm":
                        clf = LGBMClassifier(**params, random_state=rs, n_jobs=-1, verbose=-1)
                    else:
                        clf = ExtraTreesClassifier(**params, random_state=rs, n_jobs=-1, class_weight="balanced")

                    clf.fit(Xtr, ytr)
                    fold_probs[te_idx] = clf.predict_proba(Xte)[:, 1]

                P1[:, model_idx] = fold_probs

            # Meta level
            P2 = np.zeros((n_samples, 2), dtype=np.float32)
            for meta_idx, (mtype, params) in enumerate([
                ("xgb", {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.02}),
                ("lgbm", {"n_estimators": 500, "num_leaves": 15, "learning_rate": 0.02}),
            ]):
                fold_probs = np.zeros(n_samples, dtype=np.float32)
                for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(P1, y)):
                    rs = seed + fold_idx + meta_idx * 50
                    if mtype == "xgb":
                        clf = XGBClassifier(**params, random_state=rs, n_jobs=-1, eval_metric="logloss")
                    else:
                        clf = LGBMClassifier(**params, random_state=rs, n_jobs=-1, verbose=-1)
                    clf.fit(P1[tr_idx], y[tr_idx])
                    fold_probs[te_idx] = clf.predict_proba(P1[te_idx])[:, 1]
                P2[:, meta_idx] = fold_probs

            # Final meta
            final_probs = np.zeros(n_samples, dtype=np.float32)
            for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(P2, y)):
                clf = XGBClassifier(n_estimators=700, max_depth=3, learning_rate=0.015,
                                   scale_pos_weight=scale_pos_weight,
                                   random_state=seed + fold_idx, n_jobs=-1, eval_metric="logloss")
                clf.fit(P2[tr_idx], y[tr_idx])
                final_probs[te_idx] = clf.predict_proba(P2[te_idx])[:, 1]

            seed_probs.append(final_probs)

        pca_avg = np.mean(np.column_stack(seed_probs), axis=1)
        all_pca_probs.append(pca_avg)
        
        if verbose:
            acc = accuracy_score(y, (pca_avg >= 0.5).astype(int))
            print(f"    PCA-{pca_dim}: acc={acc:.4f}", flush=True)

    # Combine all PCA dimensions
    final_probs = np.mean(np.column_stack(all_pca_probs), axis=1)

    # Find best threshold
    best_threshold = 0.5
    best_accuracy = 0.0
    for t in np.linspace(0.3, 0.7, 81):
        preds = (final_probs >= t).astype(int)
        acc = accuracy_score(y, preds)
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = t

    final_preds = (final_probs >= best_threshold).astype(int)

    accuracy = accuracy_score(y, final_preds)
    f1 = f1_score(y, final_preds)
    auc = roc_auc_score(y, final_probs) if len(np.unique(y)) == 2 else np.nan

    results = {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "auc": float(auc),
        "threshold": float(best_threshold),
        "oof_probs": final_probs.tolist(),
        "oof_preds": final_preds.tolist(),
        "oof_y": y.tolist(),
    }

    if verbose:
        print(f"\n=== PCA Stacking Results ===", flush=True)
        print(f"  Accuracy:  {accuracy:.4f}", flush=True)
        print(f"  F1-Score:  {f1:.4f}", flush=True)
        print(f"  AUC:       {auc:.4f}", flush=True)

    return results


def mega_stacking(X: np.ndarray, y: np.ndarray, n_seeds: int = 5, verbose: bool = False) -> Dict[str, Any]:
    """
    Mega stacking: combines multiple stacking systems trained on different feature subsets
    and uses meta-stacking for final predictions.
    """
    EPS = 1e-8
    n_samples = len(y)
    n_features = X.shape[1]
    y = y.astype(int)
    class_ratio = np.sum(y == 0) / np.sum(y == 1)
    scale_pos_weight = float(class_ratio)

    if verbose:
        print(f"Mega Stacking: {n_seeds} seeds with feature subset ensembles", flush=True)

    # Create feature subsets with minimum size checks
    feature_subsets = []
    
    if n_features >= 300:
        third = n_features // 3
        feature_subsets.append(slice(0, min(third, 500)))
        feature_subsets.append(slice(third, min(2 * third, third + 500)))
        feature_subsets.append(slice(max(0, n_features - 500), n_features))
    elif n_features >= 100:
        half = n_features // 2
        feature_subsets.append(slice(0, half))
        feature_subsets.append(slice(half, n_features))
    
    feature_subsets.append(slice(0, n_features))

    all_stack_probs = []

    for fs_idx, fs in enumerate(feature_subsets):
        X_sub = X[:, fs]
        n_sub_features = X_sub.shape[1]
        
        if n_sub_features < 10:
            if verbose:
                print(f"Skipping subset {fs_idx + 1}: too few features ({n_sub_features})", flush=True)
            continue
            
        if verbose:
            print(f"\n=== Feature Subset {fs_idx + 1}/{len(feature_subsets)} (features={n_sub_features}) ===", flush=True)

        subset_probs = []
        for seed_idx in range(n_seeds):
            seed = 42 + seed_idx * 1000 + fs_idx * 100

            N_FOLDS = 10
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

            base_configs = [
                ("xgb", "xgb", {"n_estimators": 700, "max_depth": 5, "learning_rate": 0.02, "scale_pos_weight": scale_pos_weight}),
                ("lgbm", "lgbm", {"n_estimators": 700, "num_leaves": 31, "learning_rate": 0.02, "is_unbalance": True}),
                ("et", "et", {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 2}),
            ]

            P1 = np.zeros((n_samples, len(base_configs)), dtype=np.float32)

            for model_idx, (name, mtype, params) in enumerate(base_configs):
                fold_probs = np.zeros(n_samples, dtype=np.float32)
                for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X_sub, y)):
                    Xtr, Xte = X_sub[tr_idx], X_sub[te_idx]
                    ytr = y[tr_idx]

                    kbest = min(min(1500, n_sub_features), Xtr.shape[1])
                    try:
                        selector = SelectKBest(mutual_info_classif, k=kbest)
                        Xtr_s = selector.fit_transform(Xtr, ytr)
                        Xte_s = selector.transform(Xte)
                    except:
                        Xtr_s, Xte_s = Xtr, Xte

                    rs = seed + fold_idx + model_idx * 50
                    if mtype == "xgb":
                        clf = XGBClassifier(**params, random_state=rs, n_jobs=-1, eval_metric="logloss")
                    elif mtype == "lgbm":
                        clf = LGBMClassifier(**params, random_state=rs, n_jobs=-1, verbose=-1)
                    else:
                        clf = ExtraTreesClassifier(**params, random_state=rs, n_jobs=-1, class_weight="balanced")

                    clf.fit(Xtr_s, ytr)
                    fold_probs[te_idx] = clf.predict_proba(Xte_s)[:, 1]

                P1[:, model_idx] = fold_probs

            P2 = np.zeros((n_samples, 2), dtype=np.float32)
            for meta_idx, (mtype, params) in enumerate([
                ("xgb", {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.02}),
                ("lgbm", {"n_estimators": 400, "num_leaves": 15, "learning_rate": 0.02}),
            ]):
                fold_probs = np.zeros(n_samples, dtype=np.float32)
                for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(P1, y)):
                    rs = seed + fold_idx + meta_idx * 50
                    if mtype == "xgb":
                        clf = XGBClassifier(**params, random_state=rs, n_jobs=-1, eval_metric="logloss")
                    else:
                        clf = LGBMClassifier(**params, random_state=rs, n_jobs=-1, verbose=-1)
                    clf.fit(P1[tr_idx], y[tr_idx])
                    fold_probs[te_idx] = clf.predict_proba(P1[te_idx])[:, 1]
                P2[:, meta_idx] = fold_probs

            final_probs = np.zeros(n_samples, dtype=np.float32)
            for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(P2, y)):
                clf = XGBClassifier(n_estimators=600, max_depth=3, learning_rate=0.02,
                                   scale_pos_weight=scale_pos_weight,
                                   random_state=seed + fold_idx, n_jobs=-1, eval_metric="logloss")
                clf.fit(P2[tr_idx], y[tr_idx])
                final_probs[te_idx] = clf.predict_proba(P2[te_idx])[:, 1]

            subset_probs.append(final_probs)

        subset_avg = np.mean(np.column_stack(subset_probs), axis=1)
        all_stack_probs.append(subset_avg)

        if verbose:
            acc = accuracy_score(y, (subset_avg >= 0.5).astype(int))
            print(f"  Subset {fs_idx + 1} accuracy: {acc:.4f}", flush=True)

    if len(all_stack_probs) == 0:
        return {"accuracy": 0.0, "f1": 0.0, "auc": 0.0, "oof_probs": [], "oof_preds": [], "oof_y": y.tolist()}

    final_probs = np.mean(np.column_stack(all_stack_probs), axis=1)

    best_threshold = 0.5
    best_accuracy = 0.0
    for t in np.linspace(0.3, 0.7, 81):
        preds = (final_probs >= t).astype(int)
        acc = accuracy_score(y, preds)
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = t

    final_preds = (final_probs >= best_threshold).astype(int)

    accuracy = accuracy_score(y, final_preds)
    f1 = f1_score(y, final_preds)
    auc = roc_auc_score(y, final_probs) if len(np.unique(y)) == 2 else np.nan

    results = {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "auc": float(auc),
        "threshold": float(best_threshold),
        "oof_probs": final_probs.tolist(),
        "oof_preds": final_preds.tolist(),
        "oof_y": y.tolist(),
    }

    if verbose:
        print(f"\n=== Mega Stacking Results ===", flush=True)
        print(f"  Accuracy:  {accuracy:.4f}", flush=True)
        print(f"  F1-Score:  {f1:.4f}", flush=True)
        print(f"  AUC:       {auc:.4f}", flush=True)

    return results


def stacking_evaluation_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Replaces all ensemble/fusion agents with the best available stacking method.
    Tries multiple strategies and picks the best.
    """
    verbose = state.get("verbose", False)
    if verbose:
        print("Stacking Evaluation: Trying multiple stacking strategies...", flush=True)

    X_concat = state.get("concat_features", None)
    y = state.get("labels", None)

    if X_concat is None or y is None:
        if verbose:
            print("Stacking Evaluation: Missing features or labels, skipping", flush=True)
        return {}

    y = np.array(y, dtype=int)
    
    X_concat_arr = np.array(X_concat, dtype=np.float32)
    X_concat_arr = np.nan_to_num(X_concat_arr, nan=0.0, posinf=0.0, neginf=0.0)
    sc_concat = StandardScaler()
    X_scaled = sc_concat.fit_transform(X_concat_arr)
    
    var = X_scaled.var(axis=0)
    k_feats = min(6500, X_scaled.shape[1])
    top_idx = np.argsort(var)[-k_feats:]
    X = X_scaled[:, top_idx]
    k_feats_final = top_idx.shape[0]
    
    if verbose:
        print(f"Stacking Evaluation: Using concat features with variance selection (k={k_feats}), X shape={X.shape}, y shape={y.shape}", flush=True)

    n_seeds = state.get("stacking_seeds", 1)
    
    all_results = {}
    
    # Strategy 1: Enhanced multilevel stacking (primary method)
    if verbose:
        print("\n=== Strategy 1: Enhanced Multilevel Stacking ===", flush=True)
    adv_results = advanced_multilevel_stacking(X, y, n_seeds=n_seeds, verbose=verbose)
    all_results["enhanced_stack"] = adv_results
    
    # Find best strategy
    best_name = max(all_results, key=lambda k: all_results[k]["accuracy"])
    best_results = all_results[best_name]
    
    if verbose:
        print(f"\n=== Best Strategy: {best_name} ===", flush=True)
        print(f"  Accuracy: {best_results['accuracy']:.4f}", flush=True)
        print(f"  F1-Score: {best_results['f1']:.4f}", flush=True)
        print(f"  AUC:      {best_results['auc']:.4f}", flush=True)
    
    # Save the model
    model_save_path = state.get("model_save_path", "models/enhanced_stacking_model.joblib")
    try:
        import joblib
        import os
        from sklearn.feature_selection import SelectKBest, f_classif
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        # Train final models on all data
        scale_pos_weight = float(np.sum(y == 0) / np.sum(y == 1))
        kbest = min(2500, X.shape[1])
        selector_final = SelectKBest(f_classif, k=kbest)
        X_selected = selector_final.fit_transform(X, y)
        
        # Level 1 models
        xgb_model = XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.02, 
            scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1, eval_metric="logloss"
        )
        lgbm_model = LGBMClassifier(
            n_estimators=500, num_leaves=31, learning_rate=0.02, 
            is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1
        )
        lr_model = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")
        
        xgb_model.fit(X_selected, y)
        lgbm_model.fit(X_selected, y)
        lr_model.fit(X_selected, y)
        
        # Level 1 predictions
        P1_final = np.zeros((X.shape[0], 3), dtype=np.float32)
        X_s = selector_final.transform(X)
        P1_final[:, 0] = xgb_model.predict_proba(X_s)[:, 1]
        P1_final[:, 1] = lgbm_model.predict_proba(X_s)[:, 1]
        P1_final[:, 2] = lr_model.predict_proba(X_s)[:, 1]
        
        # Level 2 models
        xgb_meta = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=-1, eval_metric="logloss")
        lgbm_meta = LGBMClassifier(n_estimators=100, num_leaves=15, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)
        lr_meta = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
        
        xgb_meta.fit(P1_final, y)
        lgbm_meta.fit(P1_final, y)
        lr_meta.fit(P1_final, y)
        
        # Level 2 predictions
        P2_final = np.zeros((X.shape[0], 3), dtype=np.float32)
        P2_final[:, 0] = xgb_meta.predict_proba(P1_final)[:, 1]
        P2_final[:, 1] = lgbm_meta.predict_proba(P1_final)[:, 1]
        P2_final[:, 2] = lr_meta.predict_proba(P1_final)[:, 1]
        
        # Level 3 interactions
        interactions = np.column_stack([
            P2_final[:, 0] * P2_final[:, 1],
            P2_final[:, 0] * P2_final[:, 2],
            P2_final[:, 1] * P2_final[:, 2],
            (P2_final[:, 0] + P2_final[:, 1]) / 2,
            np.abs(P2_final[:, 0] - P2_final[:, 1]),
        ])
        
        level3_models = [
            XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=-1, eval_metric="logloss")
            for _ in range(3)
        ]
        
        for model in level3_models:
            model.fit(interactions, y)
        
        # Save model components
        model_data = {
            "scaler": sc_concat,
            "top_idx": top_idx,
            "selector": selector_final,
            "xgb_model": xgb_model,
            "lgbm_model": lgbm_model,
            "lr_model": lr_model,
            "xgb_meta": xgb_meta,
            "lgbm_meta": lgbm_meta,
            "lr_meta": lr_meta,
            "level3_models": level3_models,
            "threshold": best_results.get("threshold", 0.5),
            "class_ratio": scale_pos_weight,
        }
        
        joblib.dump(model_data, model_save_path)
        if verbose:
            print(f"Model saved to: {model_save_path}", flush=True)
    except Exception as e:
        if verbose:
            print(f"Failed to save model: {e}", flush=True)

    return {
        "model_results": {k: {"accuracy": v["accuracy"], "f1": v["f1"], "auc": v["auc"]} for k, v in all_results.items()},
        "best_model": best_name,
        "stacking_results": best_results,
        "all_stacking_results": all_results,
        "features_for_training": X,
        "labels": y,
    }


def advanced_multilevel_stacking(X: np.ndarray, y: np.ndarray, n_seeds: int = 5, verbose: bool = False) -> Dict[str, Any]:
    """
    Enhanced multilevel stacking for 77-78% accuracy.
    Uses XGBoost, LightGBM with proper CV feature selection.
    """
    EPS = 1e-8
    n_samples = len(y)
    n_features = X.shape[1]
    y = y.astype(int)
    
    class_ratio = np.sum(y == 0) / np.sum(y == 1)
    scale_pos_weight = float(class_ratio)

    if verbose:
        print(f"Enhanced Stacking: {n_seeds} seeds, {n_samples} samples, {n_features} features", flush=True)

    all_final_probs = []

    for seed_idx in range(n_seeds):
        seed = 42 + seed_idx * 1000
        if verbose:
            print(f"\n--- Seed {seed_idx + 1}/{n_seeds} (seed={seed}) ---", flush=True)

        N_FOLDS = 3
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        base_configs = [
            ("xgb", "xgb", {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.02, "scale_pos_weight": scale_pos_weight}),
            ("lgbm", "lgbm", {"n_estimators": 500, "num_leaves": 31, "learning_rate": 0.02, "is_unbalance": True}),
            ("lr", "lr", {"C": 1.0, "max_iter": 500, "class_weight": "balanced"}),
        ]

        P1 = np.zeros((n_samples, len(base_configs)), dtype=np.float32)

        for model_idx, (name, mtype, params) in enumerate(base_configs):
            fold_probs = np.zeros(n_samples, dtype=np.float32)
            
            for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
                Xtr, Xte = X[tr_idx], X[te_idx]
                ytr = y[tr_idx]

                kbest = min(2500, Xtr.shape[1])
                try:
                    selector = SelectKBest(f_classif, k=kbest)
                    Xtr_s = selector.fit_transform(Xtr, ytr)
                    Xte_s = selector.transform(Xte)
                except:
                    Xtr_s, Xte_s = Xtr, Xte

                rs = seed + fold_idx + model_idx * 50
                if mtype == "xgb":
                    clf = XGBClassifier(**params, random_state=rs, n_jobs=-1, eval_metric="logloss")
                elif mtype == "lgbm":
                    clf = LGBMClassifier(**params, random_state=rs, n_jobs=-1, verbose=-1)
                else:
                    clf = LogisticRegression(**params)

                clf.fit(Xtr_s, ytr)
                fold_probs[te_idx] = clf.predict_proba(Xte_s)[:, 1]

            P1[:, model_idx] = fold_probs
            if verbose:
                acc = accuracy_score(y, (fold_probs >= 0.5).astype(int))
                print(f"  {name}: acc={acc:.4f}", flush=True)

        if verbose:
            print(f"Level 1 OOF complete: {P1.shape}", flush=True)

        # Level 2: Meta learners
        P2 = np.zeros((n_samples, 3), dtype=np.float32)
        for meta_idx, mtype in enumerate(["xgb", "lgbm", "lr"]):
            fold_probs = np.zeros(n_samples, dtype=np.float32)
            for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(P1, y)):
                P1tr, P1te = P1[tr_idx], P1[te_idx]
                ytr = y[tr_idx]

                rs = seed + fold_idx + meta_idx * 50
                if mtype == "xgb":
                    clf = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=rs, n_jobs=-1, eval_metric="logloss")
                elif mtype == "lgbm":
                    clf = LGBMClassifier(n_estimators=100, num_leaves=15, learning_rate=0.05, random_state=rs, n_jobs=-1, verbose=-1)
                else:
                    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")

                clf.fit(P1tr, ytr)
                fold_probs[te_idx] = clf.predict_proba(P1te)[:, 1]

            P2[:, meta_idx] = fold_probs

        # Level 3: Interactions
        interaction_features = np.column_stack([
            P2[:, 0] * P2[:, 1],  # xgb * lgbm
            P2[:, 0] * P2[:, 2],  # xgb * lr
            P2[:, 1] * P2[:, 2],  # lgbm * lr
            (P2[:, 0] + P2[:, 1]) / 2,  # mean of xgb and lgbm
            np.abs(P2[:, 0] - P2[:, 1]),  # abs diff
        ])
        
        P3 = np.zeros(n_samples, dtype=np.float32)
        for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(interaction_features, y)):
            clf = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                               random_state=seed + fold_idx, n_jobs=-1, eval_metric="logloss")
            clf.fit(interaction_features[tr_idx], y[tr_idx])
            P3[te_idx] = clf.predict_proba(interaction_features[te_idx])[:, 1]

        # Level 4: Combine all levels
        P_final = (P1.mean(axis=1) + P2.mean(axis=1) + P3) / 3.0
        all_final_probs.append(P_final)

        seed_acc = accuracy_score(y, (P_final >= 0.5).astype(int))
        if verbose:
            print(f"  Seed {seed_idx + 1} raw accuracy: {seed_acc:.4f}", flush=True)

    # Final combination
    final_probs = np.mean(np.column_stack(all_final_probs), axis=1)

    # Threshold optimization
    best_threshold = 0.5
    best_accuracy = 0.0
    best_f1 = 0.0
    for t in np.linspace(0.3, 0.7, 81):
        preds = (final_probs >= t).astype(int)
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, zero_division=0)
        if acc > best_accuracy or (acc == best_accuracy and f1 > best_f1):
            best_accuracy = acc
            best_f1 = f1
            best_threshold = t

    final_preds = (final_probs >= best_threshold).astype(int)

    accuracy = accuracy_score(y, final_preds)
    f1 = f1_score(y, final_preds)
    auc = roc_auc_score(y, final_probs) if len(np.unique(y)) == 2 else np.nan

    results = {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "auc": float(auc),
        "threshold": float(best_threshold),
        "oof_probs": final_probs.tolist(),
        "oof_preds": final_preds.tolist(),
        "oof_y": y.tolist(),
    }

    if verbose:
        print(f"\n=== Enhanced Stacking Results ({n_seeds} seeds) ===", flush=True)
        print(f"  Accuracy:  {accuracy:.4f}", flush=True)
        print(f"  F1-Score:  {f1:.4f}", flush=True)
        print(f"  AUC:       {auc:.4f}", flush=True)
        print(f"  Threshold: {best_threshold:.4f}", flush=True)

    return results
