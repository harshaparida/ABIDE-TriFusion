# ABIDE‑TriFusion  
**Multimodal Transformer‑Fusion for Autism Classification on the ABIDE dataset**

---  

## 📖 Overview  

ABIDE‑TriFusion is an **end‑to‑end research pipeline** that combines three complementary data modalities from the **ABIDE (Autism Brain Imaging Data‑Exchange)** cohort:

| Modality | Raw source | What the pipeline extracts |
|----------|-----------|----------------------------|
| Functional Connectivity (FC) | `*.csv` correlation matrices (one per subject) | Fisher‑Z transformed, sparsified edge set, graph‑theoretic node statistics, adjacency matrices for a GCN |
| Structural MRI (sMRI) | CSV table of regional volumes / thicknesses | Standardised numeric vector (optional PCA) |
| Phenotypic / Clinical data | CSV table of demographics, IQ, site, diagnosis, … | Normalised numeric covariates + one‑hot encoded categorical fields (e.g., SEX) |

The three streams are **harmonised across scanning sites** (ComBat via `neuroHarmonize`), **fused** (concatenation + optional PCA or a learned linear projection), and fed to a **cross‑modal Transformer** that treats each modality as a token together with a CLS token for classification.

In addition to the transformer, the pipeline trains a large suite of classical tabular models, a Graph‑Neural Network (GCN), a BrainNetCNN for raw FC matrices, and several ensemble/meta‑learning strategies, providing a **comprehensive benchmark** of what works best for ASD classification on ABIDE.

---  

## 🚀 Quick Start  

```bash
# 1️⃣ Clone / copy the repository
git clone https://github.com/yourusername/ABIDE-TriFusion.git   # or copy the folder
cd ABIDE-TriFusion/autismmat

# 2️⃣ Create a clean conda environment (Python 3.9‑3.11)
conda create -n abidetri python=3.10 -y
conda activate abidetri

# 3️⃣ Install the required packages
pip install -r requirements.txt

# 4️⃣ Prepare your data (see the section “Data preparation” below)

# 5️⃣ Run the full pipeline (uses the default config.yaml)
python main.py --config config.yaml
```

The command will:

1. Load the FC matrices, sMRI table and phenotypic CSV.  
2. Align subjects that are present in *all* modalities.  
3. Run every processing **agent** (FC feature extraction, sMRI processing, phenotype handling, site harmonisation, fusion, embedding, model selection, ensembling, late‑fusion, transformer, BrainNetCNN, super‑ensemble).  
4. Save a trimmed JSON report (`results.json`) and a plain‑text summary (`summary.txt`) under the folder defined in the config (`output_dir`).  

---  

## 📂 Folder Structure  

```
autismmat/
├─ analysis.ipynb                # Exploratory notebook (quick sanity checks)
├─ main.py                       # CLI entry‑point (parses config & runs the pipeline)
├─ config.yaml                   # Default configuration file (editable)
├─ requirements.txt               # Exact version list of mandatory packages
├─ README.md                     # ← you are here
└─ src/
   ├─ __init__.py
   ├─ utils.py                   # Helper functions (logging, IO, etc.)
   └─ agentic/
       ├─ __init__.py
       └─ pipeline.py            # All agents (FC, sMRI, phenotype, fusion, models, …)
```

All core processing lives in `src/agentic/pipeline.py`.  
If you add new agents, import them in `src/agentic/__init__.py` and update the agent graph inside `pipeline.py`.

---  

## 🛠️ Installation  

The **mandatory** dependencies are listed in `requirements.txt`.  
Optional libraries (CatBoost, LightGBM, Optuna) are only needed if you want to use the corresponding agents.

```text
numpy>=1.23
pandas>=1.5
scikit-learn>=1.2
torch>=2.0
torchvision>=0.15
networkx>=3.0
xgboost>=2.0
neuroHarmonize>=0.5
langgraph>=0.0.14
pyyaml
tqdm
```

If you need the optional packages later:

```bash
pip install catboost lightgbm optuna
```

---  

## 📁 Data preparation  

The pipeline expects a **single folder** that contains the three raw data sources.  
Example layout (`my_data/`):

```
my_data/
├─ fc_matrices/                 # one CSV per subject:  <SUBJECT_ID>_fc_matrix.csv
│   ├─ 00123_fc_matrix.csv
│   └─ …
├─ structural_data_cleaned.csv   # sMRI table (must contain a column "SUB_ID")
└─ phenotypic_data.csv          # phenotype / clinical table (must contain "SUB_ID" and "DX_GROUP")
```

### Required columns  

| File | Required columns | Notes |
|------|------------------|-------|
| `*_fc_matrix.csv` | none (subject ID is taken from the filename) | Must be a square correlation matrix **without** a header. |
| `structural_data_cleaned.csv` | `SUB_ID` + any numeric ROI features | Missing values will be median‑imputed. |
| `phenotypic_data.csv` | `SUB_ID`, `DX_GROUP`, optional `AGE_AT_SCAN`, `FIQ`, `SEX`, `SITE` | `DX_GROUP` is binary: `1 = ASD`, `0 = TD`. Non‑binary values are cast to `int`. |

> **Tip:** If you already have the original ABIDE raw fMRI files, you need to compute the correlation matrices **first** and store each as a CSV as shown above.

Update the path to this folder in `config.yaml` (see the **Configuration** section).

---  

## ⚙️ Configuration  

All tunable parameters live in a **YAML** file.  
The repository ships a minimal `config.yaml`; feel free to copy it and modify any field.

```yaml
# ------------------------------------------------------------------
# Basic I/O
data_dir: "D:/Thesis/brain_analysis/autismmat/data"   # <-- point to your data folder
output_dir: "D:/Thesis/brain_analysis/autismmat/results"
verbose: true
random_state: 42

# ------------------------------------------------------------------
# FC sparsification (search space used by the FC agent)
fc_search_ps: [0.05, 0.1, 0.2, 0.3, 0.4]    # proportion of strongest edges to keep
fc_search_ks: [1500, 3000, 5000, 8000]    # number of edges to select after variance ranking
fc_search_strategies: ["prop", "mst_prop"] # "prop" = simple threshold, "mst_prop" = MST + top‑p

# ------------------------------------------------------------------
# Fusion & dimensionality reduction
fusion_method: "concat"      # "concat" (default) or "attention"
pca_dim: 256                # set 0 to skip PCA after concatenation

# ------------------------------------------------------------------
# Optional auto‑encoder embedding
embed_method: "none"        # "ae" for auto‑encoder, "none" otherwise
ae_dim: 384
ae_epochs: 40

# ------------------------------------------------------------------
# Hyper‑parameter tuning (Optuna)
tune_trials: 0              # set >0 to run Optuna search, 0 = skip

# ------------------------------------------------------------------
# Model‑specific tweaks (add any you need)
smri_pca_dim: 0
```

All keys are **passed unchanged** to the internal state dictionary, so any new agent you write can read the values directly.

---  

## ▶️ Running the pipeline  

The CLI is a thin wrapper around the agents defined in `pipeline.py`.

```bash
# Run with the default config
python main.py

# Use a custom config file
python main.py --config my_custom_config.yaml

# Override a single key from the command line (useful for quick experiments)
python main.py --config config.yaml --override verbose=False
```

### What you will see (when `verbose: true`)

```
Pipeline: loading data
Loaded 4678 FC matrices
Pipeline: aligned subjects 4250
Agent: fc_features
...
Model selection: starting
...
Ensemble: soft voting...
Late Fusion Agent: starting per-modality training
...
TransformerFusion Agent: training...
BrainNetCNN Agent: training...
Super Ensemble Agent: combining all OOF probs...
Pipeline: completed
Best model: super_wt_search   (accuracy = 0.8423)
```

All intermediate artefacts are saved automatically:

| Artefact | Where it is stored |
|----------|---------------------|
| Full (JSON‑safe) state | `<output_dir>/results.json` |
| Human‑readable summary | `<output_dir>/summary.txt` |
| Trained sklearn / XGBoost models | `<output_dir>/models/<model>.pkl` |
| Torch models (GCN, Transformer, BrainNetCNN) | `<output_dir>/models/<model>.pt` |
| Scalers / PCA objects | `<output_dir>/scalers/<modality>_scaler.pkl` |
| Auto‑encoder weights (if used) | `<output_dir>/embeddings/ae_encoder.pt` |
| OOF probabilities for every sub‑model | `<output_dir>/oof_probs/` (pickle files) |

You can reload any object later, e.g.:

```python
import joblib, torch
best_lr = joblib.load("results/models/logreg.pkl")
gcn = torch.load("results/models/gcn.pt", map_location="cpu")
```

---  

## 📊 Understanding the agents  

Each processing step is a **pure function** (`state: dict → dict`) that:

1. **Consumes** the keys it needs (e.g., `fc_dict`, `subject_ids`).  
2. **Produces** new keys (e.g., `fc_features`, `adj_batch`).  
3. **Leaves untouched** all other keys so downstream agents can still use them.

The most important agents (found in `src/agentic/pipeline.py`) are:

| Agent | Main output(s) | When you might replace/extend it |
|-------|----------------|----------------------------------|
| `fc_feature_agent` | `fc_features`, `node_features`, `adj_batch`, `fc_edge_count` | Change edge‑selection strategy, add new graph metrics. |
| `smri_agent` | `smri_features` | Swap for raw image processing or a different ROI extraction. |
| `phenotype_agent` | `phenotype_features`, `labels`, `sites` | Include extra categorical covariates (e.g., medication). |
| `harmonization_agent` | Harmonised `fc_features` / `smri_features` | Use a different site‑harmonisation method. |
| `fusion_agent` | `fused_features`, `concat_features` | Replace simple concat with a learned attention block. |
| `embed_agent` | `features_for_training` (AE‑encoded or identity) | Use a VAE, contrastive encoder, or skip embedding. |
| `model_selection_agent` | `model_results`, `best_model`, OOF probabilities | Plug a different cross‑validation scheme or extra classifiers. |
| `ensemble_agent` | Updated `model_results` (soft‑vote, weighted, stack) | Add Bayesian model averaging or other meta‑learners. |
| `late_fusion_agent` | Per‑modality models + blended results | Test hierarchical Bayesian fusion. |
| `multimodal_deep_agent` | Transformer‑fusion results (`tf_oof_probs`) | Upgrade to a full cross‑attention transformer (e.g., Perceiver). |
| `brain_net_agent` | BrainNetCNN results (`bncnn_oof_probs`) | Try newer graph‑CNN variants or attention‑based CNNs. |
| `super_ensemble_agent` | Final meta‑learner (`super_*`) and new `best_model` | Replace Dirichlet weight search with a reinforcement‑learning optimiser. |

All agents can be unit‑tested independently by providing a minimal `state` dict containing the required inputs.

---  

## 📦 Optional dependencies  

| Package | Used for | Install command |
|---------|----------|-----------------|
| `catboost` | CatBoost classifier (model zoo) | `pip install catboost` |
| `lightgbm` | LightGBM classifier & meta‑learner | `pip install lightgbm` |
| `optuna` | Automatic hyper‑parameter search (`optuna_tune_agent`) | `pip install optuna` |
| `tqdm` | Progress bars (nice‑to‑have) | `pip install tqdm` |

If any of these are missing, the pipeline will simply skip the corresponding model/agent and log a warning.

---  

## 🐞 Troubleshooting & FAQ  

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ImportError: cannot import name 'harmonizationLearn'` | `neuroHarmonize` not installed or version mismatch | `pip install neuroHarmonize` (≥ 0.5). |
| GPU not used (`torch.cuda.is_available()` is `False`) | Wrong PyTorch build for your CUDA version | Install a CUDA‑compatible wheel, e.g. `pip install torch==2.*+cu118` for CUDA 11.8. |
| `MemoryError` while loading FC matrices | Very large atlas (e.g., 400 nodes) | Reduce node count, or modify `load_fc_matrices` to use memory‑mapping (`np.load(..., mmap_mode='r')`). |
| `KeyError: 'DX_GROUP'` | Phenotype CSV uses a different column name | Rename the column to `DX_GROUP` or edit `phenotype_agent` to map your naming. |
| All models report ~0.5 accuracy | Labels severely imbalanced or not being read correctly | Verify `DX_GROUP` values, ensure `class_weight="balanced"` is applied (default). |
| `RuntimeError: CUDA out of memory` (BrainNetCNN) | Batch size too large for GPU | Reduce `bsz` inside `brain_net_agent` (default 16). |
| `optuna` not found | Optional dependency missing | `pip install optuna` or set `"tune_trials": 0` in config. |
| Pipeline crashes after `model_select` | Missing `features_for_training` (e.g., you disabled embedding) | Make sure either `fused_features` or `features_for_training` exists; the embed agent creates the latter by default. |

Run the script with `--verbose` to get a full traceback and see the current state printed at each step.


