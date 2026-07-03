# Semantic Alignment via MS Factorization

This repository contains the implementation for learning a **shared semantic coordinate system** from multiple embedding models.

The core idea is to represent each embedding model's semantic graph as

\[
G \approx M S
\]

where

- **G** : relation graph constructed from a common word set
- **S** : shared semantic coordinate system (shared signified basis)
- **M** : model-specific semantic transformation

The final goal is **not S itself**, but obtaining comparable **M** matrices across different embedding models.

---

# Repository Structure

```
semantic_alignment/
│
├── config.py                 # Experiment configuration
├── requirements.txt
├── README.md
│
├── data/
│   └── words.txt             # Common word set W
│
├── scripts/
│   ├── run_ms.py             # Main experiment
│   ├── analyze_m.py          # Analyze learned M matrices
│   └── check_cuda.py         # CUDA check
│
├── utils/
│   ├── embeddings.py
│   ├── graph.py
│   ├── factorization_ms.py
│   ├── metrics.py
│   ├── experiments_ms.py
│   ├── alignment.py
│   └── io.py
│
└── outputs/
    ├── csv/
    ├── models/
    └── figures/
```

---

# Requirements

Python 3.11 is recommended.

Create a virtual environment.

Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

Linux

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

# Install PyTorch

PyTorch should be installed **before** installing the remaining dependencies.

## NVIDIA GPU (CUDA 12.1)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## CPU only

```bash
pip install torch torchvision torchaudio
```

---

# Install remaining packages

```bash
pip install -r requirements.txt
```

---

# Verify CUDA

```bash
python scripts/check_cuda.py
```

Expected output

```
CUDA available: True
Device: cuda
GPU: NVIDIA GeForce GTX 1650
Tensor device: cuda:0
OK
```

If CUDA is unavailable, verify

- NVIDIA driver
- CUDA-compatible PyTorch installation

---

# Running the experiment

Run the complete MS factorization pipeline.

```bash
python scripts/run_ms.py
```

The pipeline performs

1. Load word set
2. Generate embeddings
3. Construct relation graph G
4. Automatic relation temperature selection
5. Learn

\[
G \approx MS
\]

6. Save learned

- common S
- model-specific M
- pairwise comparison results

---

# Output files

Typical outputs are

```
outputs/

common_S.pt

final_MS_pair_xxx.pt

selected_relation_temperatures.csv

same_G_summary.csv

cross_model_summary.csv
```

---

# Analyze learned M

After factorization,

```bash
python scripts/analyze_m.py
```

This computes

- Word-wise cosine similarity
- Factor-wise cosine similarity
- RMSE
- L1 difference
- Top-1 factor agreement

for every model pair.

---

# Mathematical formulation

For a common vocabulary

\[
W=\{w_1,\ldots,w_n\}
\]

each embedding model produces embeddings

\[
P
\]

A semantic relation graph is constructed

\[
G_{ij}
=
\exp
\left(
\frac{\cos(p_i,p_j)}{T}
\right)
\]

where

- cosine similarity defines semantic relations
- T is the relation temperature.

We then solve

\[
G \approx MS
\]

where

- G : semantic relation graph
- S : shared semantic coordinate system
- M : model-specific semantic transformation

---

# Relation temperature

Each model automatically selects its own optimal temperature.

Candidate temperatures are

```
0.05
0.07
0.10
0.15
0.25
```

The selected temperature maximizes

- factorization stability
- shared semantic consistency
- reconstruction quality

using repeated random initializations.

---

# Research objective

The objective is **not** matrix factorization itself.

Instead, this work investigates whether different embedding models can be represented within a **shared semantic coordinate system**, enabling direct comparison of their semantic transformations through the learned matrices M.

If successful,

different embedding models become comparable in a common semantic basis.

---

# Tested embedding models

Current experiments include

- sentence-transformers/all-MiniLM-L6-v2
- sentence-transformers/all-mpnet-base-v2
- BAAI/bge-base-en-v1.5
- intfloat/e5-base-v2
- thenlper/gte-base

Additional embedding models can be added by modifying

```
config.py
```

---

# Citation

If this repository contributes to your research, please cite the accompanying paper (to be added).