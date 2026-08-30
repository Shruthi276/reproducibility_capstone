# Question 4 - Capstone: End-to-End Reproducibility Drill

This repository contains the complete implementation and protocol for **Question 4 — Capstone: End-to-End Reproducibility Drill**.

---

##  Table of Contents

- [Repository Structure](#-repository-structure)
- [Reproducibility Stack & Guarantees](#-reproducibility-stack--guarantees)
- [Step-by-Step Workflow](#-step-by-step-workflow)
  - [Part 1: Partner A — Train, Log, Version, Register (6 marks)](#part-1-partner-a--train-log-version-register-6-marks)
  - [Part 2: Repository Sharing & Isolation Protocol](#part-2-repository-sharing--isolation-protocol)
  - [Part 3: Partner B — Zero-Communication Reproduction (6 marks)](#part-3-partner-b--zero-communication-reproduction-6-marks)
  - [Part 4: Partner B — Verification & MLflow Note Logging (3 marks)](#part-4-partner-b--verification--mlflow-note-logging-3-marks)
- [Reproduction Results](#-reproduction-results)
- [Script Reference](#-script-reference)

---



---

##  Repository Structure

```text
reproducibility_capstone/
├── .dvc/                       # DVC configuration and cache internals
├── .dvcignore                  # DVC ignore rules
├── .gitignore                  # Git ignore rules (ignores raw data, caches, .db)
├── data/
│   ├── .gitignore              # Ignores data/mnist.npz (tracked by DVC)
│   └── mnist.npz.dvc           # DVC pointer containing SHA256/md5 content hash
├── environment.yml             # Pinned Conda/Mamba dependencies
├── README.md                   # Repository documentation (this file)
├── REPRODUCTION_NOTE.md        # Partner B's reproduction report & verification note
├── verify_reproduction.py      # Automated metric comparison & MLflow note logging
└── src/
    ├── prepare_data.py         # Downloads & prepares MNIST dataset
    ├── train.py                # Deterministic MLP training with MLflow tracking
    ├── register_model.py       # Registers model in MLflow & transitions to 'Staging'
    └── fix_artifact_paths.py   # Utility to reconcile MLflow artifact URIs
```

---

##  Reproducibility Stack & Guarantees

| Dimension | Tool | Guarantee / Mechanism |
|---|---|---|
| **Code Versioning** | `Git` | Exact commit hash tagged in MLflow and checked out by Partner B. |
| **Data Versioning** | `DVC` | Exact dataset hash (`mnist.npz.dvc`) committed atomically with code. |
| **Environment** | `Conda / Mamba` | Fully pinned `environment.yml` specifying Python, scikit-learn, MLflow, DVC. |
| **Experiment Tracking** | `MLflow` | Logs seed (`42`), hyperparameters, metrics, git commit SHA, dataset hash, model artifact. |
| **Model Registry** | `MLflow Registry` | Registers `MNIST_Reproducibility_Model` and sets stage/alias to `Staging`. |
| **Determinism** | `Python / NumPy / Scikit-Learn` | Global seed initialization across Python `random`, `numpy.random`, and `MLPClassifier(random_state=42)`. |

---

##  Step-by-Step Workflow

### Part 1: Partner A — Train, Log, Version, Register 

#### 1. Setup Environment
```bash
conda env create -f environment.yml
conda activate mnist-mlflow
```

#### 2. Start MLflow Tracking Server
```bash
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --artifacts-destination ./mlartifacts \
    --serve-artifacts \
    --host 0.0.0.0 \
    --port 5000
```

#### 3. Prepare Dataset & Track with DVC
```bash
python src/prepare_data.py
dvc add data/mnist.npz
dvc push
```

#### 4. Train Model & Log Run to MLflow
```bash
python src/train.py
```
- Logs hyperparameters: `learning_rate=0.001`, `batch_size=64`, `epochs=3`, `seed=42`, `hidden_layer_1=128`.
- Logs metrics: `train_loss`, `test_loss`, `train_accuracy`, `test_accuracy`, `final_test_accuracy`.
- Logs tags: `git_commit`, `dataset_sha256`, `dataset_versioning=DVC`, `partner_role=Partner_A`.
- Logs artifacts: `artifacts/results.json`, model signature, serialized scikit-learn model.

#### 5. Register Model and Transition to "Staging"
```bash
python src/register_model.py
```
- Registers model `MNIST_Reproducibility_Model`.
- Transitions version to stage `Staging` (with fallback to the alias `staging` for MLflow 3.x).


---

### Part 2: Repository Sharing & Isolation Protocol

Partner A shares only the repository URL and the target Git commit SHA with Partner B:
- **Repository**: `https://github.com/Shruthi276/reproducibility_capstone.git`
- **Target Commit**: `<commit-sha>`
- **Strict Rule**: No additional communication regarding environment setup, operating system, or data copies is permitted.

---

### Part 3: Partner B — Zero-Communication Reproduction 

Partner B reproduces the result using **strictly** the allowed tool commands:

```bash
# 1. Clone the repository
git clone https://github.com/Shruthi276/reproducibility_capstone.git
cd reproducibility_capstone

# 2. Checkout the specific commit
git checkout <commit-sha>

# 3. Pull and checkout the dataset via DVC
dvc pull && dvc checkout

# 4. Recreate the exact environment
conda env create -f environment.yml
conda activate mnist-mlflow

# 5. Rerun the training script
python src/train.py
```

---

### Part 4: Partner B — Verification & MLflow Note Logging 

Partner B compares the reproduction run with Partner A's baseline run using the automated verification script:

```bash
python verify_reproduction.py \
    --original-run-id <PARTNER_A_RUN_ID> \
    --my-run-id <PARTNER_B_RUN_ID> \
    --tolerance 0.005
```

This script:
1. Validates that the primary metric (`final_test_accuracy`) matches within the stated tolerance (`±0.005`).
2. Checks parameter consistency and logs any environment variations.
3. Automatically writes a detailed markdown note into Partner A's MLflow run description (`mlflow.note.content`).
4. Generates a local report saved in [REPRODUCTION_NOTE.md](file:///c:/Users/ratho/OneDrive/Desktop/reproducibility_capstone/REPRODUCTION_NOTE.md).

---

##  Reproduction Results

As documented in [REPRODUCTION_NOTE.md](file:///c:/Users/ratho/OneDrive/Desktop/reproducibility_capstone/REPRODUCTION_NOTE.md):

| Metric | Partner A | Partner B | Difference (Delta) | Tolerance | Result |
|---|---|---|---|---|---|
| `final_test_accuracy` | **0.969200** | **0.969200** | **0.000000** | `±0.005` | ✅ **MATCH** |
| `final_test_loss` | 0.106200 | 0.106200 | 0.000000 | `±0.005` | ✅ **MATCH** |

### Reproduction Conclusion
> **The metric reproduced with 0.000000 delta (within the ±0.005 tolerance).**
> Code (`Git`), data (`DVC`), environment (`environment.yml`), and configuration/seed (`MLflow`) were successfully pinned end-to-end.

---

## Script Reference

- [`src/prepare_data.py`](file:///c:/Users/ratho/OneDrive/Desktop/reproducibility_capstone/src/prepare_data.py): Downloads MNIST 784 via OpenML, standardizes shapes into flat `(N, 784)` arrays, normalizes pixel values to `[0, 1]`, and saves compressed `data/mnist.npz`.
- [`src/train.py`](file:///c:/Users/ratho/OneDrive/Desktop/reproducibility_capstone/src/train.py): Deterministically seeds RNGs, computes file SHA256 hashes, trains `MLPClassifier`, logs parameters, metrics, tags, and model signature/artifact to MLflow.
- [`src/register_model.py`](file:///c:/Users/ratho/OneDrive/Desktop/reproducibility_capstone/src/register_model.py): Connects to the tracking server, registers the latest run model artifact to `MNIST_Reproducibility_Model`, and transitions to `Staging`.
- [`verify_reproduction.py`](file:///c:/Users/ratho/OneDrive/Desktop/reproducibility_capstone/verify_reproduction.py): Fetches runs from MLflow, compares metrics and hyperparameters, checks tolerance, logs tag `mlflow.note.content`, and writes [REPRODUCTION_NOTE.md](file:///c:/Users/ratho/OneDrive/Desktop/reproducibility_capstone/REPRODUCTION_NOTE.md).
- [`environment.yml`](file:///c:/Users/ratho/OneDrive/Desktop/reproducibility_capstone/environment.yml): Conda environment definition containing Python 3.11, NumPy, Scikit-Learn, Pandas, SciPy, MLflow, and DVC.
