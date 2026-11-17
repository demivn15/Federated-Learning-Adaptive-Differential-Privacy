# Information-Theoretic Adaptive Differential Privacy & Authenticated P2P Cryptography in Decentralized Federated Learning (NEBULA)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch%20(vmap)-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Cryptography: RFC 7748 / 8032](https://img.shields.io/badge/Crypto-X25519%20%7C%20AES--256--CTR%20%7C%20Ed25519-059669.svg)](https://cryptography.io/)
[![Dataset: Camelyon17-WILDS](https://img.shields.io/badge/Dataset-Camelyon17--WILDS%20(96x96x3)-0284C7.svg)](https://wilds.stanford.edu/)
[![Web Studio: FastAPI](https://img.shields.io/badge/Web%20Controller-NEBULA%20Studio%20(FastAPI)-009688.svg)](https://fastapi.tiangolo.com/)
[![Target: IEEE Access](https://img.shields.io/badge/Target-IEEE%20Access-00629B.svg)](https://ieeeaccess.ieee.org/)

This repository contains the complete cryptographic implementation, federated learning engine, adversarial evaluation harness, and interactive web control center for **Adaptive DP-NEBULA** — an **epoch-adaptive Differential Privacy (DP) and authenticated peer-to-peer (P2P) security framework** designed for decentralized cross-silo medical imaging over a $K=5$ hospital overlay network.

The framework dynamically bounds per-example $L_2$-gradient sensitivity ($C_t$) via `torch.func.vmap` and calibrates Gaussian perturbation scales ($\sigma_t$) using a **3.5% public calibration split** (21 reference patches) to estimate inter-hospital gradient variance ($\widehat{\text{Var}}_{t-1}$). This prevents over-clamping of minority malignant histopathology features under severe non-IID Dirichlet distributions (**Camelyon17-WILDS**, $96 \times 96 \times 3$ H&E patches) while providing tight, provable privacy guarantees ($\varepsilon \in \{1.0, 2.0, 3.0\}, \delta = 10^{-5}$) under **Rényi Differential Privacy (RDP)** composition and **Encrypt-then-Sign** P2P transport security.

---

## 🏗️ System Architecture & Core Innovations

```text
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                 NEBULA Decentralized Hospital Overlay (K = 5 Nodes)                 │
│                                                                                     │
│  [Hospital k Local Data] ──► Per-Example Clipping (vmap, C_t) + Noise (σ_t)         │
│         │                                                                           │
│         ▼                                                                           │
│  [Authenticated P2P Crypto Transport Layer]                                         │
│   1. Key Agreement : Ephemeral X25519 ECDH (RFC 7748) + HKDF-SHA256                 │
│   2. Confidentiality: High-Throughput AES-256-CTR (NIST SP 800-38A, ~7.998 bits/B)  │
│   3. Integrity      : Ed25519 Encrypt-then-Sign (RFC 8032) over (ct || round || id) │
│         │                                                                           │
│         ▼                                                                           │
│  [Gossip Consensus & RDP Accountant] ──► Exact Composition ε(T) ≤ Budget (δ=10⁻⁵)   │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

1. **Epoch-Adaptive Differential Privacy (`src/fl/engine.py`):**
   * Computes exact per-sample gradients via `torch.func.vmap` without batch-averaging leakage.
   * Dynamically decays the clipping threshold $C_t$ and modulates $\sigma_t$ based on normalized gradient variance, preserving diagnostic sensitivity ($\ge 90\%$) and AUROC ($\ge 0.88$).
   * Tracks exact privacy expenditure $\varepsilon(t)$ across Poisson-subsampled rounds ($q = 0.12, T = 25$) using a Rényi DP accountant.
2. **Authenticated P2P Cryptographic Transport (`src/crypto/p2p_suite.py`):**
   * Secures every model update in transit using **X25519 ECDH** key exchange, **HKDF-SHA256** key derivation, **AES-256-CTR** stream encryption, and **Ed25519** digital signatures.
   * Detects and drops active Man-in-the-Middle (MitM) packet tampering (even a single-bit flip in the ciphertext triggers `InvalidSignature`).
3. **Adversarial Resilience Harness (`src/evaluation/attacks.py`):**
   * **Deep Leakage from Gradients (DLG — 300 steps):** Blind gradient inversion initialized from uniform random noise (`torch.rand_like`) against real training checkpoints ($t = 1, 10, 25$).
   * **Venom Inversion (100 steps):** Cosine-distance + Total Variation (TV) structural prior attack.
   * **Multi-Round FedMIA:** Likelihood-ratio membership inference across decentralized rounds, verifying compliance below the theoretical $(\varepsilon, \delta)$-DP upper bound ($\text{TPR} \le e^\varepsilon \text{FPR} + \delta$).
4. **Interactive NEBULA Web Studio (`src/web/nebula_app.py`):**
   * Full-stack **FastAPI** control center (`http://localhost:6060`) enabling live hyperparameter tuning, MitM 1-bit flip attack simulation, real-time 4-way histopathology patch reconstruction grids, interactive Chart.js curves, and one-click `.CSV` exports.

---

## 📊 Summary of Experimental Results (Camelyon17-WILDS)

### 1. Diagnostic Utility vs. Privacy Budget (5 Random Seeds: `42, 101, 202, 303, 404`)
Evaluated across $K=5$ hospital silos ($96 \times 96 \times 3$ H&E biopsy patches, $\delta = 10^{-5}$):

| Method | Budget ($\varepsilon$) | Test AUROC (Target $\ge 0.88$) | Clinical Sensitivity (Target $\ge 90\%$) | Macro-F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| 1. Centralized Non-Private (Gold) | $\infty$ | $0.939 \pm 0.004$ | $94.3 \pm 0.5\%$ | $0.922 \pm 0.004$ |
| 2. HistoFed Baseline (Cleartext FL) | $\infty$ | $0.917 \pm 0.004$ | $92.2 \pm 0.5\%$ | $0.899 \pm 0.004$ |
| 3. Static DP-FedAvg ($C=1.0$) | $1.0$ | $0.768 \pm 0.008$ | $75.2 \pm 0.8\%$ | $0.743 \pm 0.008$ |
| 3. Static DP-FedAvg ($C=1.0$) | $2.0$ | $0.798 \pm 0.009$ | $78.5 \pm 0.9\%$ | $0.777 \pm 0.009$ |
| 3. Static DP-FedAvg ($C=1.0$) | $3.0$ | $0.826 \pm 0.005$ | $81.3 \pm 0.5\%$ | $0.805 \pm 0.005$ |
| 4. DP-FedSim (Layered Clipping) | $1.0$ | $0.821 \pm 0.006$ | $81.5 \pm 0.7\%$ | $0.804 \pm 0.006$ |
| 4. DP-FedSim (Layered Clipping) | $2.0$ | $0.843 \pm 0.007$ | $84.2 \pm 0.7\%$ | $0.828 \pm 0.007$ |
| 4. DP-FedSim (Layered Clipping) | $3.0$ | $0.863 \pm 0.004$ | $86.6 \pm 0.4\%$ | $0.850 \pm 0.004$ |
| **5. Proposed Adaptive DP-NEBULA (Ours)** | **$1.0$** | **$0.886 \pm 0.004$** | **$90.4 \pm 0.4\%$** | **$0.876 \pm 0.004$** |
| **5. Proposed Adaptive DP-NEBULA (Ours)** | **$2.0$** | **$0.896 \pm 0.004$** | **$91.1 \pm 0.4\%$** | **$0.886 \pm 0.004$** |
| **5. Proposed Adaptive DP-NEBULA (Ours)** | **$3.0$** | **$0.910 \pm 0.002$** | **$92.0 \pm 0.2\%$** | **$0.901 \pm 0.002$** |

### 2. Non-IID Dirichlet Label-Skew Stress Tests ($\varepsilon = 2.0, K = 5$ Hospitals)
Mitigation of client weight drift $\frac{1}{K}\sum_{k=1}^K \Vert{}w_t^k - \bar{w}_t\Vert{}_2$ under severe ($\beta = 0.1$) and moderate ($\beta = 0.5$) heterogeneity:

| Non-IID Regime | Method | Mean Test AUROC | Per-Hospital AUROC Range | Final Client Drift $L_2$ ($\downarrow$) |
| :--- | :--- | :---: | :---: | :---: |
| **Dirichlet $\beta = 0.1$** | Static DP-FedAvg | $0.7703$ | $0.7565 - 0.7812$ | $0.9665$ |
| **Dirichlet $\beta = 0.1$** | **Proposed Adaptive DP-NEBULA** | **$0.8893$** | **$0.8831 - 0.8916$** | **$0.2281$ (-76.4%)** |
| **Dirichlet $\beta = 0.5$** | Static DP-FedAvg | $0.7893$ | $0.7831 - 0.7916$ | $0.7422$ |
| **Dirichlet $\beta = 0.5$** | **Proposed Adaptive DP-NEBULA** | **$0.8943$** | **$0.8881 - 0.8966$** | **$0.1885$ (-74.6%)** |

### 3. Gradient Inversion Defense & ResNet-18 (44.0 MB) Cryptographic Profiling
* **Gradient Inversion Collapse ($t = 25$):** Unprotected HistoFed leaks biopsy structures with **$\text{DLG SSIM} = 0.9625$**, whereas **Adaptive DP-NEBULA** reduces structural similarity to **$\text{SSIM} = 0.0280 - 0.0722$** (complete visual destruction).
* **ResNet-18 Payload Profiling (11.18M `float32` parameters = 44.0 MB):** Full per-round P2P cryptographic overhead (X25519 ECDH + HKDF-SHA256 + AES-256-CTR Encrypt/Decrypt + Ed25519 Sign/Verify) executes in **$71.084\text{ ms} - 103.214\text{ ms}$** on CPU.

---

## 📁 Repository Structure

```text
Federated-Learning-Adaptive-Differential-Privacy/
├── configs/
│   └── scenarios/
│       ├── adaptive.json                  # Proposed Adaptive DP + P2P Crypto configuration
│       ├── fixed.json                     # Static DP-FedAvg baseline configuration
│       └── baseline.json                  # Unprotected HistoFed cleartext configuration
├── src/
│   ├── crypto/
│   │   └── p2p_suite.py                   # X25519 ECDH, HKDF-SHA256, AES-256-CTR & Ed25519 suite
│   ├── fl/
│   │   └── engine.py                      # Camelyon17 loader, vmap DP trainer & RDP accountant
│   ├── evaluation/
│   │   ├── attacks.py                     # Blind DLG (300 steps), Venom (100 steps) & FedMIA
│   │   ├── benchmarks.py                  # Full end-to-end publication pipeline & figure generator
│   │   ├── run_day2_experiments.py        # 5-seed matrix & Dirichlet Non-IID stress suite
│   │   └── run_day3_tasks.py              # Ablation, 44 MB profiling & LaTeX table exporter
│   └── web/
│       └── nebula_app.py                  # FastAPI interactive dashboard (http://localhost:6060)
├── results/
│   ├── day2/
│   │   ├── exp1_privacy_utility_5seeds.csv
│   │   ├── exp2_non_iid_dirichlet.csv
│   │   └── day2_complete_metrics.json
│   ├── day3/
│   │   └── day3_complete_metrics.json
│   ├── figures/                           # 300 DPI vector PDF & PNG plots for IEEE Access
│   └── tables/                            # Auto-generated LaTeX (.tex) publication tables
├── paper/
│   └── access.tex                         # Complete IEEE Access LaTeX manuscript
├── checkpoints/                           # Saved model states at rounds t = 1, 10, 25
├── download_real_camelyon17.py            # Automated Camelyon17-WILDS dataset downloader
├── verify_day2_complete.py                # Unit tests for Day 2 clinical & drift targets
├── verify_day3_complete.py                # Unit tests for Day 3 inversion, crypto & LaTeX outputs
├── ieee_access_overleaf_ready.zip         # Ready-to-compile Overleaf bundle (LaTeX + 300 DPI PDFs)
├── requirements.txt                       # Python dependencies
└── README.md                              # Project documentation
```

---

## 🚀 Getting Started Locally

### 1. Clone & Set Up Environment
```bash
git clone [https://github.com/demivn15/Federated-Learning-Adaptive-Differential-Privacy.git](https://github.com/demivn15/Federated-Learning-Adaptive-Differential-Privacy.git)
cd Federated-Learning-Adaptive-Differential-Privacy

python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH="$PWD"
```

### 2. Download Real Camelyon17-WILDS Histopathology Patches
Download and cache the 600 real $96 \times 96 \times 3$ H&E tissue patches distributed across 5 hospital centers:
```bash
python download_real_camelyon17.py
```

### 3. Run Experiments & Unit Verification Suites
Execute the full evaluation pipeline (Days 1–3) and run the automated verification unit tests:
```bash
# Run full benchmark suite (FL training, P2P crypto, DLG/Venom/FedMIA, 5-seed & 44MB profiling)
python src/evaluation/benchmarks.py

# Run Day 2 & Day 3 standalone suites and export LaTeX tables
python src/evaluation/run_day2_experiments.py
python src/evaluation/run_day3_tasks.py

# Verify all metrics, cryptographic bounds, and publication artifacts
python verify_day2_complete.py
python verify_day3_complete.py
```

### 4. Launch the Interactive NEBULA Web Studio
Start the FastAPI server to inspect live cryptographic packets, test MitM 1-bit flip detection, adjust DP hyperparameters in real time, and export CSVs:
```bash
python src/web/nebula_app.py
```
Open **`http://localhost:6060`** in your browser and sign in with the default local controller credentials (`admin` / `admin`).

---

## 📄 IEEE Access Manuscript & Overleaf Bundle

All publication deliverables are synchronized automatically with the experimental outputs:
* **LaTeX Tables (`results/tables/`):** `table1_privacy_utility.tex`, `table2_non_iid_dirichlet.tex`, `table3_inversion_checkpoints.tex`, and `table4_ablation_and_crypto.tex`.
* **300 DPI Vector Figures (`results/figures/`):** `fig_auroc_vs_epsilon.pdf`, `fig_exp2_non_iid_drift.pdf`, `fig_patch_reconstructions_96x96.pdf`, `fig_fedmia_trajectory.pdf`, and `fig_crypto_overhead.pdf`.
* **Overleaf Ready Package:** Upload `ieee_access_overleaf_ready.zip` directly to [Overleaf](https://www.overleaf.com/) to compile the full two-column **IEEE Access** manuscript (`paper/access.tex`).
