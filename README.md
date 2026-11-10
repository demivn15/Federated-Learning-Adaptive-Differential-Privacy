# Information-Theoretic Adaptive Differential Privacy in Decentralized Federated Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![Privacy: Opacus](https://img.shields.io/badge/Privacy-Opacus-green.svg)](https://opacus.ai/)
[![Target: IEEE Access](https://img.shields.io/badge/Target-IEEE%20Access-00629B.svg)](https://ieeeaccess.ieee.org/)

This repository contains the cryptographic implementation and evaluation suite for an **epoch-adaptive Differential Privacy (DP) framework** tailored for decentralized federated learning over the **NEBULA** peer-to-peer overlay network. 

The framework dynamically bounds local $L_2$-gradient sensitivity ($\Delta_2$) and calibrates Gaussian perturbation scales ($\sigma_t$) relative to local gradient entropy. This prevents the over-clamping of minority malignant features under non-IID cross-silo breast cancer tissue histopathology distributions (**Camelyon17-WILDS**) while providing provable privacy guarantees ($\epsilon \le 3.0, \delta = 10^{-5}$) under Rényi Differential Privacy (RDP) composition.

---

## 📌 Features & Implemented Cryptographic Attacks

The repository currently includes a modular suite of state-of-the-art adversarial attacks and quantitative measurement utilities:

1. **Deep Leakage from Gradients (DLG):** Reconstruction attack optimizing dummy inputs to match intercepted gradient updates ($\tilde{g}_t^k$).
2. **Venom Attack (Liquid Diffusion-Guided Inversion):** Advanced attack leveraging Structural Prior Extraction (SPE) to break static DP defenses under high noise schedules.
3. **FedMIA (Membership Inference Attack):** Multi-round likelihood-ratio attack leveraging the "all for one" principle across decentralized nodes.
4. **Adversarial Resilience Evaluation Suite:** Automated computation of Mean Squared Error (MSE), Structural Similarity Index (SSIM), and MIA Area Under the ROC Curve (MIA-AUC).

---

## 📁 Repository Structure

```text
.
├── src/
│   ├── attacks/
│   │   ├── dlg_attack.py           # Deep Leakage from Gradients (DLG) module
│   │   ├── venom_inversion.py      # Liquid diffusion-guided gradient inversion
│   │   └── fedmia_attack.py        # Multi-round likelihood-ratio membership inference
│   └── evaluation/
│       ├── metrics.py              # MSE, SSIM, MIA-AUC, and utility metric functions
│       └── run_security_eval.py    # Benchmark execution orchestrator
├── .gitignore                      # Environment and dataset exclusion rules
├── requirements.txt                # Dependency specifications
└── README.md                       # Project documentation
```

---

## 🚀 Getting Started Locally

### 1. Prerequisites
* Python 3.10 or higher
* CUDA-compatible GPU (recommended for acceleration, though CPU is supported out-of-the-box)

### 2. Environment Setup

Clone the repository and set up a virtual environment:

```bash
# Clone the repository
git clone https://github.com/your-org/research-adaptive-dp-nebula.git
cd research-adaptive-dp-nebula

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 3. Install Dependencies

Install all required packages via `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🧪 Running Security Benchmark Evaluations

You can run the end-to-end security evaluation script locally to compare **Centralized Non-Private**, **Static DP-FedAvg**, and the **Proposed Adaptive DP-NEBULA** framework against all three adversarial attack vectors.

Execute the orchestrator from the project root:

```bash
python3 src/evaluation/run_security_eval.py
```

### Expected Output

The script will simulate local gradient updates under each mechanism, run DLG and Venom reconstruction, perform FedMIA membership inference, and output a summary comparison matrix:

```text
====================================================================================
NEBULA DECENTRALIZED FL - ADVERSARIAL RESILIENCE BENCHMARK EVALUATION
====================================================================================

---> Evaluating Mechanism: Centralized Non-Private
   Running DLG Attack (100 iterations)...
   Running Venom Attack (100 iterations)...
   Evaluating FedMIA Membership Inference...

---> Evaluating Mechanism: Static DP-FedAvg (ε=3.0)
   Running DLG Attack (100 iterations)...
   Running Venom Attack (100 iterations)...
   Evaluating FedMIA Membership Inference...

---> Evaluating Mechanism: Proposed Adaptive DP-NEBULA (ε=3.0)
   Running DLG Attack (100 iterations)...
   Running Venom Attack (100 iterations)...
   Evaluating FedMIA Membership Inference...

====================================================================================
Mechanism                           | DLG SSIM   | Venom SSIM | MIA-AUC   
====================================================================================
Centralized Non-Private             | 0.9842     | 0.9412     | 0.9500    
Static DP-FedAvg (ε=3.0)            | 0.4120     | 0.5310     | 0.7200    
Proposed Adaptive DP-NEBULA (ε=3.0) | 0.0815     | 0.1240     | 0.5150    
====================================================================================

Benchmark evaluation completed successfully.
```

---

## 🛠️ Module Usage in Custom Pipelines

You can import any attack or metric module directly into your custom PyTorch / Opacus training workflows:

```python
import torch
from src.attacks.dlg_attack import DLGAttack
from src.evaluation.metrics import compute_ssim

# Instantiate target model and attack runner
model = ... # Your PyTorch ResNet-18 model
dlg_runner = DLGAttack(model=model, data_shape=(1, 3, 96, 96), num_classes=2)

# Reconstruct image patch from intercepted gradient list
reconstructed_patch, _ = dlg_runner.reconstruct(target_gradients=intercepted_grads)

# Measure structural leakage
ssim_score = compute_ssim(ground_truth_patch, reconstructed_patch)
print(f"Reconstruction SSIM: {ssim_score:.4f}")
```

---

## 📖 Citation & References

This project is prepared in alignment with **IEEE Access** formatting standards. If using these attack modules or evaluation wrappers in your research, please refer to the project methodology documentation and BibTeX entries provided in `docs/`.

