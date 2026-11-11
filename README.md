# TrojanNet-GNN: Hybrid GraphSAGE and Cost-Sensitive Ensemble Learning for Hardware Trojan Detection

[![Journal](https://img.shields.io/badge/Journal-JETTA_(Springer_Nature)-blue.svg)](https://www.springer.com/journal/10836)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch--Geometric-2.3+-orange.svg)](https://pyg.org/)

Official repository for the paper:  
**"Effective Detection of Hardware Trojan Locations in Gate-Level Netlists Using Hybrid GraphSAGE and Cost-Sensitive Ensemble Learning"**  
*Journal of Electronic Testing: Theory and Applications (JETTA), Springer Nature.*  
**Authors:** Sudipto Maity, Surajit Dutta, Anindan Mondal, and Bibhash Sen (National Institute of Technology Durgapur, India).

---

## 📖 Overview

Detecting gate-level Hardware Trojans (HTs) in third-party and synthesized Intellectual Property (IP) cores is a critical challenge in modern hardware security. Hardware Trojans exhibit extreme class imbalance (often fewer than $0.01\%$ malicious gates in netlists exceeding hundreds of thousands of logic gates).

This repository provides an autonomous, **golden-reference-free** hardware Trojan detection framework combining **Inductive Graph Sampling and Aggregation (GraphSAGE)** with **Cost-Sensitive Ensemble Classifiers** (Extra Trees, Random Forest, XGBoost, CatBoost, SVM).

```
Gate-Level Netlist DAG G=(V, E)
             │
             ▼
  8-D Behavioral Node Features ──► 3-Layer GraphSAGE Encoder ──► 15-D Topological Embeddings
  (TP, Inactivity, Fan-In/Out)            (with Jumping Knowledge)             │
             │                                                                 │
             └─────────────────────── Concatenation ───────────────────────────┘
                                           │
                                           ▼
                                 23-D Hybrid Feature Vector
                                           │
                                           ▼
                            Cost-Sensitive Ensemble Classification
                            (Extra Trees, XGBoost, CatBoost, RF, SVM)
```

---

## 🔬 Benchmark Dataset (108 Circuits across 13 Families)

The framework is evaluated under a strict **Leave-One-Circuit-Out (LOCO) Cross-Family Protocol** across all **108 unique benchmark netlists** (over 1.48 million gates total):

| Benchmark Family | Source Suite | Circuit Count | Example Trojan Types / Trigger Mechanism |
|---|---|:---:|---|
| **AES-128** | Trust-Hub | 28 | LFSR counter triggers, side-channel leakage, state alteration (`AES_T100`–`T2800`) |
| **RS232** | Trust-Hub | 15 | Baud rate disruption, transmitter denial of service, character triggers |
| **b19** | ITC-99 | 2 | Large sequential dual-core microprocessor benchmark |
| **EthernetMAC10GE** | Trust-Hub | 4 | Frame alteration, denial of service in 10-Gigabit Media Access Controller |
| **wb_conmax** | Trust-Hub | 3 | Wishbone interconnect bus contention and master denial of service |
| **c2670** | ISCAS-85 | 7 | Combinational ALU and control logic benchmarks |
| **c3540** | ISCAS-85 | 7 | 8-bit ALU and combinational processing netlists |
| **c5315** | ISCAS-85 | 7 | 9-bit ALU with lookahead carry generators |
| **c6288** | ISCAS-85 | 7 | 16-bit array multiplier |
| **s13207** | ISCAS-89 | 7 | Sequential benchmark with peripheral controllers |
| **s1423** | ISCAS-89 | 7 | Sequential control netlists with multi-stage flip-flops |
| **s15850** | ISCAS-89 | 7 | Sequential arithmetic and communication blocks |
| **s35932** | ISCAS-89 | 7 | Large sequential digital processing circuit |
| **Total** | | **108** | **1,489,029 Total Gates** |

A complete per-circuit breakdown with exact gate counts, Trojan gate counts, and verified hierarchical netlist names is cataloged in [`data/benchmark_circuits_catalog.csv`](data/benchmark_circuits_catalog.csv).

---

## 🧮 Behavioral Feature Formulation ($f_1$ – $f_8$)

To ensure **100% golden-reference-free autonomous operation**, all circuits are evaluated without differential comparison against clean netlists ($f_2 \equiv 0.0$ clamped for all families):

1. **$f_1(v) = TP(v)$**: Empirical Transition Probability under $M=10{,}000$ simulation vectors:
   $$TP(v) = \frac{1}{M}\sum_{k=1}^M \mathbb{I}(v^{(k)} \neq v^{(k-1)})$$
2. **$f_2(v) = 0.0$**: Autonomous setting (eliminates clean-twin leakage across all 13 families).
3. **$f_3(v) = P_0(v) = 1.0 - TP(v)$**: Inactivity probability ($r(f_1, f_3) = -1.0000$).
4. **$f_4(v) = \mu_{TP}(v)$**: Mean transition probability of 1-hop topological neighbors:
   $$\mu_{TP}(v) = \frac{1}{|N(v)|} \sum_{u \in N(v)} TP(u)$$
5. **$f_5(v) = \Delta TP_{\text{nb}}(v)$**: Sum of neighbor transition probability differences:
   $$\Delta TP_{\text{nb}}(v) = \sum_{u \in N(v)} |TP(v) - TP(u)|$$
6. **$f_6(v) = z_{TP}(v)$**: Local neighborhood transition z-score:
   $$z_{TP}(v) = \frac{TP(v) - \mu_{TP}(v)}{\sigma_{TP}(v) + 10^{-7}}$$
7. **$f_7(v) = \text{deg}_{\text{in}}(v)$**: Gate fan-in / input degree.
8. **$f_8(v) = \text{deg}_{\text{out}}(v)$**: Gate fan-out / load degree.

Verification matrices:
- [`results/Table_S3_correlation.csv`](results/Table_S3_correlation.csv): $7 \times 7$ Pearson correlation matrix of the 7 active features ($r(f_1, f_3) = -1.0000$).
- [`results/Table_S4_pca.csv`](results/Table_S4_pca.csv): 7-component PCA explained variance table ($\text{PC7} = 0.0000$ due to exact collinearity $f_3 = 1 - f_1$).

---

## 📂 Repository Structure

```
TrojanNet-GNN/
├── data/
│   └── benchmark_circuits_catalog.csv   # Catalog of all 108 evaluated circuits with gate counts
├── docs/
│   ├── ANSWERS_TO_OPEN_QUESTIONS_A1_A5.md  # Detailed technical response to review questions
│   ├── RESPONSE_TO_REVIEWERS_TECHNICAL_ANSWERS.md # Point-by-point reviewer rebuttal
│   ├── response-to-reviewers.pdf        # Compiled response PDF
│   └── reviewer_response_complete.tex   # LaTeX source for revision documents
├── results/
│   ├── Table_S3_correlation.csv         # Clean 7x7 Pearson feature correlation matrix
│   ├── Table_S4_pca.csv                 # Clean 7-component PCA variance decomposition
│   ├── aes_trojan_nodes_with_names.csv  # Verified list of all 401 Trojan gates in AES-128
│   ├── non_iscas_trojan_nodes_with_names.csv # 454 positive gates in B19, Ethernet, wb_conmax
│   ├── rs232_trojan_nodes_with_verdict.csv # Trust-Hub verified positive gates in RS232
│   ├── determinism_run1.csv             # Determinism check Run 1 (Seed 42, Extra Trees)
│   ├── determinism_run2.csv             # Determinism check Run 2 (Bitwise identical)
│   ├── strict_loco_summary.csv          # Comprehensive strict-LOCO multi-seed summary
│   └── htpipe.pdf                       # Vector pipeline architecture diagram (Fig. 2)
├── src/
│   ├── data_preparation/               # Feature extraction and netlist graph builders
│   │   ├── aes_pipeline.py              # AES netlist behavioral simulation & feature extractor
│   │   ├── rs232_pipeline.py            # RS232 netlist feature pipeline
│   │   ├── b19_pipeline.py              # ITC-99 b19 processor pipeline
│   │   ├── ethernet_pipeline.py         # 10GE Ethernet MAC pipeline
│   │   └── wb_conmax_pipeline.py        # Wishbone interconnect pipeline
│   └── models/                          # GNN and downstream ensemble classifiers
│       ├── run_strict_loco_6classifiers.py # Strict cross-family LOCO across 13 families & 5 seeds
│       ├── run_complete_ablations_loco.py  # Reviewer 1 & 2 ablation suite (8 configurations)
│       └── localization_pipeline.py     # Unsupervised propagation anomaly detection
├── requirements.txt                     # Python dependencies
└── README.md
```

---

## 🚀 Reproduction & Usage

### 1. Installation

```bash
git clone https://github.com/sudiptoMaity2005/TrojanNet-GNN.git
cd TrojanNet-GNN
pip install -r requirements.txt
```

### 2. Running Strict Cross-Family LOCO Evaluation (5 Seeds)

Evaluates 6 classifiers (GNN Head, SVM, XGBoost, CatBoost, Random Forest, Extra Trees) under strict cross-family hold-out across seeds `[42, 123, 2024, 7, 999]` with $f_2 \equiv 0.0$:

```bash
python3 src/models/run_strict_loco_6classifiers.py --device cuda --save_gate_scores
```

### 3. Running Reviewer Ablations Suite (Task B2)

Executes all 8 ablation configurations (representations `raw_8`, `emb_15`, `hybrid_23`; architecture `no_jk`, `no_residual`; unweighted loss; tree depth sweeps `1, 2, 4, 5`; and feature drop groups):

```bash
python3 src/models/run_complete_ablations_loco.py --device cuda
```

---

## 📜 Citation

If you find this work or benchmark dataset useful, please cite our JETTA paper:

```bibtex
@article{maity2024trojannet,
  title     = {Effective Detection of Hardware Trojan Locations in Gate-Level Netlists Using Hybrid GraphSAGE and Cost-Sensitive Ensemble Learning},
  author    = {Maity, Sudipto and Dutta, Surajit and Mondal, Anindan and Sen, Bibhash},
  journal   = {Journal of Electronic Testing: Theory and Applications (JETTA)},
  publisher = {Springer Nature},
  year      = {2024}
}
```

---

## 📄 License
This repository is licensed under the [MIT License](LICENSE).
