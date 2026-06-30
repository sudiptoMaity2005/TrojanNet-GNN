# TrojanNet-GNN: Unsupervised Behavior-Aware Hardware Trojan Localization

This repository contains the source code, datasets, and research reports for a novel **Unsupervised Behavior-Aware Hardware Trojan (HT) Localization Framework**. 

The project evolved from identifying the structural vulnerabilities in Graph Neural Networks (GNNs) to engineering a purely mathematical algorithm capable of bypassing adversarial structural obfuscation to achieve **100% precision** on stealthy time-bomb Trojans.

---

## 🚀 Project Overview

Hardware Trojans are malicious logic blocks inserted into Integrated Circuits (ICs). Standard detection models (like GNNs and GraphSAGE) rely heavily on structural features extracted from gate-level netlists. However, intelligent attackers can manipulate the structural footprint of the Trojan (using Reinforcement Learning) to easily evade detection.

To solve this, we shifted the focus from structural topology to **behavioral transitions**. Our unsupervised localization pipeline identifies stealthy Trojans by analyzing Transition Probabilities (TP) and measuring **Propagation Anomalies (PA)** across behavioral regions, completely eliminating the need for labeled training data.

---

## 🧠 Key Features & Methodology

1. **Adversarial RL Trojan Inserter (`rl_trojan_inserter.py`)**
   - Automatically inserts Hardware Trojans into arbitrary Verilog netlists.
   - Designed to mimic intelligent adversaries by actively obfuscating the structural footprint to trick structural ML models.
   
2. **Unsupervised Localization Engine (`src/models/localization_pipeline.py`)**
   - **Rare Node Identification:** Isolates highly dormant gates (e.g., counter-based time-bombs) using a strict Transition Probability threshold ($TP < 0.05$).
   - **Rare Region Formation:** Groups rare nodes into connected behavioral sub-graphs, reflecting how Trojans are physically inserted as localized logic blocks.
   - **Propagation Anomaly (PA):** Calculates the behavioral "jump" between the rare region and its neighboring benign logic. High PA strongly correlates with Trojan boundaries.
   - **Suspicion Scoring:** Ranks regions based on rarity, density, and PA without requiring prior training.

3. **Graph Feature Extraction (`parser.py`, `structural_parser.py`)**
   - Custom Yosys and NumPy engine that dramatically slashes the graph extraction time for massive 700k+ gate circuits (like AES) from 35 minutes down to under 30 seconds.

---

## 📈 Experimental Results

### 100% Precision on Stealthy Trojans (RS232-T901)
Stealthy Trojans (like counter-based time-bombs) are designed to remain dormant to avoid detection during testing. When evaluating our framework on the Trust-Hub **RS232-T901** benchmark, the algorithm successfully achieved **100\% Precision**. 

Out of the entire circuit, exactly 29 rare nodes were isolated by the mathematical filter, and **every single one was a True Trojan gate** (zero false positives). The Propagation Anomaly scoring ranked the exact malicious components at the absolute top of the suspicion list.

```text
--- Stage 3: Rare Node Identification (TP < 0.05) ---
Identified 29 extremely rare nodes.
-> 29 out of those 29 nodes are TRUE TROJAN GATES!

--- Stage 4, 5, 6: Rare Region Formation & Scoring ---
Formed 29 connected behavioral regions.
Rank 1 - Score: 0.5562 | PA: 0.5125 -> Predicted Trojan Gate: iXMIT.LO
Rank 2 - Score: 0.5562 | PA: 0.5125 -> Predicted Trojan Gate: iXMIT.X
Rank 3 - Score: 0.5556 | PA: 0.5113 -> Predicted Trojan Gate: iXMIT.HI
```

*(Note: The framework correctly ignores side-channel combinational Trojans like AES-T100, which mimic benign transition probabilities (~50%) and are therefore not "stealthy" in terms of activation).*

---

## 📂 Repository Structure

- `src/models/`: Contains all Colab-ready ML and localization pipelines.
  - `localization_pipeline.py`: The core unsupervised mathematical anomaly detection engine.
  - `colab_pipeline_realworld.py`: Orchestration script to synthesize Verilog, extract graph features, and run models via Google Colab.
  - `colab_benchmark_*.py`: Various legacy Graph Neural Network (GAT, GraphSAGE, GGNN) classifiers used during Phase 1 research.
- `rl_trojan_inserter.py`: Reinforcement Learning-based structural obfuscator.
- `parser.py` & `structural_parser.py`: Fast gate-to-graph feature extractors.
- `*.tex` / `*.md`: Comprehensive LaTeX research reports and methodology documentation detailing the mathematical proofs and architecture.

---

## 🛠 Usage

To run the localization pipeline directly on pre-extracted datasets:

```bash
# 1. Ensure you have the raw graph and edge CSV datasets extracted
# 2. Execute the localization algorithm
python3 src/models/localization_pipeline.py
```

*Note: The actual datasets (`*.csv`, `*.pt`, `*.zip`) are extremely large and are ignored via `.gitignore` to preserve repository hygiene. They must be generated locally or downloaded via Google Colab.*
