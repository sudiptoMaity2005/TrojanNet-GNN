# Detecting Hardware Trojans via Topological Graph Neural Networks: A Spatial Analysis Approach

## Abstract
Hardware Trojans present a severe security threat to modern integrated circuits. Traditional detection methods, such as the TARMAC algorithm, rely on statistical thresholding of trigger probabilities. While sensitive, these methods suffer from severe false positive rates due to their inability to contextualize localized probabilities within the broader circuit topology. In this paper, we propose a novel topological detection framework utilizing Graph Neural Networks (GNNs). By mapping hardware netlists into graph structures and processing transition probabilities as node features through a GraphSAGE architecture, our model successfully discriminates between benign rare-switching gates and malicious structural cliques. Evaluated on ISCAS-85, ISCAS-89, and AES benchmarks, our Global GraphSAGE model achieves state-of-the-art precision (>95%) on combinational logic circuits. However, limitations arise when processing deep sequential circuits, highlighting the necessity for future hybrid temporal-spatial architectures.

## 1. Introduction
The globalization of the semiconductor supply chain has introduced vulnerabilities, allowing rogue entities to insert Hardware Trojans into integrated circuits (ICs). These Trojans are designed to be "stealthy," remaining dormant until activated by a highly specific and rare sequence of inputs. 

### 1.1 The Limitations of TARMAC
Existing detection algorithms, such as TARMAC (Trigger Activation by Randomness and Maximum Clique), identify Trojans by calculating the transition probability (`TP`) of each gate. However, TARMAC suffers from two major limitations:
1. **The False Positive Crisis:** Relying purely on statistical rarity yields a crippling number of false positives in modern, multi-million gate ICs, as many benign gates inherently exhibit low transition probabilities.
2. **NP-Complete Verification:** To filter out these false positives, TARMAC uses Boolean Satisfiability (SAT) solvers (such as Z3) to mathematically verify potential trigger cliques. SAT solving is an NP-Complete mathematical problem, rendering it completely unscalable and prone to execution timeouts for large circuits.

### 1.2 Our Solution: Why GNNs are Better
To address these limitations, we propose replacing the NP-Complete SAT solver verification stage with a Graph Neural Network (GNN). 
- By treating the circuit netlist as a directed graph, the GNN leverages **spatial graph convolution** to learn the structural "cliques" that characterize Hardware Trojans.
- This allows our approach to successfully distinguish between a benign rare gate and a malicious structural clique in **fractions of a second**, completely eliminating the need for slow SAT solvers.

## 2. Methodology

### 2.1 Dataset Generation and Graph Representation
We converted raw gate-level Verilog netlists into PyTorch Geometric `Data` structures. 
- **Nodes ($V$):** Each logic gate is represented as a node. We extracted an 8-dimensional feature vector for each node: `[f1_TP, f2_TPDiff, f3_Rare, f4_NbMean, f5_NbDist, f6_LZ, f7_FanIn, f8_FanOut]`.
- **Edges ($E$):** The physical wiring between gates is represented as a directed `edge_index` tensor.

### 2.2 Global GraphSAGE Architecture
To ensure scalability and generalization across diverse circuit families, we implemented a single Master Global Model utilizing the **GraphSAGE (Graph Sample and Aggregate)** architecture. 
Unlike standard Graph Convolutional Networks (GCNs) that require the full graph Laplacian, GraphSAGE generates embeddings by sampling and aggregating features from a node's local neighborhood. This allows the model to learn structural signatures that generalize to unseen circuits. The model concludes with a 2-Layer Multi-Layer Perceptron (MLP) classification head to output the final Trojan probability.

### 2.3 Logarithmic Class Weighting
The primary challenge in Trojan detection is extreme class imbalance (e.g., 99.9% benign gates vs. 0.1% Trojan gates). Standard cross-entropy loss functions collapse under this imbalance, resulting in a trivial model that classifies all gates as benign. 
To counteract this, we applied **Logarithmic Class Weighting** to the PyTorch `CrossEntropyLoss` function. By exponentially scaling the gradient penalties for minority class misclassifications, the network is forced to prioritize the identification of the rare Trojan nodes.

## 3. Experimental Results

### 3.1 Baseline Comparison
To establish a baseline, we evaluated the original statistical TARMAC methodology. By mathematically isolating the optimal threshold for the raw Trigger Probability (`f1_TP`) feature, the baseline approach achieved high recall but suffered catastrophic precision failure (averaging ~1% precision). This confirms that probability thresholding alone cannot reliably distinguish between Trojans and rare benign logic.

### 3.2 GNN Performance
Our Global GraphSAGE model was evaluated across 9 circuit families (Small, Medium, Large, and AES). 
The integration of topological analysis resulted in a massive performance leap:
- **Combinational Circuits:** The model achieved state-of-the-art precision. The `c2670` circuit yielded **95.8% Precision** (F1: 0.80), and `c6288` yielded **94.4% Precision** (F1: 0.78).
- **Cryptographic Cores:** On the AES benchmark, the model achieved near-perfect detection with **100% Recall** and **97.4% Precision** (F1: 0.99).
- **Inference Speed:** Anomalies were inferred in an average of **0.003 seconds**, orders of magnitude faster than traditional Z3 SAT solvers.

## 4. Discussion and Limitations

While the GraphSAGE model excels on static combinational logic, benchmarking revealed significant degradation in F1-scores when processing deep sequential circuits, such as `s13207` and `s15850`.

### 4.1 The Problem We Face: Sequential Circuits
This limitation stems from the "Memory State Problem." Sequential circuits rely heavily on Flip-Flops and Registers, creating feedback loops that hold logic states across multiple clock cycles. Our current GraphSAGE architecture is purely **Spatial**; it processes a static snapshot of the circuit. Because it lacks a temporal dimension, it cannot observe how logic values transition through flip-flops over time, rendering it blind to Trojans that require complex temporal activation sequences.

## 5. Conclusion and Future Work

In Phase 1, we successfully demonstrated that replacing NP-Complete SAT solvers with Graph Neural Networks drastically improves the precision and scalability of Hardware Trojan detection in combinational circuits. By utilizing spatial message passing, the GNN successfully filters out the false positives that cripple traditional statistical methods.

To overcome the current limitations regarding sequential circuits, Phase 2 of this research will transition from a purely spatial architecture to a **Temporal Graph Network**. 

### 5.1 The Hybrid Solution: GGNN (GNN + RNN)
We propose the implementation of a **Gated Graph Neural Network (GGNN)**. By embedding a Recurrent Neural Network (RNN)—such as a Gated Recurrent Unit (GRU)—directly within the graph convolution layers, the model becomes a powerful Hybrid:
1. **The GNN Component:** Will continue to process the spatial topology of the logic gates.
2. **The RNN Component:** Will process the temporal state changes across multiple clock cycles, learning the behavior of the flip-flops.

This unified hybrid architecture aims to achieve state-of-the-art precision across **both** combinational and sequential integrated circuits simultaneously.
