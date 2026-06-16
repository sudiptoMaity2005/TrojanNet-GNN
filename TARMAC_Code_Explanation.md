# TARMAC Code Explanation (`tarmac_optimized.py`)

This document provides a detailed, chunk-by-chunk and line-by-line explanation of the `tarmac_optimized.py` script.

## 1. Imports and Setup (Lines 1-11)
```python
import os
import sys
import random
import time
import numpy as np
import networkx as nx
import z3

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parser import NetlistGraphBuilder
```
*   **Lines 1-7:** Standard Python imports. `numpy` is used for fast math matrices. `networkx` is used for graph (node/edge) operations. `z3` is Microsoft's Satisfiability solver.
*   **Lines 10-11:** Adds the parent directory to the Python system path so the script can import our custom `NetlistGraphBuilder` class from `parser.py`, which is responsible for turning Verilog files into NetworkX graphs.

## 2. Z3 Boolean Constraints Builder (Lines 13-78)
```python
class TarmacZ3Builder:
    def __init__(self, graph):
        self.graph = graph
        self.z3_vars = {}
        self.solver = z3.Solver()
        self._build_vars()
        self._build_constraints()
```
*   **Line 13-20:** This class takes the circuit graph and converts it into a giant mathematical equation for the Z3 solver. It creates an empty dictionary `z3_vars` to hold Z3 variables and initializes `z3.Solver()`.

```python
    def _build_vars(self):
        for node in self.graph.nodes():
            self.z3_vars[node] = z3.Bool(node)
```
*   **Line 21-23:** Loops through every single wire (node) in the circuit graph and creates a corresponding Boolean (True/False or 1/0) variable inside the Z3 engine.

```python
    def _build_constraints(self):
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get('type', 'UNKNOWN').upper()
            if node_type == 'INPUT': continue
            
            preds = list(self.graph.predecessors(node))
            z_node = self.z3_vars[node]
            z_preds = [self.z3_vars[p] for p in preds]
```
*   **Lines 25-37:** This loops through all nodes again to apply logic rules. If a node is an `INPUT` pin, it skips it because inputs don't have logic rules (they are decided by the user). It grabs the node's predecessors (the wires coming into the logic gate) and gets their Z3 variables.

```python
            if node_type == 'AND':
                self.solver.add(z_node == z3.And(*z_preds))
            elif node_type == 'NAND':
                self.solver.add(z_node == z3.Not(z3.And(*z_preds)))
            # ... handles OR, NOR, XOR, XNOR, NOT
```
*   **Lines 39-78:** This giant block checks what type of logic gate the node represents (`AND`, `NAND`, `XOR`, etc.). It then adds a mathematical constraint to the Z3 solver. For example, if it's an AND gate, it tells Z3: *"The output variable MUST equal the mathematical AND of all the input variables."*

## 3. Fast Probability Simulator (Lines 80-155)
```python
class FastProbabilitySimulator:
    def __init__(self, graph, num_vectors=100000):
        # Setup graph and inputs
```
*   **Lines 80-94:** This class generates random 1s and 0s and simulates the logic gates to find which signals trigger rarely.

```python
    def simulate_probabilities(self):
        vectors = np.random.randint(0, 2, size=(len(self.primary_inputs), self.num_vectors), dtype=np.int8)
```
*   **Line 90:** Uses Numpy to instantly generate an array of 10,000 completely random 1s and 0s for every primary input pin of the circuit.

```python
        code = []
        code.append("import numpy as np")
        code.append("def simulate_fast(vectors, num_vectors):")
        # ... builds dynamic python code string
```
*   **Lines 95-148:** This is a highly advanced Python trick called **JIT Compilation (Just-In-Time)**. Simulating 10,000 vectors through Python loops is extremely slow. Instead, these lines actually write a completely new Python function *as a string* in memory, highly optimized for this specific circuit's layout using fast Numpy bitwise operations (`&`, `|`, `^`), and then executes the string using `exec()`.

```python
        ones_count = simulate_fast(vectors, self.num_vectors)
        
        probabilities = {}
        for i, node in enumerate(self.nodes):
            probabilities[node] = ones_count[i] / self.num_vectors
        return probabilities
```
*   **Lines 149-155:** Runs the ultra-fast simulated function. It takes the total number of times a wire became '1' (`ones_count`), divides it by 10,000 (`num_vectors`), and stores the probability. 

## 4. The Core TARMAC Algorithm (Lines 158-273)
```python
def tarmac_test_generation(circuit_file, top_module, num_vectors=10, rare_threshold=0.01):
```
*   **Line 158:** The main function. We want to generate `10` highly targeted test vectors. A signal is "rare" if its probability is less than `0.01` (1%).

```python
    builder = NetlistGraphBuilder(circuit_file, top_module=top_module)
    graph = builder.build()
```
*   **Lines 163-166:** Parses the raw Verilog `.v` file into the NetworkX graph.

```python
    sim = FastProbabilitySimulator(graph, num_vectors=10000)
    probs = sim.simulate_probabilities()
```
*   **Lines 169-173:** Runs the `FastProbabilitySimulator` from section 3 to get the probabilities of every wire.

```python
    PTS = {}
    for node, p in probs.items():
        if p < rare_threshold:
            PTS[node] = True  
        elif p > (1.0 - rare_threshold):
            PTS[node] = False
```
*   **Lines 175-185:** Iterates through all probabilities. If a node is mostly `0` (probability < 1%), we store `True` (1) as its rare state. If a node is mostly `1` (probability > 99%), we store `False` (0) as its rare state. These are our **Potential Trigger Signals (PTS)**.

```python
    z3_builder = TarmacZ3Builder(graph)
    solver = z3_builder.solver
    if solver.check() != z3.sat:
        return []
```
*   **Lines 188-195:** Feeds the circuit into the `TarmacZ3Builder`. It runs a quick safety check: if the base circuit logic is mathematically impossible (`!= z3.sat`), it crashes.

```python
    undirected_graph = graph.to_undirected()
    max_hops = 5
```
*   **Lines 204-207:** Prepares for the "Proximity Optimization". It creates a two-way (undirected) graph so we can easily measure the physical distance between gates. `max_hops = 5` means gates must be within 5 connections to be considered part of the same Trojan.

```python
    for i in range(num_vectors):
        solver.push() # Save base constraints
        P = list(PTS.keys())
        random.shuffle(P) # Random sampling
```
*   **Lines 208-213:** Starts the loop to generate test vectors. `solver.push()` saves the clean state of the solver. We take the list of rare signals (`P`) and randomize their order.

```python
        while P:
            v = P.pop()
            
            # --- STRUCTURAL PROXIMITY CHECK ---
            if current_clique_nodes:
                neighborhood = nx.single_source_shortest_path_length(undirected_graph, v, cutoff=max_hops)
                if not any(c_node in neighborhood for c_node in current_clique_nodes):
                    continue
```
*   **Lines 219-232:** The optimization! We pop a rare signal `v`. If we already have rare signals stored in our `current_clique_nodes`, we check the 5-hop physical neighborhood around `v`. If none of our current clique nodes are nearby, we instantly `continue` (skip), bypassing the heavy math solver entirely!

```python
            rv = PTS[v]
            solver.push()
            solver.add(z3_builder.z3_vars[v] == rv)
            
            if solver.check() == z3.sat:
                clique_hits += 1
                current_clique_nodes.append(v)
            else:
                solver.pop()
```
*   **Lines 234-245:** The core Markov Chain/Z3 check. We push a new state to the solver and explicitly demand: *"Force variable `v` to its rare state `rv`"*. We run `solver.check()`. If it returns `z3.sat` (Satisfiable), it means the circuit logic physically allows this. We keep it in our clique. If it's impossible, we `solver.pop()` (delete the constraint).

```python
        solver.check()
        model = solver.model()
        vector = {}
        for pi in pis:
            val = model.evaluate(z3_builder.z3_vars[pi])
            vector[pi] = 1 if z3.is_true(val) else 0
        test_vectors.append(vector)
        solver.pop()
```
*   **Lines 248-260:** Once all rare nodes have been tested, we ask the Z3 solver to generate the actual mathematical model that satisfied the massive clique. We extract the exact 1s and 0s required for the Primary Inputs (`pis`) to make this happen, save it as a `vector`, and `pop()` to reset the solver for the next loop!
