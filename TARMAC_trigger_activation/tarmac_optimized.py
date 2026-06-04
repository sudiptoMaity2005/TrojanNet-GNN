import os
import sys
import random
import time
import numpy as np
import networkx as nx
import z3

# Add parent directory to path so we can import parser
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parser import NetlistGraphBuilder

class TarmacZ3Builder:
    def __init__(self, graph):
        self.graph = graph
        self.z3_vars = {}
        self.solver = z3.Solver()
        self._build_vars()
        self._build_constraints()

    def _build_vars(self):
        for node in self.graph.nodes():
            self.z3_vars[node] = z3.Bool(node)

    def _build_constraints(self):
        # We need to add constraints for all gates.
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get('type', 'UNKNOWN').upper()
            if node_type == 'INPUT':
                continue # Inputs are free variables
                
            preds = list(self.graph.predecessors(node))
            if not preds:
                continue
                
            z_node = self.z3_vars[node]
            z_preds = [self.z3_vars[p] for p in preds]
            
            if node_type == 'AND':
                self.solver.add(z_node == z3.And(*z_preds))
            elif node_type == 'NAND':
                self.solver.add(z_node == z3.Not(z3.And(*z_preds)))
            elif node_type == 'OR':
                self.solver.add(z_node == z3.Or(*z_preds))
            elif node_type == 'NOR':
                self.solver.add(z_node == z3.Not(z3.Or(*z_preds)))
            elif node_type == 'XOR':
                # z3.Xor only takes 2 args, fold it
                expr = z_preds[0]
                for p in z_preds[1:]:
                    expr = z3.Xor(expr, p)
                self.solver.add(z_node == expr)
            elif node_type == 'XNOR':
                expr = z_preds[0]
                for p in z_preds[1:]:
                    expr = z3.Xor(expr, p)
                self.solver.add(z_node == z3.Not(expr))
            elif node_type == 'NOT':
                self.solver.add(z_node == z3.Not(z_preds[0]))
            elif node_type == 'OUTPUT' or node_type == 'BEHAVIORAL':
                # Treat behavioral assignments or direct outputs as wire copies if single input, else xor
                if len(z_preds) == 1:
                    self.solver.add(z_node == z_preds[0])
                else:
                    expr = z_preds[0]
                    for p in z_preds[1:]:
                        expr = z3.Xor(expr, p)
                    self.solver.add(z_node == expr)
            else:
                # Default unknown gates to simple wire/xor
                if len(z_preds) == 1:
                    self.solver.add(z_node == z_preds[0])
                else:
                    expr = z_preds[0]
                    for p in z_preds[1:]:
                        expr = z3.Xor(expr, p)
                    self.solver.add(z_node == expr)


class FastProbabilitySimulator:
    def __init__(self, graph, num_vectors=100000):
        self.graph = graph
        self.num_vectors = num_vectors
        self.primary_inputs = [n for n, d in graph.nodes(data=True) if d.get('type') == 'INPUT']
        self.nodes = list(self.graph.nodes())
        
        # JIT logic identical to simulator.py but summing 1s instead of transitions
    
    def simulate_probabilities(self):
        vectors = np.random.randint(0, 2, size=(len(self.primary_inputs), self.num_vectors), dtype=np.int8)
        
        node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        num_nodes = len(self.nodes)
        
        code = []
        code.append("import numpy as np")
        code.append("def simulate_fast(vectors, num_vectors):")
        code.append("    state = np.zeros(" + str(num_nodes) + ", dtype=np.int8)")
        code.append("    ones_count = np.zeros(" + str(num_nodes) + ", dtype=np.int32)")
        code.append("    for v_idx in range(num_vectors):")
        
        for i, pi in enumerate(self.primary_inputs):
            code.append(f"        state[{node_to_idx[pi]}] = vectors[{i}, v_idx]")
            
        # Settle combinational logic
        for _ in range(5):
            for node in self.nodes:
                if node in self.primary_inputs:
                    continue
                preds = list(self.graph.predecessors(node))
                idx = node_to_idx[node]
                
                if not preds:
                    code.append(f"        state[{idx}] = 0")
                    continue
                    
                gt = self.graph.nodes[node].get('type', 'UNKNOWN').upper()
                p_idxs = [node_to_idx[p] for p in preds]
                
                if gt == 'AND':
                    expr = " & ".join([f"state[{p}]" for p in p_idxs])
                elif gt == 'NAND':
                    expr = "1 - (" + " & ".join([f"state[{p}]" for p in p_idxs]) + ")"
                elif gt == 'OR':
                    expr = " | ".join([f"state[{p}]" for p in p_idxs])
                elif gt == 'NOR':
                    expr = "1 - (" + " | ".join([f"state[{p}]" for p in p_idxs]) + ")"
                elif gt == 'XOR':
                    expr = " ^ ".join([f"state[{p}]" for p in p_idxs])
                elif gt == 'XNOR':
                    expr = "1 - (" + " ^ ".join([f"state[{p}]" for p in p_idxs]) + ")"
                elif gt == 'NOT':
                    expr = f"1 - state[{p_idxs[0]}]"
                elif gt == 'OUTPUT':
                    expr = f"state[{p_idxs[0]}]"
                else:
                    expr = " ^ ".join([f"state[{p}]" for p in p_idxs])
                    
                code.append(f"        state[{idx}] = {expr}")
                
        # Count ones
        code.append("        ones_count += state")
        code.append("    return ones_count")
        
        local_vars = {}
        exec("\n".join(code), globals(), local_vars)
        simulate_fast = local_vars['simulate_fast']
        
        ones_count = simulate_fast(vectors, self.num_vectors)
        
        probabilities = {}
        for i, node in enumerate(self.nodes):
            probabilities[node] = ones_count[i] / self.num_vectors
            
        return probabilities


def tarmac_test_generation(circuit_file, top_module, num_vectors=10, rare_threshold=0.01):
    print(f"--- Running TARMAC on {circuit_file} ---")
    start_time = time.time()
    
    # 1. Parse Graph
    print("1. Parsing Verilog into Graph...")
    builder = NetlistGraphBuilder(circuit_file, top_module=top_module)
    graph = builder.build()
    print(f"   Nodes: {len(graph.nodes())}, Edges: {len(graph.edges())}")
    
    # 2. Find Potential Trigger Signals (PTS)
    print(f"2. Simulating 10,000 vectors for rare signal profiling (Threshold: {rare_threshold})...")
    sim = FastProbabilitySimulator(graph, num_vectors=10000)
    probs = sim.simulate_probabilities()
    
    print(f"   Min Probability: {min(probs.values()):.4f}, Max Probability: {max(probs.values()):.4f}")
    
    PTS = {}
    for node, p in probs.items():
        if p < rare_threshold:
            PTS[node] = True  # Rare value is 1
        elif p > (1.0 - rare_threshold):
            PTS[node] = False # Rare value is 0
            
    print(f"   Found {len(PTS)} Rare Signals (PTS).")
    if len(PTS) == 0:
        print("   No rare signals found! Adjust threshold.")
        return []

    # 3. Build Z3 Constraints (The Satisfiability Graph Base)
    print("3. Building Z3 Boolean Logic constraints...")
    z3_builder = TarmacZ3Builder(graph)
    solver = z3_builder.solver
    
    # Check if base circuit is satisfiable at all
    if solver.check() != z3.sat:
        print("   Error: Circuit logic is unsatisfiable by default (impossible).")
        return []
    
    # 4. Maximal Clique Sampling (OPTIMIZED TARMAC Algorithm)
    print("4. Executing OPTIMIZED TARMAC Clique Sampling (with Structural Proximity Filter)...")
    test_vectors = []
    
    pis = [n for n, d in graph.nodes(data=True) if d.get('type') == 'INPUT']
    
    # Pre-compute undirected graph for fast neighborhood checks
    print("   -> Preparing Undirected Graph for Proximity Filtering...")
    undirected_graph = graph.to_undirected()
    max_hops = 5  # Maximum graph distance for two rare signals to be considered part of the same trigger
    
    for i in range(num_vectors):
        solver.push() # Save base constraints
        
        # P is candidate set of rare signals
        P = list(PTS.keys())
        random.shuffle(P) # Random sampling
        
        current_clique_nodes = []
        clique_hits = 0
        bypassed_checks = 0
        
        while P:
            v = P.pop()
            
            # --- STRUCTURAL PROXIMITY CHECK (OPTIMIZATION) ---
            # If we already have nodes in the clique, ensure 'v' is physically close (within max_hops)
            # to at least one node in the current clique. Otherwise, it can't be part of the same Trojan!
            if current_clique_nodes:
                # Find all neighbors within max_hops of v
                neighborhood = nx.single_source_shortest_path_length(undirected_graph, v, cutoff=max_hops)
                # If none of the current clique nodes are in v's neighborhood, skip Z3 completely!
                if not any(c_node in neighborhood for c_node in current_clique_nodes):
                    bypassed_checks += 1
                    continue
            # -------------------------------------------------
            
            rv = PTS[v]
            solver.push()
            solver.add(z3_builder.z3_vars[v] == rv)
            
            if solver.check() == z3.sat:
                # Keep constraint, it's satisfiable!
                clique_hits += 1
                current_clique_nodes.append(v)
                # We do NOT pop. The solver maintains this constraint.
            else:
                # Unsatisfiable with current clique, discard it
                solver.pop()
        
        # Finished sampling clique. Extract test vector from model
        solver.check() # Re-verify current SAT state
        model = solver.model()
        
        vector = {}
        for pi in pis:
            val = model.evaluate(z3_builder.z3_vars[pi])
            vector[pi] = 1 if z3.is_true(val) else 0
            
        test_vectors.append(vector)
        print(f"   Vector {i+1}/{num_vectors} generated. Activated {clique_hits} rare signals simultaneously.")
        print(f"   [Optimization] Bypassed Z3 Solver for {bypassed_checks} disconnected nodes!")
        
        solver.pop() # Remove clique constraints, back to base circuit
        
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"TARMAC finished in {elapsed:.2f} seconds.")
    
    return {
        "nodes": len(graph.nodes()),
        "edges": len(graph.edges()),
        "rare_signals": len(PTS),
        "vectors_generated": len(test_vectors),
        "time_seconds": elapsed,
        "test_vectors": test_vectors
    }

if __name__ == "__main__":
    # Test on one of the RS232 circuits
    import glob
    rs232_files = glob.glob("../rs232/trojans/T100/*.v")
    if len(rs232_files) > 0:
        tarmac_test_generation(rs232_files, top_module="uart", num_vectors=5, rare_threshold=0.2)
    else:
        print("RS232 files not found. Ensure you run from TARMAC_trigger_activation directory.")
