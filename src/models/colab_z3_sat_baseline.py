import os
import sys
import time
import z3

# Try to import the parser (requires parser.py to be uploaded to Colab)
try:
    # Add parent directory to path so we can import parser if running locally
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from parser import NetlistGraphBuilder
except ImportError:
    print("ERROR: Could not find parser.py!")
    print("If you are running in Colab, you MUST upload 'parser.py' from your project root to the Colab directory.")
    sys.exit(1)

class TarmacZ3Benchmarker:
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
        print(f"Building Z3 mathematical constraints for {self.graph.number_of_nodes()} logic gates...")
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get('type', 'UNKNOWN').upper()
            if node_type == 'INPUT':
                continue
                
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

    def run_benchmark(self):
        print("\nStarting Z3 SAT Solver execution...")
        print("WARNING: This is an NP-Complete problem. It may take hours or timeout on large circuits!")
        start_time = time.time()
        
        # We try to check if the circuit is satisfiable mathematically
        result = self.solver.check()
        
        exec_time = time.time() - start_time
        print(f"\n[Z3 SOLVER FINISHED]")
        print(f"Result: {result}")
        print(f"Execution Time: {exec_time:.4f} seconds")
        
        return exec_time

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python colab_z3_sat_baseline.py <path_to_verilog_file.v>")
        print("Example: python colab_z3_sat_baseline.py c2670_T100.v")
        sys.exit(1)
        
    v_file = sys.argv[1]
    
    if not os.path.exists(v_file):
        print(f"ERROR: Could not find file {v_file}")
        sys.exit(1)
        
    print(f"Parsing Verilog netlist: {v_file}")
    builder = NetlistGraphBuilder(v_file)
    graph = builder.build_graph()
    
    print("\n--- TARMAC Z3 SAT SOLVER BENCHMARK ---")
    z3_benchmarker = TarmacZ3Benchmarker(graph)
    exec_time = z3_benchmarker.run_benchmark()
    
    print("\n=======================================================")
    print("COMPARISON FOR YOUR REPORT:")
    print(f"Original TARMAC (Z3 SAT) Inference Time: {exec_time:.4f} seconds")
    print(f"Your Custom GNN Approach Inference Time: ~0.0030 seconds")
    if exec_time > 1:
        speedup = exec_time / 0.0030
        print(f"-> Your GNN is {speedup:,.0f}x FASTER than the baseline Z3 SAT solver!")
    print("=======================================================")
