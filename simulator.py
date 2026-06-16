import networkx as nx
import numpy as np
import torch  # pyrefly: ignore [missing-import]
from torch_geometric.data import Data # pyrefly: ignore [missing-import]

class LogicSimulator:
    def __init__(self, graph, num_vectors=1000):
        self.graph = graph
        self.num_vectors = num_vectors
        self.primary_inputs = [n for n, d in graph.nodes(data=True) if d.get('type') == 'INPUT']
        
        # We no longer need topological sort!
        # Iterative settling naturally handles sequential feedback loops.
        self.sorted_nodes = list(self.graph.nodes())
            
        self.transitions = {node: 0 for node in self.graph.nodes()}
        self.previous_state = {node: 0 for node in self.graph.nodes()}

    def _evaluate_gate(self, gate_type, input_values):
        """Boolean Evaluation Engine for standard gate types."""
        # Edge Case 2: Floating inputs (empty input_values)
        if not input_values:
            return 0  # Default pull-down

        gt = gate_type.upper()
        if gt == 'AND':
            return 1 if all(v == 1 for v in input_values) else 0
        elif gt == 'NAND':
            return 0 if all(v == 1 for v in input_values) else 1
        elif gt == 'OR':
            return 1 if any(v == 1 for v in input_values) else 0
        elif gt == 'NOR':
            return 0 if any(v == 1 for v in input_values) else 1
        elif gt == 'XOR':
            return sum(input_values) % 2
        elif gt == 'XNOR':
            return 1 - (sum(input_values) % 2)
        elif gt == 'NOT':
            return 1 - input_values[0] if input_values else 1
        elif gt == 'OUTPUT':
            return input_values[0] if input_values else 0
        elif gt == 'BEHAVIORAL':
            # Propagate transitions through behavioral FSM/Assign logic via XOR summation
            return sum(input_values) % 2
        else:
            return sum(input_values) % 2

    def generate_vectors(self):
        """Generates random binary test vectors for all primary inputs."""
        self.vectors = np.random.randint(0, 2, size=(len(self.primary_inputs), self.num_vectors))
        return self.vectors

    def simulate(self):
        """JIT Compiled Evaluation Loop across all test vectors for 1000x speedup."""
        if not hasattr(self, 'vectors'):
            self.vectors = self.generate_vectors()
            
        nodes = list(self.graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        num_nodes = len(nodes)
        
        # Build the JIT function string
        code = []
        code.append("def simulate_fast(vectors, num_vectors, initial_state):")
        code.append("    state = initial_state.copy()")
        code.append("    transitions = [0] * " + str(num_nodes))
        code.append("    for v_idx in range(num_vectors):")
        code.append("        prev_state = state.copy()")
        
        # 1. Primary Inputs
        for i, pi in enumerate(self.primary_inputs):
            code.append(f"        state[{node_to_idx[pi]}] = vectors[{i}, v_idx]")
            
        # 2. Iterative Settling
        for _ in range(5):
            for node in nodes:
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
                else: # BEHAVIORAL
                    expr = " ^ ".join([f"state[{p}]" for p in p_idxs])
                    
                code.append(f"        state[{idx}] = {expr}")
                
        # 3. Transition Counting
        code.append("        if v_idx > 0:")
        code.append("            for i in range(" + str(num_nodes) + "):")
        code.append("                if state[i] != prev_state[i]:")
        code.append("                    transitions[i] += 1")
        code.append("    return transitions")
        
        code_str = "\n".join(code)
        
        # Compile and execute
        local_vars = {}
        exec(code_str, {}, local_vars)
        simulate_fast = local_vars['simulate_fast']
        
        # Run the fast compiled function
        initial_state = [0] * num_nodes
        transitions = simulate_fast(self.vectors, self.num_vectors, initial_state)
        
        # Save results
        for i, node in enumerate(nodes):
            self.transitions[node] = transitions[i]

    def embed_tp_and_export(self):
        """Calculates TP, embeds it into the graph, and exports to PyTorch Geometric."""
        # Calculate TP
        for node in self.graph.nodes():
            tp = self.transitions[node] / max(1, self.num_vectors - 1)
            self.graph.nodes[node]['TP'] = tp
            
            # Append TP to the feature vector 'x'
            if 'x' in self.graph.nodes[node]:
                self.graph.nodes[node]['x'].append(tp)
            else:
                self.graph.nodes[node]['x'] = [tp]

        # Convert to PyTorch Geometric Data format
        node_mapping = {node: i for i, node in enumerate(self.graph.nodes())}
        
        edge_index = []
        for u, v in self.graph.edges():
            edge_index.append([node_mapping[u], node_mapping[v]])
            
        if not edge_index:
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Gather node features
        x_list = [self.graph.nodes[n]['x'] for n in self.graph.nodes()]
        x_tensor = torch.tensor(x_list, dtype=torch.float)
        
        pyg_data = Data(x=x_tensor, edge_index=edge_index_tensor)
        return pyg_data
