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
            return input_values[0] if input_values else 0

    def generate_vectors(self):
        """Generates random binary test vectors for all primary inputs."""
        self.vectors = np.random.randint(0, 2, size=(len(self.primary_inputs), self.num_vectors))
        return self.vectors

    def simulate(self):
        """Topological Evaluation Loop across all test vectors."""
        if not hasattr(self, 'vectors'):
            self.vectors = self.generate_vectors()
        
        for v_idx in range(self.num_vectors):
            # Seed the current state with the previous clock cycle's state (essential for FSMs/flip-flops)
            current_state = self.previous_state.copy()
            
            # 1. Assign values to primary inputs for the new clock cycle
            for i, pi in enumerate(self.primary_inputs):
                current_state[pi] = self.vectors[i, v_idx]
                
            # 2. Iterative Settling (evaluate paths until stable)
            for _ in range(5):
                next_state = current_state.copy()
                
                # Evaluate all nodes (order doesn't matter since we iterate)
                for node in self.graph.nodes():
                    if node in self.primary_inputs:
                        continue
                        
                    # Gather inputs from predecessors using CURRENT state
                    predecessors = list(self.graph.predecessors(node))
                    input_values = [current_state.get(pred, 0) for pred in predecessors]
                    
                    gate_type = self.graph.nodes[node].get('type', 'UNKNOWN')
                    next_state[node] = self._evaluate_gate(gate_type, input_values)
                
                # If stable, break early to save compute
                if next_state == current_state:
                    break
                current_state = next_state
                
            # 3. Transition Counting (Side-channel heuristic)
            if v_idx > 0:
                for node in self.graph.nodes():
                    if current_state.get(node, 0) != self.previous_state.get(node, 0):
                        self.transitions[node] += 1
                        
            # Update previous state for the next cycle
            self.previous_state = current_state

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
