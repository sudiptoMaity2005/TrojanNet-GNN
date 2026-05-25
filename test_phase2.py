from parser import NetlistGraphBuilder
from simulator import LogicSimulator

if __name__ == "__main__":
    print("--- Phase 1: Parsing Netlist ---")
    builder = NetlistGraphBuilder("test.v")
    graph = builder.build()
    print(f"Graph built with {graph.number_of_nodes()} nodes.")
    
    print("\n--- Phase 2: Logic Simulation & TP Calculation ---")
    # Simulate 1000 random vectors to calculate Transition Probabilities
    sim = LogicSimulator(graph, num_vectors=1000)
    print(f"Simulating 1000 clock cycles/test vectors...")
    sim.simulate()
    
    # Calculate TP and export to PyTorch Geometric Data format
    pyg_data = sim.embed_tp_and_export()
    
    print("\n--- Simulation Results ---")
    print(f"Transition Probabilities (TP) embedded in nodes:")
    for node in graph.nodes():
        print(f"Node '{node}': TP = {graph.nodes[node].get('TP', 0.0):.3f}")
        
    print("\n--- Final PyTorch Geometric Export ---")
    print(pyg_data)
    print(f"Node Features (x) Shape: {pyg_data.x.shape}")
    print(f"Edge Index (connectivity) Shape: {pyg_data.edge_index.shape}")
