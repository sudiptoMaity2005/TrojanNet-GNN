import json
import numpy as np
import time
import sys
import random

def get_rare_wires(mod, num_vectors=5000, K=4):
    """Simulates the circuit to find K rare internal wires for the trigger."""
    print("--- Running NumPy Simulation for TP Analysis ---")
    
    # 1. Build basic DAG for simulation
    import networkx as nx
    graph = nx.DiGraph()
    bit_to_node = {}
    
    for p_name, p_data in mod.get("ports", {}).items():
        for b in p_data["bits"]:
            if isinstance(b, int):
                name = f"PORT_{p_name}_{b}"
                graph.add_node(name, type='INPUT' if p_data['direction']=='input' else 'OUTPUT')
                bit_to_node[b] = name
                
    # Identify constant drivers in cells
    constant_bits = {}
    for c_name, c_data in mod.get("cells", {}).items():
        c_type = c_data["type"].replace("$_", "").replace("_", "").upper()
        g_node = f"GATE_{c_name}"
        graph.add_node(g_node, type=c_type)
        
        for port, bits in c_data.get("connections", {}).items():
            p_dir = c_data.get("port_directions", {}).get(port, "input")
            for b in bits:
                if isinstance(b, int):
                    w_node = bit_to_node.get(b)
                    if not w_node:
                        w_node = f"WIRE_{b}"
                        graph.add_node(w_node, type='WIRE', bit_id=b)
                        bit_to_node[b] = w_node
                    if p_dir == "input":
                        graph.add_edge(w_node, g_node)
                    elif p_dir == "output":
                        graph.add_edge(g_node, w_node)
                elif isinstance(b, str): # e.g. "0" or "1"
                    pass # Constant, ignore for TP routing
                    
    # Simulate
    states = {}
    dag = graph.copy()
    sources = []
    for n, d in dag.nodes(data=True):
        if d.get('type') in ['INPUT', 'DFFP', 'DFFN'] or dag.in_degree(n) == 0:
            sources.append(n)
            dag.remove_edges_from(list(dag.in_edges(n)))
            
    for n in sources:
        states[n] = np.random.randint(0, 2, size=num_vectors, dtype=np.int8)
        
    try:
        topo = list(nx.topological_sort(dag))
    except nx.NetworkXUnfeasible:
        topo = list(dag.nodes())
        
    for n in topo:
        if n in states: continue
        preds = list(graph.predecessors(n))
        if not preds:
            states[n] = np.zeros(num_vectors, dtype=np.int8)
            continue
            
        gt = graph.nodes[n].get('type', 'WIRE')
        if gt in ['WIRE', 'OUTPUT']:
            states[n] = states[preds[0]]
        elif gt == 'NOT':
            states[n] = 1 - states[preds[0]]
        elif gt == 'AND':
            res = states[preds[0]]
            for p in preds[1:]: res = res & states[p]
            states[n] = res
        elif gt == 'OR':
            res = states[preds[0]]
            for p in preds[1:]: res = res | states[p]
            states[n] = res
        elif gt == 'XOR':
            res = states[preds[0]]
            for p in preds[1:]: res = res ^ states[p]
            states[n] = res
        elif gt == 'NAND':
            res = states[preds[0]]
            for p in preds[1:]: res = res & states[p]
            states[n] = 1 - res
        elif gt == 'NOR':
            res = states[preds[0]]
            for p in preds[1:]: res = res | states[p]
            states[n] = 1 - res
        elif gt == 'XNOR':
            res = states[preds[0]]
            for p in preds[1:]: res = res ^ states[p]
            states[n] = 1 - res
        elif gt == 'MUX':
            if len(preds) >= 3:
                a, b, s = states[preds[0]], states[preds[1]], states[preds[2]]
                states[n] = np.where(s == 1, b, a)
            else:
                states[n] = states[preds[0]]
        else:
            res = states[preds[0]]
            for p in preds[1:]: res = res ^ states[p]
            states[n] = res
            
    # Calculate TP and SP (Signal Probability)
    node_stats = []
    for n in graph.nodes():
        if graph.nodes[n].get('type') == 'WIRE':
            # Skip primary outputs or inputs if possible
            if not any(sub in n for sub in ['PORT']):
                arr = states.get(n)
                if arr is not None:
                    tp = np.sum(arr[1:] != arr[:-1]) / max(1, num_vectors - 1)
                    sp = np.mean(arr)
                    bit_id = graph.nodes[n].get('bit_id')
                    if bit_id is not None and tp > 0.0: # Ignore completely dead logic
                        node_stats.append({'node': n, 'bit_id': bit_id, 'tp': tp, 'sp': sp})
                        
    # Sort by TP ascending (mimic RL rare node targeting)
    node_stats.sort(key=lambda x: x['tp'])
    
    # Select top K rare wires
    selected = node_stats[:K]
    print(f"Selected {K} rarest wires for trigger:")
    for s in selected:
        print(f"  - Bit {s['bit_id']}: TP={s['tp']:.6f}, SP={s['sp']:.6f}")
        
    return selected

def insert_trojan(json_path, output_path, K=4):
    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # Assume 1 main module
    top_module = list(data["modules"].keys())[0]
    mod = data["modules"][top_module]
    
    # 1. Find rare wires
    rare_wires = get_rare_wires(mod, K=K)
    
    # Find max bit ID to allocate new wires
    max_bit = 0
    for c_name, c_data in mod.get("cells", {}).items():
        for port, bits in c_data.get("connections", {}).items():
            for b in bits:
                if isinstance(b, int) and b > max_bit:
                    max_bit = b
                    
    # 2. Insert Trigger Logic
    current_trigger_bits = []
    
    cell_idx = 1
    # For each rare wire, condition it based on SP
    for w in rare_wires:
        bit_id = w['bit_id']
        sp = w['sp']
        if sp > 0.5:
            # Wire is usually 1. Trigger when 0 (insert NOT)
            max_bit += 1
            new_wire = max_bit
            not_cell = {
                "hide_name": 0,
                "type": "$_NOT_",
                "parameters": {},
                "attributes": {},
                "port_directions": {"A": "input", "Y": "output"},
                "connections": {"A": [bit_id], "Y": [new_wire]}
            }
            mod["cells"][f"TROJAN_NOT_{cell_idx}"] = not_cell
            current_trigger_bits.append(new_wire)
            cell_idx += 1
        else:
            # Wire is usually 0. Trigger when 1.
            current_trigger_bits.append(bit_id)
            
    # Combine trigger bits with AND gates
    while len(current_trigger_bits) > 1:
        next_trigger_bits = []
        for i in range(0, len(current_trigger_bits), 2):
            if i + 1 < len(current_trigger_bits):
                max_bit += 1
                new_wire = max_bit
                and_cell = {
                    "hide_name": 0,
                    "type": "$_AND_",
                    "parameters": {},
                    "attributes": {},
                    "port_directions": {"A": "input", "B": "input", "Y": "output"},
                    "connections": {"A": [current_trigger_bits[i]], "B": [current_trigger_bits[i+1]], "Y": [new_wire]}
                }
                mod["cells"][f"TROJAN_AND_{cell_idx}"] = and_cell
                next_trigger_bits.append(new_wire)
                cell_idx += 1
            else:
                next_trigger_bits.append(current_trigger_bits[i])
        current_trigger_bits = next_trigger_bits
        
    final_trigger_bit = current_trigger_bits[0]
    print(f"Trigger logic inserted. Final trigger wire ID: {final_trigger_bit}")
    
    # 3. Insert Payload Logic
    # Pick a random internal wire that is an input to some gate
    internal_inputs = []
    for c_name, c_data in mod.get("cells", {}).items():
        if "TROJAN" in c_name: continue
        for port, bits in c_data.get("connections", {}).items():
            if c_data.get("port_directions", {}).get(port) == "input":
                for b in bits:
                    if isinstance(b, int):
                        internal_inputs.append(b)
                        
    target_bit = random.choice(internal_inputs)
    print(f"Selected target bit {target_bit} for payload.")
    
    # Create payload XOR
    max_bit += 1
    payload_wire = max_bit
    xor_cell = {
        "hide_name": 0,
        "type": "$_XOR_",
        "parameters": {},
        "attributes": {},
        "port_directions": {"A": "input", "B": "input", "Y": "output"},
        "connections": {"A": [target_bit], "B": [final_trigger_bit], "Y": [payload_wire]}
    }
    mod["cells"]["TROJAN_PAYLOAD_XOR"] = xor_cell
    
    # Reroute target_bit to payload_wire in all original cells
    replaced_count = 0
    for c_name, c_data in mod.get("cells", {}).items():
        if c_name == "TROJAN_PAYLOAD_XOR": continue # don't replace in our payload gate
        for port, bits in c_data.get("connections", {}).items():
            if c_data.get("port_directions", {}).get(port) == "input":
                new_bits = []
                for b in bits:
                    if b == target_bit:
                        new_bits.append(payload_wire)
                        replaced_count += 1
                    else:
                        new_bits.append(b)
                c_data["connections"][port] = new_bits
                
    print(f"Payload inserted. Rerouted {replaced_count} connections to Trojan output.")
    
    # Save infected JSON
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"--- Trojan successfully inserted! Saved to {output_path} ---")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python rl_trojan_inserter.py <input_synth.json> <output_infected.json>")
        sys.exit(1)
        
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    insert_trojan(in_file, out_file, K=4)
