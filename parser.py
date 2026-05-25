import networkx as nx
import matplotlib.pyplot as plt
from pyverilog.vparser.parser import parse  # pyrefly: ignore [missing-import]

class NetlistGraphBuilder:
    def __init__(self, filepath):
        self.filepath = filepath
        self.graph = nx.DiGraph()
        self.net_drivers = {} # net_name -> driving_node
        self.net_receivers = {} # net_name -> list of receiving_nodes
        
    def build(self):
        ast, _ = parse([self.filepath])
        
        for desc in ast.description.definitions:
            if type(desc).__name__ == 'ModuleDef':
                self.process_module(desc)
        
        self.construct_edges()
        self.extract_features()
        return self.graph

    def process_module(self, module):
        if hasattr(module, 'portlist') and module.portlist:
            for port in module.portlist.ports:
                port_type = type(port).__name__
                if port_type == 'Ioport':
                    decl = port.first
                    decl_type = type(decl).__name__
                    if decl_type == 'Input':
                        self.graph.add_node(decl.name, type='INPUT')
                        self.net_drivers[decl.name] = decl.name
                    elif decl_type == 'Output':
                        self.graph.add_node(decl.name, type='OUTPUT')
                        if decl.name not in self.net_receivers:
                            self.net_receivers[decl.name] = []
                        self.net_receivers[decl.name].append(decl.name)

        for item in module.items:
            item_type = type(item).__name__
            if item_type == 'Decl':
                for decl in item.list:
                    decl_type = type(decl).__name__
                    if decl_type == 'Input':
                        self.graph.add_node(decl.name, type='INPUT')
                        self.net_drivers[decl.name] = decl.name
                    elif decl_type == 'Output':
                        self.graph.add_node(decl.name, type='OUTPUT')
                        if decl.name not in self.net_receivers:
                            self.net_receivers[decl.name] = []
                        self.net_receivers[decl.name].append(decl.name)
            elif item_type == 'InstanceList':
                for instance in item.instances:
                    self.process_instance(instance)
                    
    def process_instance(self, instance):
        gate_type = instance.module.upper()
        node_name = instance.name
        if not node_name:
            node_name = f"{gate_type}_{id(instance)}"
        
        self.graph.add_node(node_name, type=gate_type)
        
        is_primitive = gate_type.lower() in ['and', 'or', 'nand', 'nor', 'xor', 'xnor', 'not', 'buf']
        
        if is_primitive:
            # For primitives, the first port is the output, subsequent are inputs
            output_wire = instance.portlist[0].argname.name
            self.net_drivers[output_wire] = node_name
            
            for port in instance.portlist[1:]:
                input_wire = port.argname.name
                if input_wire not in self.net_receivers:
                    self.net_receivers[input_wire] = []
                self.net_receivers[input_wire].append(node_name)
        else:
            # For module instantiations, look at port names
            for port in instance.portlist:
                port_name = port.portname
                wire_name = port.argname.name if hasattr(port.argname, 'name') else None
                if not wire_name:
                    continue
                
                # Heuristic for determining if port is output (Y, Q, OUT)
                if port_name.upper() in ['Y', 'Q', 'OUT']:
                    self.net_drivers[wire_name] = node_name
                else:
                    if wire_name not in self.net_receivers:
                        self.net_receivers[wire_name] = []
                    self.net_receivers[wire_name].append(node_name)
                    
    def construct_edges(self):
        for net, receivers in self.net_receivers.items():
            if net in self.net_drivers:
                driver = self.net_drivers[net]
                for receiver in receivers:
                    self.graph.add_edge(driver, receiver, wire=net)

    def extract_features(self):
        gate_types = set([nx.get_node_attributes(self.graph, 'type').get(n, 'UNKNOWN') for n in self.graph.nodes()])
        gate_types = sorted(list(gate_types))
        type_to_idx = {t: i for i, t in enumerate(gate_types)}
        
        in_degrees = dict(self.graph.in_degree())
        out_degrees = dict(self.graph.out_degree())
        deg_centrality = nx.degree_centrality(self.graph)
        
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get('type', 'UNKNOWN')
            
            one_hot = [0.0] * len(gate_types)
            if node_type in type_to_idx:
                one_hot[type_to_idx[node_type]] = 1.0
            
            struct_feat = [
                float(in_degrees[node]),
                float(out_degrees[node]),
                float(deg_centrality[node])
            ]
            
            self.graph.nodes[node]['x'] = one_hot + struct_feat
            
        print(f"Extracted features for {self.graph.number_of_nodes()} nodes. Feature vector length: {len(gate_types) + 3}")
        return self.graph

def visualize_graph(graph):
    plt.figure(figsize=(10, 8))
    
    # Create layout
    pos = nx.spring_layout(graph, seed=42)
    
    # Define colors based on node type
    color_map = []
    labels = {}
    for node in graph.nodes():
        node_type = graph.nodes[node].get('type', 'UNKNOWN')
        labels[node] = f"{node}\n({node_type})"
        if node_type == 'INPUT':
            color_map.append('lightgreen')
        elif node_type == 'OUTPUT':
            color_map.append('salmon')
        else:
            color_map.append('skyblue')
            
    # Draw graph components
    nx.draw_networkx_nodes(graph, pos, node_color=color_map, node_size=1500, edgecolors='black')
    nx.draw_networkx_edges(graph, pos, arrowstyle='->', arrowsize=20, edge_color='gray')
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10, font_weight='bold')
    
    # Draw edge labels (wire names)
    edge_labels = nx.get_edge_attributes(graph, 'wire')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red')
    
    plt.title("Netlist DAG Visualization")
    plt.axis('off')
    plt.tight_layout()
    
    # Save and show
    plt.savefig('graph.png', dpi=300, bbox_inches='tight')
    print("Graph visualization saved to graph.png")
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        builder = NetlistGraphBuilder(sys.argv[1])
        g = builder.build()
        print(f"Graph constructed with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges.")
        
        # Visualize the graph
        visualize_graph(g)
