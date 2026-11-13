import networkx as nx
import matplotlib.pyplot as plt
from pyverilog.vparser.parser import parse  # pyrefly: ignore [missing-import]
from pyverilog.vparser.ast import InstanceList, Assign, Always, NonblockingSubstitution, BlockingSubstitution, Identifier, Pointer, Partselect, ModuleDef  # pyrefly: ignore [missing-import]

class NetlistGraphBuilder:
    def __init__(self, filepaths, top_module="uart"):
        self.filepaths = filepaths if isinstance(filepaths, list) else [filepaths]
        self.top_module = top_module
        self.graph = nx.DiGraph()
        self.modules = {}

    def build(self):
        ast, _ = parse(self.filepaths)
        for desc in ast.description.definitions:
            if isinstance(desc, ModuleDef):
                self.modules[desc.name] = desc
                
        if self.top_module not in self.modules:
            # Fallback to the first module if top_module isn't found
            self.top_module = list(self.modules.keys())[0]
            
        self._flatten_module(self.top_module, instance_prefix="")
        self.extract_features()
        return self.graph

    def _get_identifiers(self, node):
        """Recursively extract all Identifier names from an AST node (RHS or LHS)."""
        ids = []
        if isinstance(node, Identifier):
            ids.append(node.name)
        elif isinstance(node, (Pointer, Partselect)):
            if isinstance(node.var, Identifier):
                ids.append(node.var.name)
        elif hasattr(node, 'children'):
            for child in node.children():
                ids.extend(self._get_identifiers(child))
        return ids

    def _flatten_module(self, module_name, instance_prefix=""):
        module = self.modules[module_name]
        
        # 1. Add Ports
        if hasattr(module, 'portlist') and module.portlist:
            for port in module.portlist.ports:
                if type(port).__name__ == 'Ioport':
                    decl = port.first
                    node_name = f"{instance_prefix}{decl.name}"
                    if type(decl).__name__ == 'Input':
                        self.graph.add_node(node_name, type='INPUT')
                    elif type(decl).__name__ == 'Output':
                        self.graph.add_node(node_name, type='OUTPUT')

        # 2. Parse Items
        for item in module.items:
            item_type = type(item).__name__
            
            # Extract Input/Output Declarations (Verilog-1995 style)
            if item_type == 'Decl':
                for decl in item.list:
                    decl_type = type(decl).__name__
                    if decl_type == 'Input':
                        node_name = f"{instance_prefix}{decl.name}"
                        self.graph.add_node(node_name, type='INPUT')
                    elif decl_type == 'Output':
                        node_name = f"{instance_prefix}{decl.name}"
                        self.graph.add_node(node_name, type='OUTPUT')
            
            # Structural Instantiations (Gates or Sub-modules)
            if item_type == 'InstanceList':
                mod_name = item.module
                is_gate = mod_name.lower() in ['and', 'nand', 'or', 'nor', 'xor', 'xnor', 'not']
                
                for instance in item.instances:
                    inst_name = instance.name
                    new_prefix = f"{instance_prefix}{inst_name}." if instance_prefix else f"{inst_name}."
                    
                    if is_gate:
                        gate_node = f"{instance_prefix}{inst_name}"
                        self.graph.add_node(gate_node, type=mod_name.upper())
                        
                        if len(instance.portlist) > 0:
                            out_port = instance.portlist[0].argname
                            out_wire = f"{instance_prefix}{out_port.name}" if hasattr(out_port, 'name') else f"{instance_prefix}{out_port}"
                            self.graph.add_edge(gate_node, out_wire, wire=out_wire)
                            
                            for in_port in instance.portlist[1:]:
                                in_wire = f"{instance_prefix}{in_port.argname.name}" if hasattr(in_port.argname, 'name') else f"{instance_prefix}{in_port.argname}"
                                self.graph.add_edge(in_wire, gate_node, wire=in_wire)
                    
                    elif mod_name in self.modules:
                        # Hierarchical Sub-module
                        self._flatten_module(mod_name, new_prefix)
                        
                        # Map Ports
                        sub_mod = self.modules[mod_name]
                        port_dirs = {}
                        formal_ports = []
                        if hasattr(sub_mod, 'portlist') and sub_mod.portlist:
                            for p in sub_mod.portlist.ports:
                                if type(p).__name__ == 'Ioport':
                                    port_dirs[p.first.name] = type(p.first).__name__
                                    formal_ports.append(p.first.name)
                                elif type(p).__name__ == 'Port':
                                    formal_ports.append(p.name)
                                    
                        # Extract directions for Non-ANSI style
                        for item in sub_mod.items:
                            if type(item).__name__ == 'Decl':
                                for decl in item.list:
                                    decl_type = type(decl).__name__
                                    if decl_type in ['Input', 'Output', 'Inout']:
                                        port_dirs[decl.name] = decl_type
                                    
                        for port_idx, portarg in enumerate(instance.portlist):
                            port_name = portarg.portname
                            if not port_name and port_idx < len(formal_ports):
                                port_name = formal_ports[port_idx]
                                
                            if port_name:
                                sub_node = f"{new_prefix}{port_name}"
                                arg_ids = self._get_identifiers(portarg.argname)
                                
                                p_dir = port_dirs.get(port_name, 'unknown')
                                for arg_id in arg_ids:
                                    wire_node = f"{instance_prefix}{arg_id}"
                                    if p_dir == 'Input':
                                        self.graph.add_edge(wire_node, sub_node, wire=wire_node)
                                    elif p_dir == 'Output':
                                        self.graph.add_edge(sub_node, wire_node, wire=wire_node)
                                    else:
                                        self.graph.add_edge(wire_node, sub_node, wire=wire_node)
                                        self.graph.add_edge(sub_node, wire_node, wire=wire_node)
                    
                    else:
                        # Unknown module or standard cell
                        gate_node = f"{instance_prefix}{inst_name}"
                        self.graph.add_node(gate_node, type=mod_name.upper())
                        
                        for portarg in instance.portlist:
                            port_name = portarg.portname
                            if port_name:
                                arg_ids = self._get_identifiers(portarg.argname)
                                p_dir = 'unknown'
                                if port_name in ['Y', 'Q', 'QN', 'Z', 'OUT', 'DOUT']:
                                    p_dir = 'Output'
                                elif port_name in ['A', 'B', 'C', 'D', 'E', 'F', 'S', 'S0', 'S1', 'CK', 'CLK', 'D0', 'D1', 'A0', 'A1', 'B0', 'B1', 'C0', 'C1', 'SI', 'SE', 'SN', 'RN']:
                                    p_dir = 'Input'
                                
                                for arg_id in arg_ids:
                                    wire_node = f"{instance_prefix}{arg_id}"
                                    if p_dir == 'Input':
                                        self.graph.add_edge(wire_node, gate_node, wire=wire_node)
                                    elif p_dir == 'Output':
                                        self.graph.add_edge(gate_node, wire_node, wire=wire_node)
                                    else:
                                        self.graph.add_edge(wire_node, gate_node, wire=wire_node)
                                        self.graph.add_edge(gate_node, wire_node, wire=wire_node)

            # Behavioral Assignments (Assign)
            elif item_type == 'Assign':
                lhs_ids = self._get_identifiers(item.left.var)
                rhs_ids = self._get_identifiers(item.right.var)
                for lhs in lhs_ids:
                    lhs_node = f"{instance_prefix}{lhs}"
                    self.graph.add_node(lhs_node, type='BEHAVIORAL')
                    for rhs in rhs_ids:
                        rhs_node = f"{instance_prefix}{rhs}"
                        self.graph.add_edge(rhs_node, lhs_node, wire=rhs_node)

            # Behavioral Always Blocks
            elif item_type == 'Always':
                # Extract sensitivity list variables (like sys_clk, sys_rst_l)
                sens_ids = []
                if hasattr(item, 'sens_list'):
                    for sens in item.sens_list.list:
                        sens_ids.extend(self._get_identifiers(sens.sig))
                        
                def extract_assignments(node, current_conds=[]):
                    assignments = []
                    node_type = type(node).__name__
                    
                    # Track control-flow conditions
                    new_conds = list(current_conds)
                    if node_type == 'IfStatement':
                        new_conds.extend(self._get_identifiers(node.cond))
                    elif node_type == 'CaseStatement':
                        new_conds.extend(self._get_identifiers(node.comp))
                        
                    if node_type in ['NonblockingSubstitution', 'BlockingSubstitution']:
                        assignments.append((node, new_conds))
                    elif hasattr(node, 'children'):
                        for child in node.children():
                            assignments.extend(extract_assignments(child, new_conds))
                    return assignments
                
                assigns = extract_assignments(item.statement)
                for assign, cond_ids in assigns:
                    lhs_ids = self._get_identifiers(assign.left.var)
                    # The right hand side includes the actual RHS *plus* conditions *plus* sensitivity list
                    rhs_ids = self._get_identifiers(assign.right.var) + cond_ids + sens_ids
                    
                    for lhs in lhs_ids:
                        lhs_node = f"{instance_prefix}{lhs}"
                        self.graph.add_node(lhs_node, type='BEHAVIORAL')
                        for rhs in rhs_ids:
                            rhs_node = f"{instance_prefix}{rhs}"
                            self.graph.add_edge(rhs_node, lhs_node, wire=rhs_node)

    def extract_features(self):
        gate_types = ['INPUT', 'OUTPUT', 'AND', 'NAND', 'OR', 'NOR', 'XOR', 'XNOR', 'NOT', 'DFF', 'BUF', 'BEHAVIORAL', 'UNKNOWN']
        for node in self.graph.nodes():
            n_type = self.graph.nodes[node].get('type', 'UNKNOWN').upper()
            
            normalized_type = 'UNKNOWN'
            if 'NND' in n_type or 'NAND' in n_type: normalized_type = 'NAND'
            elif 'AND' in n_type: normalized_type = 'AND'
            elif 'NOR' in n_type or 'NR' in n_type: normalized_type = 'NOR'
            elif 'OR' in n_type: normalized_type = 'OR'
            elif 'XNOR' in n_type: normalized_type = 'XNOR'
            elif 'XOR' in n_type: normalized_type = 'XOR'
            elif 'INV' in n_type or 'NOT' in n_type: normalized_type = 'NOT'
            elif 'DFF' in n_type or 'SDFF' in n_type: normalized_type = 'DFF'
            elif 'BUF' in n_type: normalized_type = 'BUF'
            elif n_type in ['INPUT', 'OUTPUT', 'BEHAVIORAL']: normalized_type = n_type
            
            one_hot = [1.0 if normalized_type == gt else 0.0 for gt in gate_types]
            in_deg = float(self.graph.in_degree(node))
            out_deg = float(self.graph.out_degree(node))
            feat = one_hot + [in_deg, out_deg]
            self.graph.nodes[node]['x'] = feat

def visualize_graph(graph):
    print("Graph visualization skipped in batch mode to avoid UI hangs.")
