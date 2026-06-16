import glob
from pyverilog.vparser.parser import parse
from pyverilog.vparser.ast import ModuleDef, Ioport

files = glob.glob("AES_circuit/AES/*.v")
files = [f for f in files if not f.endswith("test_aes_128.v")]
ast, _ = parse(files)
for desc in ast.description.definitions:
    if isinstance(desc, ModuleDef) and desc.name == 'S':
        print("Module S ports:")
        for p in desc.portlist.ports:
            if type(p).__name__ == 'Ioport':
                print(f"Port {p.first.name}: {type(p.first).__name__}")
