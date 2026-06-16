import glob
from pyverilog.vparser.preprocessor import preprocess

files = glob.glob("AES_circuit/AES T100/*.v")
files = [f for f in files if not f.endswith("test_aes_128.v")]
text = preprocess(files)
with open("debug.v", "w") as f:
    f.write(text)
