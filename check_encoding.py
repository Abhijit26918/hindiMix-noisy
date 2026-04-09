import json
path = r'd:/multilingual hate speech _code mixed/hindiMix-noisy/notebooks/phase2/noisebridge_xlmr_kaggle.ipynb'
with open(path, encoding='utf-8') as f:
    nb = json.load(f)
found = False
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    bad = [c for c in src if ord(c) > 127 and ord(c) < 256]
    if bad:
        print(f"Cell {i}: {set(bad)}")
        found = True
if not found:
    print("All clean!")
