import json, re

path = r'd:/multilingual hate speech _code mixed/hindiMix-noisy/notebooks/phase2/noisebridge_xlmr_kaggle.ipynb'
with open(path, encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    bad = [(j, repr(src[max(0,j-3):j+5])) for j, c in enumerate(src) if 127 < ord(c) < 256]
    if bad:
        print(f"Cell {i}:")
        for pos, ctx in bad[:10]:
            print(f"  pos {pos}: {ctx}")
