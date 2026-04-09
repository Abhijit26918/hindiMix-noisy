import json
path = r'd:/multilingual hate speech _code mixed/hindiMix-noisy/notebooks/phase2/noisebridge_xlmr_kaggle.ipynb'
with open(path, encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    for j, c in enumerate(src):
        if ord(c) > 127 and ord(c) < 256:
            print(f"Cell {i}, pos {j}: {repr(src[max(0,j-5):j+10])}")
