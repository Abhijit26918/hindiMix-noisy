import json, sys

path = r'd:/multilingual hate speech _code mixed/hindiMix-noisy/notebooks/phase2/noisebridge_xlmr_kaggle.ipynb'
with open(path, encoding='utf-8') as f:
    nb = json.load(f)

out = open(r'd:/multilingual hate speech _code mixed/hindiMix-noisy/bad_chars.txt', 'w', encoding='utf-8')
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    bad = [(j, repr(src[max(0,j-3):j+5])) for j, c in enumerate(src) if ord(c) > 127]
    if bad:
        out.write(f"Cell {i}:\n")
        for pos, ctx in bad[:15]:
            out.write(f"  pos {pos}: {ctx}\n")
out.close()
print("Done")
