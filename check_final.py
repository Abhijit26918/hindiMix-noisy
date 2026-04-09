import json

path = r'd:/multilingual hate speech _code mixed/hindiMix-noisy/notebooks/phase2/noisebridge_xlmr_kaggle.ipynb'
with open(path, encoding='utf-8') as f:
    nb = json.load(f)

mojibake_starts = {'\u00e2', '\u00c3', '\u00ce', '\ufffd'}
found = False
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    bad = [src[j:j+4] for j in range(len(src)) if src[j] in mojibake_starts]
    if bad:
        found = True
        out = open(r'd:/multilingual hate speech _code mixed/hindiMix-noisy/remaining.txt', 'w', encoding='utf-8')
        out.write(f"Cell {i}: {bad[:10]}\n")
        out.close()

if not found:
    open(r'd:/multilingual hate speech _code mixed/hindiMix-noisy/remaining.txt', 'w').write('All clean!\n')
print('Done')
