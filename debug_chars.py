import json

path = r'd:/multilingual hate speech _code mixed/hindiMix-noisy/notebooks/phase2/noisebridge_xlmr_kaggle.ipynb'
with open(path, encoding='utf-8') as f:
    nb = json.load(f)

cell = nb['cells'][8]
src = ''.join(cell['source'])

out = open(r'd:/multilingual hate speech _code mixed/hindiMix-noisy/debug.txt', 'w', encoding='utf-8')
# find sequences with U+00E2
for j, c in enumerate(src):
    if c == '\u00e2':
        window = src[j:j+6]
        codepoints = [hex(ord(x)) for x in window]
        out.write(f"pos {j}: chars={repr(window)} codepoints={codepoints}\n")
out.close()
print('Done')
