# -*- coding: utf-8 -*-
import json

path = r'd:/multilingual hate speech _code mixed/hindiMix-noisy/notebooks/phase2/noisebridge_xlmr_kaggle.ipynb'
with open(path, encoding='utf-8') as f:
    nb = json.load(f)

# cp1252 mojibake: UTF-8 bytes read as cp1252
# E2 80 94 (em dash U+2014) -> â (U+00E2) + \x80->€(U+20AC) + \x94->"(U+201D)
# E2 86 92 (right arrow U+2192) -> â + \x86->†(U+2020) + \x92->'(U+2019)
replacements = [
    ('\u00e2\u20ac\u201d', '\u2014'),   # â€" -> em dash
    ('\u00e2\u2020\u2019', '\u2192'),   # â†' -> right arrow
]

for cell in nb['cells']:
    new_src = []
    for line in cell['source']:
        for bad, good in replacements:
            line = line.replace(bad, good)
        new_src.append(line)
    cell['source'] = new_src

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

# verify
with open(path, encoding='utf-8') as f:
    nb2 = json.load(f)
still_bad = []
for i, cell in enumerate(nb2['cells']):
    src = ''.join(cell['source'])
    if '\u00e2\u20ac' in src or '\u00e2\u2020' in src:
        still_bad.append(i)

if still_bad:
    open(r'd:/multilingual hate speech _code mixed/hindiMix-noisy/remaining.txt','w').write(f'Still bad: cells {still_bad}\n')
else:
    open(r'd:/multilingual hate speech _code mixed/hindiMix-noisy/remaining.txt','w').write('All clean!\n')
print('Done')
