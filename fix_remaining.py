import json

path = r'd:/multilingual hate speech _code mixed/hindiMix-noisy/notebooks/phase2/noisebridge_xlmr_kaggle.ipynb'
with open(path, encoding='utf-8') as f:
    nb = json.load(f)

# Only fix true mojibake sequences, leave proper Unicode alone
mojibake = {
    '\u00e2\u0080\u0094': '\u2014',  # â€" -> em dash
    '\u00e2\u0086\u0092': '\u2192',  # â†' -> right arrow
    '\ufffd': '',                     # remove replacement chars
}

for cell in nb['cells']:
    new_src = []
    for line in cell['source']:
        for bad, good in mojibake.items():
            line = line.replace(bad, good)
        new_src.append(line)
    cell['source'] = new_src

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Done')
