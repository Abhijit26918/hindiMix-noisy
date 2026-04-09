import json

path = r'd:/multilingual hate speech _code mixed/hindiMix-noisy/notebooks/phase2/noisebridge_xlmr_kaggle.ipynb'
with open(path, encoding='utf-8') as f:
    nb = json.load(f)

bad = '\u00e2\u20ac\u201d'
good = '\u2014'

cell = nb['cells'][8]
count = 0
new_src = []
for line in cell['source']:
    if bad in line:
        count += 1
        line = line.replace(bad, good)
    new_src.append(line)
cell['source'] = new_src

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

# verify
with open(path, encoding='utf-8') as f:
    text = f.read()

remaining = '\u00e2\u20ac' in text
open(r'd:/multilingual hate speech _code mixed/hindiMix-noisy/remaining.txt','w',encoding='utf-8').write(
    f'Replaced {count} occurrences. Still bad: {remaining}\n'
)
print('Done')
