import json

with open('data_filtering.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'msk_chord_2024_clinical_data.csv' in source or 'Cancer Type' in source:
            print(source[:500])
