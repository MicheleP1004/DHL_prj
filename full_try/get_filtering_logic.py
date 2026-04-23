import json

with open('data_filtering.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'msk_chord_2024_clinical_data.csv' in source or 'data_filtered' in source:
            print("--- CELL START ---")
            lines = source.split('\n')
            for line in lines[:20]:
                print(line)
            print("...")
