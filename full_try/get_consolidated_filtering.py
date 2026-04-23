import json
import sys

with open('data_filtering.ipynb', 'r') as f:
    nb = json.load(f)

all_imports = set()

# Extract imports
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if line.startswith('import ') or line.startswith('from '):
                all_imports.add(line.strip() + '\n')
            else:
                new_source.append(line)
        
        while new_source and new_source[0].strip() == '':
            new_source.pop(0)
        
        cell['source'] = new_source

# Inject
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        sorted_imports = sorted(list(all_imports))
        header = [
            "# ==========================================\n",
            "# IMPORT LIBRERIE GLOBALI\n",
            "# ==========================================\n"
        ]
        cell['source'] = header + sorted_imports + ['\n'] + cell['source']
        break

print(json.dumps(nb, indent=1))
