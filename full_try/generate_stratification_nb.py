import json

cells = []

def add_markdown(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split('\n')]
    })

def add_code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split('\n')]
    })

# ==================== MARKDOWN INIT ====================
add_markdown("# STRATIFICATION ANALYSIS\nQuesto notebook permette di scindere le coorti principali (es. `kras_pancreas`) in sub-coorti basate su Sesso, Età o Abitudine al Fumo, rigenerando le matrici omiche e confrontando le reti risultanti.")

# ==================== INIT CODE ====================
code_init = """# ==========================================
# IMPORT LIBRERIE GLOBALI E SETUP
# ==========================================
import os
import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# Parametri Generali
TARGET_GENE = 'KRAS'
COORTI = ["kras_pancreas", "kras_lung", "kras_colon"]
INPUT_BASE_DIR = "./data_filtered"
OUTPUT_STRAT_DIR = "./data_stratified"

os.makedirs(OUTPUT_STRAT_DIR, exist_ok=True)
print("✅ Ambiente configurato e cartella data_stratified creata.")"""
add_code(code_init)

# ==================== DATA FILTERING MARKDOWN ====================
add_markdown("## 1. DATA STRATIFICATION FILTERING\nQuesta funzione legge i file clinici originali, li divide per il criterio selezionato e affetta le enormi matrici di Mutazioni, CNA e SV per mantenere solo i pazienti rilevanti.")

# ==================== DATA FILTERING CODE ====================
code_filtering = """def slice_omics_data(cohort_path, new_dir, master_df, valid_ids):
    \"\"\"
    Affetta le matrici omiche (MUT, CNA, SV) basandosi sui valid_ids.
    \"\"\"
    files_to_slice = ['F_data_mutations.txt', 'F_data_cna.txt', 'F_data_sv.txt']
    
    for f in files_to_slice:
        source_file = os.path.join(cohort_path, f)
        # Controlla per varianti .csv o .txt
        if not os.path.exists(source_file):
            source_file = source_file.replace('.txt', '.csv')
            if not os.path.exists(source_file):
                continue
                
        target_df = pd.read_csv(source_file, sep='\\t', low_memory=False)
        
        # Identifica le colonne da tenere (Hugo_Symbol, Entrez_Gene_Id, e quelle nei valid_ids)
        cols_to_keep = []
        for col in target_df.columns:
            if col in ['Hugo_Symbol', 'Entrez_Gene_Id'] or col in valid_ids:
                cols_to_keep.append(col)
                
        sliced_df = target_df[cols_to_keep]
        # Salva la matrice stratificata
        out_ext = '.txt' if f.endswith('.txt') else '.csv'
        out_path = os.path.join(new_dir, f)
        sliced_df.to_csv(out_path, sep='\\t', index=False)


def stratify_cohort(cohort, criteria="Sex"):
    \"\"\"
    Divide una coorte in base al parametro criteria ('Sex', 'Age', 'Smoking').
    \"\"\"
    print(f"\\n{'='*60}\\nStratificazione {cohort.upper()} per {criteria}\\n{'='*60}")
    
    cohort_path = f"{INPUT_BASE_DIR}/{cohort}"
    cohort_base = cohort.replace("kras_", "")
    master_file = f"KRAS_F_{cohort_base}.csv"
    
    master_path = os.path.join(cohort_path, master_file)
    if not os.path.exists(master_path):
        print(f"[!] Master file non trovato: {master_path}")
        return
        
    df_master = pd.read_csv(master_path, sep='\\t')
    
    # Definiamo i gruppi in base al criterio
    groups = {}
    if criteria == "Sex":
        groups['Male'] = df_master[df_master['Sex'] == 'Male']
        groups['Female'] = df_master[df_master['Sex'] == 'Female']
    elif criteria == "Age":
        groups['Under65'] = df_master[pd.to_numeric(df_master['Current Age'], errors='coerce') < 65]
        groups['Over65'] = df_master[pd.to_numeric(df_master['Current Age'], errors='coerce') >= 65]
    elif criteria == "Smoking":
        smoker_keywords = ['Former/Current Smoker', 'Current Smoker', 'Former Smoker']
        groups['Smoker'] = df_master[df_master['Smoking History (NLP)'].isin(smoker_keywords)]
        groups['NonSmoker'] = df_master[df_master['Smoking History (NLP)'] == 'Never']
    else:
        print("Criterio non supportato.")
        return

    # Processiamo ogni gruppo
    for group_name, group_df in groups.items():
        if group_df.empty:
            print(f"  [-] Gruppo {group_name}: 0 pazienti. Salto.")
            continue
            
        print(f"  [+] Gruppo {group_name}: {len(group_df)} pazienti.")
        
        # Creiamo la cartella di output
        new_dir = os.path.join(OUTPUT_STRAT_DIR, f"{cohort}_{criteria}_{group_name}")
        os.makedirs(new_dir, exist_ok=True)
        
        # Salviamo il master file del sottogruppo
        new_master_path = os.path.join(new_dir, master_file)
        group_df.to_csv(new_master_path, sep='\\t', index=False)
        
        # Affettiamo le matrici omiche
        valid_ids = set(group_df['Sample_Id'].astype(str).unique())
        slice_omics_data(cohort_path, new_dir, group_df, valid_ids)
        
    print(f"✅ Stratificazione completata per {cohort}.")
"""
add_code(code_filtering)

add_code("""# ESEGUIAMO LA STRATIFICAZIONE PER I TRE TUMORI
# (De-commenta i criteri che vuoi testare)

criterio_scelto = "Sex" # Puoi usare "Sex", "Age", o "Smoking"

for c in COORTI:
    stratify_cohort(c, criteria=criterio_scelto)
""")

# ==================== CO-OCCURRENCE MARKDOWN ====================
add_markdown("## 2. CO-OCCURRENCE & NETWORK GENERATION\nQui calcoliamo la co-occorrenza unificata (Fisher's Exact Test) per le coorti stratificate e costruiamo i grafi MultiGraph.")

# ==================== CO-OCCURRENCE CODE ====================
code_cooc = """def extract_any_alteration_matrix(cohort_strat_dir):
    \"\"\"Crea la matrice binaria 'Any Alteration' (MUT or CNA or SV) per il sottogruppo.\"\"\"
    mut_file = os.path.join(cohort_strat_dir, 'F_data_mutations.txt')
    cna_file = os.path.join(cohort_strat_dir, 'F_data_cna.txt')
    if not os.path.exists(cna_file): cna_file = cna_file.replace('.txt', '.csv')
    sv_file = os.path.join(cohort_strat_dir, 'F_data_sv.txt')
    if not os.path.exists(sv_file): sv_file = sv_file.replace('.txt', '.csv')

    df_mut = pd.read_csv(mut_file, sep='\\t', index_col=0) if os.path.exists(mut_file) else pd.DataFrame()
    df_cna = pd.read_csv(cna_file, sep='\\t', index_col=0) if os.path.exists(cna_file) else pd.DataFrame()
    df_sv  = pd.read_csv(sv_file,  sep='\\t', index_col=0) if os.path.exists(sv_file) else pd.DataFrame()

    if 'Entrez_Gene_Id' in df_mut.columns: df_mut.drop(columns=['Entrez_Gene_Id'], inplace=True)
    if 'Entrez_Gene_Id' in df_cna.columns: df_cna.drop(columns=['Entrez_Gene_Id'], inplace=True)
    if 'Entrez_Gene_Id' in df_sv.columns:  df_sv.drop(columns=['Entrez_Gene_Id'],  inplace=True)

    bin_mut = (df_mut.notna() & (df_mut != 0) & (df_mut != '')).astype(int)
    bin_cna = ((df_cna == 2) | (df_cna == -2)).astype(int)
    bin_sv  = (df_sv.notna() & (df_sv != 0) & (df_sv != '')).astype(int)

    all_genes = set(bin_mut.index).union(set(bin_cna.index)).union(set(bin_sv.index))
    all_samples = set(bin_mut.columns).union(set(bin_cna.columns)).union(set(bin_sv.columns))

    df_any = pd.DataFrame(0, index=list(all_genes), columns=list(all_samples))
    
    # Operazione di OR logico
    df_any = df_any.add(bin_mut, fill_value=0)
    df_any = df_any.add(bin_cna, fill_value=0)
    df_any = df_any.add(bin_sv, fill_value=0)
    
    return (df_any >= 1).astype(int), bin_mut, bin_cna, bin_sv


def build_stratified_network(strat_dir_name, p_val_thresh=0.05, min_cooc=3):
    \"\"\"Calcola la co-occorrenza globale per un sottogruppo e restituisce un nx.MultiGraph.\"\"\"
    cohort_path = os.path.join(OUTPUT_STRAT_DIR, strat_dir_name)
    if not os.path.exists(cohort_path): return None
    
    df_any, df_mut, df_cna, df_sv = extract_any_alteration_matrix(cohort_path)
    
    # Filtro geni mutati in almeno min_cooc campioni
    freqs = df_any.sum(axis=1)
    valid_genes = freqs[freqs >= min_cooc].index.tolist()
    if TARGET_GENE not in valid_genes: valid_genes.append(TARGET_GENE)
    
    df_filtered = df_any.loc[valid_genes]
    num_samples = df_filtered.shape[1]
    
    edges = []
    # Calcolo Fisher per tutte le coppie possibili
    for g1, g2 in combinations(valid_genes, 2):
        row1, row2 = df_filtered.loc[g1], df_filtered.loc[g2]
        both = ((row1 == 1) & (row2 == 1)).sum()
        if both < min_cooc: continue
            
        g1_only = ((row1 == 1) & (row2 == 0)).sum()
        g2_only = ((row1 == 0) & (row2 == 1)).sum()
        neither = ((row1 == 0) & (row2 == 0)).sum()
        
        odds, p_val = fisher_exact([[both, g1_only], [g2_only, neither]], alternative='greater')
        if p_val <= p_val_thresh:
            edges.append((g1, g2, p_val, both))
            
    # Correzione p-value FDR
    if not edges: return None
    df_res = pd.DataFrame(edges, columns=['G1', 'G2', 'p_val', 'cooc'])
    df_res['p_adj'] = multipletests(df_res['p_val'], method='fdr_bh')[1]
    df_sig = df_res[df_res['p_adj'] <= p_val_thresh]
    
    # Creiamo il MultiGraph
    G = nx.MultiGraph()
    for _, row in df_sig.iterrows():
        g1, g2, w = row['G1'], row['G2'], row['cooc']
        
        # Aggiungo arco per ogni omica condivisa in ALMENO un campione
        v1_mut, v2_mut = df_mut.loc[g1] if g1 in df_mut.index else None, df_mut.loc[g2] if g2 in df_mut.index else None
        if v1_mut is not None and v2_mut is not None and ((v1_mut==1) & (v2_mut==1)).sum() > 0:
            G.add_edge(g1, g2, key='MUT', weight=w, color='red')
            
        v1_cna, v2_cna = df_cna.loc[g1] if g1 in df_cna.index else None, df_cna.loc[g2] if g2 in df_cna.index else None
        if v1_cna is not None and v2_cna is not None and ((v1_cna==1) & (v2_cna==1)).sum() > 0:
            G.add_edge(g1, g2, key='CNA', weight=w, color='blue')
            
        v1_sv, v2_sv = df_sv.loc[g1] if g1 in df_sv.index else None, df_sv.loc[g2] if g2 in df_sv.index else None
        if v1_sv is not None and v2_sv is not None and ((v1_sv==1) & (v2_sv==1)).sum() > 0:
            G.add_edge(g1, g2, key='SV', weight=w, color='green')
            
    return G
"""
add_code(code_cooc)

add_code("""# Generazione reti per un tumore d'esempio (Pancreas)
net_male = build_stratified_network(f"kras_pancreas_{criterio_scelto}_Male")
net_female = build_stratified_network(f"kras_pancreas_{criterio_scelto}_Female")

print(f"Rete Maschile: {net_male.number_of_nodes() if net_male else 0} Nodi, {net_male.number_of_edges() if net_male else 0} Archi")
print(f"Rete Femminile: {net_female.number_of_nodes() if net_female else 0} Nodi, {net_female.number_of_edges() if net_female else 0} Archi")
""")

# ==================== COMPARATIVE ANALYSIS MARKDOWN ====================
add_markdown("## 3. COMPARATIVE ANALYSIS (Topologia e Hubs)\nIn questa sezione confrontiamo le metriche della rete (Densità, Modularità) e valutiamo l'overlap dei Top Hub tra i due gruppi.")

code_compare = """def compute_hubs(G, top_n=10):
    if G is None or G.number_of_nodes() == 0: return set()
    # Usiamo Degree Centrality come proxy per l'importanza globale
    deg_cent = nx.degree_centrality(G)
    hubs = sorted(deg_cent.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return set([h[0] for h in hubs])

def compare_networks(G1, G2, name1="Group 1", name2="Group 2"):
    print(f"\\n{'='*60}\\n📊 COMPARISON: {name1} vs {name2}\\n{'='*60}")
    
    # Topologia Base
    d1 = nx.density(G1) if G1 else 0
    d2 = nx.density(G2) if G2 else 0
    print(f"Densità Rete: {name1} = {d1:.4f} | {name2} = {d2:.4f}")
    
    # Jaccard Index dei Top Hub
    hubs1 = compute_hubs(G1, top_n=20)
    hubs2 = compute_hubs(G2, top_n=20)
    
    if hubs1 and hubs2:
        intersection = hubs1.intersection(hubs2)
        union = hubs1.union(hubs2)
        jaccard = len(intersection) / len(union)
        
        print(f"\\n[!] Jaccard Index dell'Overlap dei Top 20 Hub: {jaccard:.2f}")
        print(f"🔸 Hub Comuni ({len(intersection)}): {', '.join(intersection)}")
        print(f"🔸 Hub Esclusivi {name1}: {', '.join(hubs1 - hubs2)}")
        print(f"🔸 Hub Esclusivi {name2}: {', '.join(hubs2 - hubs1)}")
    else:
        print("Non abbastanza dati per confrontare gli Hub.")

compare_networks(net_male, net_female, "Maschi", "Femmine")
"""
add_code(code_compare)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("stratification_analysis.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook 'stratification_analysis.ipynb' generato con successo!")
