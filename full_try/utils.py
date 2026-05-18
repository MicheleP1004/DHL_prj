# ==========================================
# utils.py — Pipeline di analisi mutazioni (SNV)
# ==========================================
# Utilizzo dal notebook:
#
#   from utils import (generate_matrices, calculate_statistics,
#                      plot_volcanos, build_all_networks,
#                      calculate_metrics, compare_networks)
#
#   PATH        = "./kras_pancreas"   # cartella dati di input E output
#   TARGET_GENE = "KRAS"
#
#   COOCC_PARAMS = {"p_val": 0.05, "log2or": 1.0, "min_cooc": 3}
#   ME_PARAMS    = {"p_val": 0.01, "log2or": -1.0}
#
#   generate_matrices(PATH, TARGET_GENE)
#   calculate_statistics(PATH, TARGET_GENE, COOCC_PARAMS, ME_PARAMS)
#   plot_volcanos(PATH, TARGET_GENE, COOCC_PARAMS, ME_PARAMS)
#   build_all_networks(PATH, TARGET_GENE, COOCC_PARAMS)
#   calculate_metrics(PATH, TARGET_GENE, COOCC_PARAMS)
#
#   # Confronto fra due coorti (opzionale)
#   compare_networks("./pancreas", "./kras_pancreas", COOCC_PARAMS, COOCC_PARAMS)
# ==========================================

import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import networkx.algorithms.community as nx_comm
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# Helpers interni
# ---------------------------------------------------------------------------

def _cohort_name(path: str) -> str:
    """Restituisce il nome della coorte dal basename del path."""
    return os.path.basename(os.path.normpath(path))


def _out(path: str, subdir: str) -> str:
    """Crea e restituisce la sottocartella di output dentro `path`."""
    d = os.path.join(path, subdir)
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# 1. GENERAZIONE MATRICI BINARIE
# ---------------------------------------------------------------------------

def generate_matrices(path: str, target_gene: str = "KRAS") -> None:
    """
    Legge F_data_mutations.txt da `path`, costruisce la matrice binaria
    e la matrice di co-occorrenza, e le salva in `path/matrices/`.

    Per la coorte 'colon' applica automaticamente il filtro MSI se
    il file F_colon.csv è presente.

    Parameters
    ----------
    path : str
        Cartella contenente i dati di input (es. "./kras_pancreas").
        Tutti i file di output vengono salvati nelle sottocartelle di `path`.
    target_gene : str
        Gene target (default "KRAS"). Utilizzato solo per l'intestazione.
    """
    cohort_name = _cohort_name(path)
    print(f"\n{'='*60}")
    print(f"🧬 PREPARAZIONE DATI MUTAZIONI (SNV): {cohort_name.upper()}")
    print("="*60)

    mut_file = os.path.join(path, "F_data_mutations.txt")
    if not os.path.exists(mut_file):
        print(f"[!] File {mut_file} non trovato. Salto.")
        return

    df_mut = pd.read_csv(mut_file, sep="\t")

    functional = [
        "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
        "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins", "Splice_Site",
    ]

    # --- Statistiche "prima" (baseline) ---
    df_mut_func_all = df_mut[df_mut["Variant_Classification"].isin(functional)]
    pazienti_totali_iniziali = df_mut["Sample_Id"].nunique()
    muts_per_patient_before = df_mut_func_all.groupby("Sample_Id").size()
    tutti_i_pazienti_iniziali = pd.Series(0, index=df_mut["Sample_Id"].unique())
    muts_per_patient_before = muts_per_patient_before.combine_first(tutti_i_pazienti_iniziali)

    # --- Filtro MSI per colon ---
    if cohort_name == "colon":
        clin_file = os.path.join(path, "F_colon.csv")
        if os.path.exists(clin_file):
            try:
                df_clin = pd.read_csv(clin_file, sep="\t")
                if "MSI Type" not in df_clin.columns:
                    df_clin = pd.read_csv(clin_file, sep=",")
            except Exception:
                df_clin = pd.read_csv(clin_file, sep=",")

            if "MSI Type" in df_clin.columns and "Sample_Id" in df_clin.columns:
                stable_samples = df_clin[df_clin["MSI Type"] == "Stable"]["Sample_Id"].tolist()
                df_mut_filtered = df_mut[df_mut["Sample_Id"].isin(stable_samples)]
                pazienti_rimasti = df_mut_filtered["Sample_Id"].nunique()

                df_mut_func_filtered = df_mut_filtered[
                    df_mut_filtered["Variant_Classification"].isin(functional)
                ]
                muts_per_patient_after = df_mut_func_filtered.groupby("Sample_Id").size()
                pazienti_validi_series = pd.Series(0, index=df_mut_filtered["Sample_Id"].unique())
                muts_per_patient_after = muts_per_patient_after.combine_first(pazienti_validi_series)

                print("\n⚖️ --- IMPATTO DEL FILTRO MSI (MSS vs ALL) ---")
                print(f"  • Pazienti:        {pazienti_totali_iniziali} -> {pazienti_rimasti} "
                      f"(Rimossi {pazienti_totali_iniziali - pazienti_rimasti} pazienti ipermutati)")
                print(f"  • Tot. Mutazioni:  {int(muts_per_patient_before.sum())} -> "
                      f"{int(muts_per_patient_after.sum())}")
                print(f"  • Media Mut/Paz:   {muts_per_patient_before.mean():.1f} -> "
                      f"{muts_per_patient_after.mean():.1f}")
                print(f"  • Mediana Mut/Paz: {muts_per_patient_before.median():.1f} -> "
                      f"{muts_per_patient_after.median():.1f}")
                print(f"  • MAX Mut/Paz:     {int(muts_per_patient_before.max())} -> "
                      f"{int(muts_per_patient_after.max())}  <-- Il dato che sbroglia il gomitolo!")
                print("--------------------------------------------------")
                df_mut = df_mut_filtered
            else:
                print("[!] Colonne MSI Type o Sample_Id non trovate nel file clinico.")
        else:
            print(f"[!] File clinico {clin_file} non trovato per il colon.")

    # --- Matrice binaria finale ---
    df_mut_func_final = df_mut[df_mut["Variant_Classification"].isin(functional)]
    events = df_mut_func_final[["Sample_Id", "Hugo_Symbol"]].drop_duplicates()
    pat_gene_counts = pd.crosstab(events["Sample_Id"], events["Hugo_Symbol"])
    binary_matrix = (pat_gene_counts > 0).astype(int)

    tutti_i_pazienti_finali = df_mut["Sample_Id"].unique()
    binary_matrix = binary_matrix.reindex(tutti_i_pazienti_finali, fill_value=0)

    num_samples, num_genes = binary_matrix.shape
    events_per_gene = binary_matrix.sum(axis=0)

    print("\n📊 --- STATISTICHE RETE FINALE (SNV) ---")
    print(f"  • Dimensioni: {num_samples} Campioni x {num_genes} Geni unici")
    print(f"  • Densità matrice: "
          f"{(binary_matrix.sum().sum() / (num_samples * num_genes)) * 100:.2f}%")

    top_genes = events_per_gene.sort_values(ascending=False).head(5)
    print("\n  🔥 Top 5 Geni Driver (SNV):")
    for g, count in top_genes.items():
        print(f"    - {g}: {count} pazienti ({(count/num_samples)*100:.1f}%)")
    print("====================================================\n")

    co_occ_matrix = binary_matrix.T.dot(binary_matrix)

    out_dir = _out(path, "matrices")
    binary_matrix.to_csv(os.path.join(out_dir, f"M_binary_{cohort_name}.tsv"), sep="\t")
    co_occ_matrix.to_csv(os.path.join(out_dir, f"M_cooccurrence_{cohort_name}.tsv"), sep="\t")
    print(f"✅ Matrici SNV salvate in: {out_dir}")


# ---------------------------------------------------------------------------
# 2. ANALISI STATISTICA (FISHER + FDR)
# ---------------------------------------------------------------------------

def calculate_statistics(
    path: str,
    target_gene: str = "KRAS",
    coocc_params: dict | None = None,
    me_params: dict | None = None,
) -> None:
    """
    Calcola co-occorrenze (tutti vs tutti) e mutua esclusività
    (target_gene vs tutti) sulla matrice binaria già salvata da
    generate_matrices(). Salva i risultati in `path/stats/`.

    Parameters
    ----------
    path : str
        Stessa cartella passata a generate_matrices().
    target_gene : str
        Gene target per la mutua esclusività (default "KRAS").
    coocc_params : dict, optional
        Parametri co-occorrenza. Usato solo per riferimento; i filtri
        vengono applicati nelle fasi successive.
    me_params : dict, optional
        Parametri mutua esclusività (idem).
    """
    cohort_name = _cohort_name(path)
    print(f"\n--- 🧮 CALCOLO STATISTICHE: {cohort_name.upper()} ---")

    bin_file = os.path.join(path, "matrices", f"M_binary_{cohort_name}.tsv")
    if not os.path.exists(bin_file):
        print(f"[!] File binario non trovato: {bin_file}. Eseguire prima generate_matrices().")
        return

    df_bin = pd.read_csv(bin_file, sep="\t", index_col=0).fillna(0)
    genes = df_bin.columns.tolist()
    n_total = len(df_bin)
    gene_counts = df_bin.sum(axis=0).astype(int)

    out_dir = _out(path, "stats")

    # --- 1. Co-occorrenza full (tutti vs tutti) ---
    full_results = []
    print("[*] Calcolo Co-occorrenza globale...")
    for g1, g2 in combinations(genes, 2):
        a = ((df_bin[g1] == 1) & (df_bin[g2] == 1)).sum()
        if a < 2:
            continue
        b = gene_counts[g1] - a
        c = gene_counts[g2] - a
        d = n_total - (a + b + c)

        odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
        log2or = np.log2(odds_ratio) if odds_ratio > 0 else -10.0

        full_results.append({
            "Gene_A": g1, "Gene_B": g2, "Co_Occurrence_Count": a,
            "P_Value": p_value, "Log2OR": log2or,
        })

    if full_results:
        df_full = pd.DataFrame(full_results)
        df_full["P_Adj"] = multipletests(df_full["P_Value"], method="fdr_bh")[1]
        df_full["Log2OR"] = df_full["Log2OR"].replace(
            [np.inf, -np.inf], [10.0, -10.0]
        )
        df_full.to_csv(
            os.path.join(out_dir, f"Full_Cooccurrence_Stats_{cohort_name}.tsv"),
            sep="\t", index=False,
        )
        print("✅ Statistiche di Co-occorrenza salvate.")

    # --- 2. Mutua esclusività (target vs tutti) ---
    if target_gene not in genes:
        print(f"[!] {target_gene} assente. Salto Mutua Esclusività.")
        return

    me_results = []
    print(f"[*] Calcolo Mutua Esclusività per {target_gene}...")
    target_mut = df_bin[target_gene]

    for gene in genes:
        if gene == target_gene:
            continue
        gene_mut = df_bin[gene]

        both_mut    = ((target_mut == 1) & (gene_mut == 1)).sum()
        only_target = ((target_mut == 1) & (gene_mut == 0)).sum()
        only_gene   = ((target_mut == 0) & (gene_mut == 1)).sum()
        neither_mut = ((target_mut == 0) & (gene_mut == 0)).sum()

        oddsratio, p_value = fisher_exact(
            [[both_mut, only_target], [only_gene, neither_mut]], alternative="less"
        )

        if oddsratio == 0:
            log2or = -10.0
        elif oddsratio == np.inf:
            log2or = 10.0
        else:
            log2or = np.log2(oddsratio)

        me_results.append({
            "Gene_A": target_gene, "Gene_B": gene,
            "Co_Occurrence_Count": both_mut,
            "Only_KRAS": only_target, "Only_Gene_B": only_gene, "Neither": neither_mut,
            "P_Value": p_value, "Log2OR": log2or,
        })

    if me_results:
        df_me = pd.DataFrame(me_results)
        df_me["P_Adj"] = multipletests(df_me["P_Value"], method="fdr_bh")[1]
        df_me = df_me.sort_values("P_Value")
        df_me.to_csv(
            os.path.join(out_dir, f"Mutual_Exclusivity_Stats_{cohort_name}.tsv"),
            sep="\t", index=False,
        )
        print("✅ Statistiche di Mutua Esclusività salvate.")


# ---------------------------------------------------------------------------
# 3. VOLCANO PLOTS
# ---------------------------------------------------------------------------

def plot_volcanos(
    path: str,
    target_gene: str = "KRAS",
    coocc_params: dict | None = None,
    me_params: dict | None = None,
) -> None:
    """
    Genera i volcano plot (co-occorrenza e mutua esclusività) e li salva
    in `path/plots/`.

    Parameters
    ----------
    path : str
        Stessa cartella passata alle funzioni precedenti.
    target_gene : str
        Gene target da evidenziare (default "KRAS").
    coocc_params : dict, optional
        Soglie per co-occorrenza: {"p_val": 0.05, "log2or": 1.0}.
        Se None usa i valori di default.
    me_params : dict, optional
        Soglie per mutua esclusività: {"p_val": 0.01, "log2or": -1.0}.
        Se None usa i valori di default.
    """
    if coocc_params is None:
        coocc_params = {"p_val": 0.05, "log2or": 1.0}
    if me_params is None:
        me_params = {"p_val": 0.01, "log2or": -1.0}

    cohort_name = _cohort_name(path)
    print(f"\n--- 🌋 GENERAZIONE PLOTS (P-Adj): {cohort_name.upper()} ---")

    out_dir   = _out(path, "plots")
    stats_dir = os.path.join(path, "stats")

    # --- Plot Co-occorrenze ---
    full_file = os.path.join(stats_dir, f"Full_Cooccurrence_Stats_{cohort_name}.tsv")
    if os.path.exists(full_file):
        df = pd.read_csv(full_file, sep="\t")
        df_kras = df[
            (df["Gene_A"] == target_gene) | (df["Gene_B"] == target_gene)
        ].copy()

        if not df_kras.empty:
            p_thresh  = coocc_params["p_val"]
            log_thresh = coocc_params["log2or"]

            df_kras["Partner"] = df_kras.apply(
                lambda row: row["Gene_B"] if row["Gene_A"] == target_gene else row["Gene_A"],
                axis=1,
            )
            df_kras["neg_log10_p_adj"] = -np.log10(df_kras["P_Adj"] + 1e-300)
            cond_sig = (df_kras["Log2OR"] >= log_thresh) & (df_kras["P_Adj"] <= p_thresh)
            df_kras["Stato"] = np.where(cond_sig, "Significativo", "Non Significativo")

            plt.figure(figsize=(12, 7))
            sns.scatterplot(
                data=df_kras, x="Log2OR", y="neg_log10_p_adj", hue="Stato",
                palette={"Significativo": "#d62728", "Non Significativo": "#b0b0b0"},
                alpha=0.7, s=60, edgecolor=None,
            )
            plt.axvline(x=log_thresh, color="red", linestyle="--", alpha=0.5,
                        label=f"Soglia Log2OR (>= {log_thresh})")
            plt.axhline(y=-np.log10(p_thresh), color="blue", linestyle="--", alpha=0.5,
                        label=f"Soglia P-Adj (<= {p_thresh})")

            for _, row in df_kras[cond_sig].iterrows():
                plt.text(
                    row["Log2OR"] + 0.05, row["neg_log10_p_adj"] + 0.05,
                    row["Partner"], fontsize=9, weight="bold", color="black",
                )

            plt.title(
                f"Volcano Plot Co-occorrenze ({target_gene}) - {cohort_name.upper()} (FDR Corrected)",
                fontsize=14,
            )
            plt.xlabel("Forza dell'associazione (Log2 Odds Ratio)")
            plt.ylabel("Significatività (-Log10 P-Adjusted)")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            plt.savefig(
                os.path.join(out_dir, f"Volcano_Cooccurrence_{cohort_name}.png"), dpi=150
            )
            plt.show()
            print("✅ Volcano Plot Co-occorrenze (P-Adj) salvato e visualizzato.\n")

    # --- Plot Mutua Esclusione ---
    me_file = os.path.join(stats_dir, f"Mutual_Exclusivity_Stats_{cohort_name}.tsv")
    if os.path.exists(me_file):
        df_me = pd.read_csv(me_file, sep="\t")
        p_thresh_me  = me_params["p_val"]
        log_thresh_me = me_params["log2or"]

        df_me["neg_log10_p_adj"] = -np.log10(df_me["P_Adj"] + 1e-300)
        cond_sig_me = (df_me["Log2OR"] <= log_thresh_me) & (df_me["P_Adj"] <= p_thresh_me)
        df_me["Stato"] = np.where(cond_sig_me, "Significativo", "Non Significativo")

        plt.figure(figsize=(12, 7))
        sns.scatterplot(
            data=df_me, x="Log2OR", y="neg_log10_p_adj", hue="Stato",
            palette={"Significativo": "#1f77b4", "Non Significativo": "#b0b0b0"},
            alpha=0.7, s=60, edgecolor=None,
        )
        plt.axvline(x=log_thresh_me, color="red", linestyle="--", alpha=0.5,
                    label=f"Soglia Log2OR (<= {log_thresh_me})")
        plt.axhline(y=-np.log10(p_thresh_me), color="blue", linestyle="--", alpha=0.5,
                    label=f"Soglia P-Adj (<= {p_thresh_me})")

        for _, row in df_me[cond_sig_me].iterrows():
            plt.text(
                row["Log2OR"] - 0.05, row["neg_log10_p_adj"] + 0.05,
                row["Gene_B"], fontsize=9, weight="bold", color="black",
                horizontalalignment="right",
            )

        plt.title(
            f"Volcano Plot Mutua Esclusione ({target_gene}) - {cohort_name.upper()} (FDR Corrected)",
            fontsize=14,
        )
        plt.xlabel("Mutua Esclusività (Log2 Odds Ratio)")
        plt.ylabel("Significatività (-Log10 P-Adjusted)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, f"Volcano_Mutual_Exclusivity_{cohort_name}.png"), dpi=150
        )
        plt.show()
        print("✅ Volcano Plot Mutua Esclusione (P-Adj) salvato e visualizzato.\n")


# ---------------------------------------------------------------------------
# 4. RETI STATICHE 2D (FULL + FILTERED)
# ---------------------------------------------------------------------------

def _draw_static_network(
    G: nx.Graph,
    coocc_matrix: pd.DataFrame,
    cohort_name: str,
    out_base: str,
    target_gene: str,
    is_full: bool = False,
) -> None:
    """Disegna e salva una rete statica (interna)."""
    if G.number_of_nodes() == 0:
        return

    # --- Community detection ---
    if G.number_of_nodes() < 12:
        communities = [set(G.nodes())]
    else:
        try:
            import leidenalg
            import igraph as ig
            from sklearn.metrics import silhouette_score

            nodes = list(G.nodes())
            node_idx = {n: i for i, n in enumerate(nodes)}
            edges_ig = [(node_idx[u], node_idx[v]) for u, v in G.edges()]
            weights = [G[u][v].get("weight", 1) for u, v in G.edges()]

            g_ig = ig.Graph(n=len(nodes), edges=edges_ig, directed=False)
            g_ig.es["weight"] = weights
            g_ig.vs["name"] = nodes

            partition = leidenalg.find_partition(
                g_ig, leidenalg.ModularityVertexPartition,
                weights="weight", seed=42,
            )
            communities = [set(nodes[i] for i in part) for part in partition]

            modularity = partition.modularity
            try:
                lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight=None))
                dist_matrix = np.zeros((len(nodes), len(nodes)))
                for i, n1 in enumerate(nodes):
                    for j, n2 in enumerate(nodes):
                        if i == j:
                            dist_matrix[i, j] = 0
                        else:
                            dist_matrix[i, j] = lengths.get(n1, {}).get(n2, 100)
                labels = np.zeros(len(nodes))
                for c_idx, comm in enumerate(communities):
                    for node in comm:
                        labels[node_idx[node]] = c_idx
                if 1 < len(communities) < len(nodes):
                    silhouette = silhouette_score(dist_matrix, labels, metric="precomputed")
                else:
                    silhouette = 0.0
            except Exception:
                silhouette = 0.0

            print(
                f"\n📊 LEIDEN METRICS -> K: {len(communities)} | "
                f"Modularity: {modularity:.4f} | Silhouette: {silhouette:.4f}"
            )

        except Exception:
            try:
                communities = nx_comm.louvain_communities(G, weight="weight", seed=42)
            except AttributeError:
                communities = nx_comm.greedy_modularity_communities(G, weight="weight")

    cluster_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            cluster_map[node] = i

    # Salvataggio cluster
    cluster_data = [{"Cluster_ID": c_id, "Gene": node} for node, c_id in cluster_map.items()]
    df_cluster = pd.DataFrame(cluster_data).sort_values(by=["Cluster_ID", "Gene"])

    net_dir = os.path.join(out_base, "networks")
    os.makedirs(net_dir, exist_ok=True)
    net_suffix = "FULL" if is_full else "FILTERED"
    cluster_path = os.path.join(net_dir, f"Cluster_Genes_{net_suffix}_{cohort_name}.tsv")
    df_cluster.to_csv(cluster_path, sep="\t", index=False)
    print(f"📁 File cluster salvato in: {cluster_path}")

    # --- Grafico ---
    plt.figure(figsize=(18, 14) if is_full else (14, 10))
    pos = nx.spring_layout(G, k=0.5 if is_full else 0.9, iterations=50, seed=42)

    if target_gene in G.nodes():
        partner_nodes = [n for n in G.nodes() if n != target_gene]
        target_nodes  = [target_gene]
    else:
        partner_nodes = list(G.nodes())
        target_nodes  = []

    partner_colors = [cluster_map.get(n, 0) for n in partner_nodes]
    partner_sizes  = [
        min(max(150, int(coocc_matrix.loc[n, n]) * 25 if n in coocc_matrix.index else 150), 3000)
        for n in partner_nodes
    ]

    edge_list = list(G.edges())
    weights = [max(1.0, np.log2(G[u][v]["weight"]) * 1.2) for u, v in edge_list]
    nx.draw_networkx_edges(G, pos, alpha=0.25, width=weights, edge_color="gray")

    if partner_nodes:
        vmax_val = max(max(partner_colors), 1) if partner_colors else 1
        nx.draw_networkx_nodes(
            G, pos, nodelist=partner_nodes,
            node_color=partner_colors, cmap=plt.cm.tab20,
            vmin=0, vmax=vmax_val,
            node_size=partner_sizes, alpha=0.9,
            edgecolors="white", linewidths=1.5,
        )

    if target_nodes:
        target_tot = (
            int(coocc_matrix.loc[target_gene, target_gene])
            if target_gene in coocc_matrix.index else 50
        )
        target_size = min(max(800, target_tot * 25), 4000)
        nx.draw_networkx_nodes(
            G, pos, nodelist=target_nodes,
            node_color="#d62728", node_shape="*",
            node_size=target_size, edgecolors="black", linewidths=1.5,
        )

    nx.draw_networkx_labels(
        G, pos,
        font_size=8 if is_full else 10,
        font_weight="bold", font_color="black",
    )

    net_type = "FULL" if is_full else f"FILTRATA ({target_gene})"
    plt.title(
        f"Rete Co-occorrenze {net_type} - {cohort_name.upper()}\n"
        f"Cluster identificati: {len(communities)}",
        fontsize=16, fontweight="bold",
    )

    if target_nodes:
        plt.plot([], [], "*", color="#d62728", markersize=15,
                 label=f"Target ({target_gene})", markeredgecolor="black")
    plt.plot([], [], "o", color="gray", markersize=10,
             label="Geni Partner (colore = cluster)")
    plt.legend(loc="upper right", scatterpoints=1, frameon=True, shadow=True)
    plt.axis("off")
    plt.tight_layout()

    filename = (
        f"Static_Full_Network_{cohort_name}.png"
        if is_full
        else f"Static_Filtered_Network_{target_gene}_{cohort_name}.png"
    )
    img_path = os.path.join(net_dir, filename)
    plt.savefig(img_path, dpi=200)
    plt.show()
    print(
        f"✅ Rete {net_type} per {cohort_name.upper()} salvata. "
        f"Nodi: {G.number_of_nodes()}, Archi: {G.number_of_edges()}"
    )


def build_all_networks(
    path: str,
    target_gene: str = "KRAS",
    coocc_params: dict | None = None,
) -> None:
    """
    Costruisce la rete full di co-occorrenze e la rete filtrata 1-hop
    attorno a `target_gene`. Salva immagini e file cluster in `path/networks/`.

    Parameters
    ----------
    path : str
        Stessa cartella passata alle funzioni precedenti.
    target_gene : str
        Gene target (default "KRAS").
    coocc_params : dict, optional
        Soglie: {"p_val": 0.05, "log2or": 1.0, "min_cooc": 3}.
    """
    if coocc_params is None:
        coocc_params = {"p_val": 0.05, "log2or": 1.0, "min_cooc": 3}

    cohort_name = _cohort_name(path)
    print(f"\n--- 🌐 GENERAZIONE RETI 2D: {cohort_name.upper()} ---")

    stats_file = os.path.join(path, "stats", f"Full_Cooccurrence_Stats_{cohort_name}.tsv")
    coocc_file = os.path.join(path, "matrices", f"M_cooccurrence_{cohort_name}.tsv")

    if not os.path.exists(stats_file) or not os.path.exists(coocc_file):
        print(f"[!] File mancanti per {cohort_name}. Eseguire prima le fasi precedenti.")
        return

    df_stats    = pd.read_csv(stats_file, sep="\t")
    coocc_matrix = pd.read_csv(coocc_file, sep="\t", index_col=0)

    p_thresh   = coocc_params["p_val"]
    log_thresh = coocc_params["log2or"]
    min_cooc   = coocc_params["min_cooc"]

    valid_edges = df_stats[
        (df_stats["P_Value"] <= p_thresh) &
        (df_stats["Log2OR"] >= log_thresh) &
        (df_stats["Co_Occurrence_Count"] >= min_cooc)
    ]

    G_full = nx.Graph()
    for _, row in valid_edges.iterrows():
        G_full.add_edge(row["Gene_A"], row["Gene_B"], weight=int(row["Co_Occurrence_Count"]))

    if G_full.number_of_nodes() == 0:
        print(f"[-] Nessuna rete formata per {cohort_name}.")
        return

    # Rete FULL
    _draw_static_network(G_full, coocc_matrix, cohort_name, path, target_gene, is_full=True)

    # Rete FILTRATA (1-hop)
    if target_gene in G_full.nodes():
        neighbors = list(G_full.neighbors(target_gene))
        G_filtered = G_full.subgraph(neighbors + [target_gene]).copy()
        _draw_static_network(G_filtered, coocc_matrix, cohort_name, path, target_gene, is_full=False)
    else:
        print(f"[!] {target_gene} non presente nella rete (non supera i filtri).")


# ---------------------------------------------------------------------------
# 5. METRICHE TOPOLOGICHE
# ---------------------------------------------------------------------------

def calculate_metrics(
    path: str,
    target_gene: str = "KRAS",
    coocc_params: dict | None = None,
) -> None:
    """
    Calcola le metriche topologiche (degree, betweenness, closeness,
    clustering) sulla rete filtrata 1-hop attorno a `target_gene`.
    Salva i risultati in `path/stats/`.

    Parameters
    ----------
    path : str
        Stessa cartella passata alle funzioni precedenti.
    target_gene : str
        Gene target (default "KRAS").
    coocc_params : dict, optional
        Soglie: {"p_val": 0.05, "log2or": 1.0, "min_cooc": 3}.
    """
    if coocc_params is None:
        coocc_params = {"p_val": 0.05, "log2or": 1.0, "min_cooc": 3}

    cohort_name = _cohort_name(path)
    print(f"\n--- 📊 ESTRAZIONE METRICHE: {cohort_name.upper()} ---")

    stats_file = os.path.join(path, "stats", f"Full_Cooccurrence_Stats_{cohort_name}.tsv")
    out_dir    = _out(path, "stats")

    if not os.path.exists(stats_file):
        print(f"[!] File stats mancante per {cohort_name}.")
        return

    df_stats = pd.read_csv(stats_file, sep="\t")

    p_thresh   = coocc_params["p_val"]
    log_thresh = coocc_params["log2or"]
    min_cooc   = coocc_params["min_cooc"]

    valid_edges = df_stats[
        (df_stats["P_Value"] <= p_thresh) &
        (df_stats["Log2OR"] >= log_thresh) &
        (df_stats["Co_Occurrence_Count"] >= min_cooc)
    ]

    G_full = nx.Graph()
    for _, row in valid_edges.iterrows():
        G_full.add_edge(row["Gene_A"], row["Gene_B"], weight=int(row["Co_Occurrence_Count"]))

    if target_gene not in G_full.nodes():
        print(f"[-] {target_gene} non presente, calcolo metriche annullato.")
        return

    neighbors = list(G_full.neighbors(target_gene))
    G = G_full.subgraph(neighbors + [target_gene]).copy()

    if G.number_of_nodes() == 0:
        return

    degree_cent      = nx.degree_centrality(G)
    betweenness_cent = nx.betweenness_centrality(G)
    closeness_cent   = nx.closeness_centrality(G)
    clustering_coeffs = nx.clustering(G)

    metrics_data = [
        {
            "Gene": node,
            "Degree": G.degree(node),
            "Degree_Centrality": round(degree_cent.get(node, 0), 4),
            "Betweenness_Centrality": round(betweenness_cent.get(node, 0), 4),
            "Closeness_Centrality": round(closeness_cent.get(node, 0), 4),
            "Clustering_Coefficient": round(clustering_coeffs.get(node, 0), 4),
        }
        for node in G.nodes()
    ]

    metrics_df = pd.DataFrame(metrics_data).sort_values(
        by="Degree_Centrality", ascending=False
    )
    metrics_df.to_csv(
        os.path.join(out_dir, f"Network_Metrics_Filtered_{cohort_name}.tsv"),
        sep="\t", index=False,
    )

    print("✅ Metriche salvate. Top 3 Hub identificati:")
    print(metrics_df[["Gene", "Degree", "Betweenness_Centrality"]].head(3).to_string(index=False))


# ---------------------------------------------------------------------------
# 6. CONFRONTO FRA DUE COORTI
# ---------------------------------------------------------------------------

def compare_networks(
    path_global: str,
    path_filtered: str,
    coocc_params_global: dict | None = None,
    coocc_params_filtered: dict | None = None,
) -> None:
    """
    Confronta la rete di co-occorrenze di due coorti distinte,
    evidenziando archi in comune, persi e guadagnati.

    Parameters
    ----------
    path_global : str
        Path della coorte di riferimento (es. "./pancreas").
    path_filtered : str
        Path della coorte filtrata (es. "./kras_pancreas").
    coocc_params_global : dict, optional
        Soglie per la coorte globale.
    coocc_params_filtered : dict, optional
        Soglie per la coorte filtrata.
    """
    if coocc_params_global is None:
        coocc_params_global = {"p_val": 0.05, "log2or": 1.0, "min_cooc": 3}
    if coocc_params_filtered is None:
        coocc_params_filtered = coocc_params_global

    global_cohort   = _cohort_name(path_global)
    filtered_cohort = _cohort_name(path_filtered)
    print(f"\n--- 🔀 CONFRONTO RETI: {global_cohort.upper()} vs {filtered_cohort.upper()} ---")

    stats_global   = os.path.join(path_global, "stats", f"Full_Cooccurrence_Stats_{global_cohort}.tsv")
    stats_filtered = os.path.join(path_filtered, "stats", f"Full_Cooccurrence_Stats_{filtered_cohort}.tsv")

    if not os.path.exists(stats_global) or not os.path.exists(stats_filtered):
        print("[!] Dati mancanti per il confronto.")
        return

    df_g = pd.read_csv(stats_global, sep="\t", low_memory=False)
    df_f = pd.read_csv(stats_filtered, sep="\t", low_memory=False)

    def _filter(df: pd.DataFrame, params: dict) -> pd.DataFrame:
        return df[
            (df["P_Value"] <= params["p_val"]) &
            (df["Log2OR"]  >= params["log2or"]) &
            (df["Co_Occurrence_Count"] >= params["min_cooc"])
        ]

    edges_g = _filter(df_g, coocc_params_global)
    edges_f = _filter(df_f, coocc_params_filtered)

    set_g = {tuple(sorted([r["Gene_A"], r["Gene_B"]])) for _, r in edges_g.iterrows()}
    set_f = {tuple(sorted([r["Gene_A"], r["Gene_B"]])) for _, r in edges_f.iterrows()}

    common   = set_g & set_f
    unique_g = set_g - set_f
    unique_f = set_f - set_g

    print(f"Archi totali in {global_cohort.upper()}: {len(set_g)}")
    print(f"Archi totali in {filtered_cohort.upper()}: {len(set_f)}")
    print(f"Archi in comune: {len(common)}")
    print(f"Archi persi (presenti solo in {global_cohort.upper()}): {len(unique_g)}")
    print(f"Archi guadagnati (presenti solo in {filtered_cohort.upper()}): {len(unique_f)}\n")

    if unique_f:
        print("Top 5 archi guadagnati (per Log2OR):")
        new_edges_df = edges_f[
            edges_f.apply(
                lambda r: tuple(sorted([r["Gene_A"], r["Gene_B"]])) in unique_f, axis=1
            )
        ]
        print(
            new_edges_df
            .sort_values(by="Log2OR", ascending=False)
            [["Gene_A", "Gene_B", "Log2OR", "P_Adj"]]
            .head()
            .to_string(index=False)
        )
        print()
