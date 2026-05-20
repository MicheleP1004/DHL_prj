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

import time
import re
import requests
import matplotlib.gridspec as gridspec
from collections import defaultdict
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
import gseapy as gp

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

# UTILIZZO DAL NOTEBOOK:
#
#   from utils import (
#       analyze_intracluster_metrics, find_intracluster_hubs,
#       enrich_network_hubs, plot_hub_enrichment,
#       run_cluster_enrichment, enrich_cluster_hubs,
#       compare_clustering_methods,
#       compute_pancancer_superhubs_robust, find_consensus_clusters,
#       run_all_aggregations,
#   )
#
#   PATH        = "./kras_pancreas"
#   TARGET_GENE = "KRAS"
#   COOCC_PARAMS = {"p_val": 0.05, "log2or": 1.0, "min_cooc": 3}
#
#   # Funzioni per singola coorte
#   analyze_intracluster_metrics(PATH, TARGET_GENE, is_full=True, coocc_params=COOCC_PARAMS)
#   find_intracluster_hubs(PATH, TARGET_GENE, is_full=True, coocc_params=COOCC_PARAMS)
#   enrich_network_hubs(PATH, TARGET_GENE, is_full=True)
#   plot_hub_enrichment(PATH, TARGET_GENE, is_full=True)
#   run_cluster_enrichment(PATH, TARGET_GENE, is_full=True)
#   enrich_cluster_hubs(PATH, TARGET_GENE, is_full=True)
#   compare_clustering_methods(PATH, TARGET_GENE, COOCC_PARAMS)
#
#   # Funzioni pan-cancer (lista di path + cartella output)
#   PAN_PATHS  = ["./kras_pancreas", "./kras_lung", "./kras_colon"]
#   PAN_OUTDIR = "./pan_cancer_integrated"
#   compute_pancancer_superhubs_robust(PAN_PATHS, PAN_OUTDIR)
#   find_consensus_clusters(PAN_PATHS, PAN_OUTDIR, min_overlap=3)
#   run_all_aggregations(PAN_PATHS, PAN_OUTDIR)
# ==============================================================================


# ---------------------------------------------------------------------------
# Dipendenze opzionali (community detection)
# ---------------------------------------------------------------------------
try:
    import leidenalg
    import igraph as ig
    _LEIDEN_AVAILABLE = True
except ImportError:
    _LEIDEN_AVAILABLE = False

try:
    import infomap as im_lib
    _INFOMAP_AVAILABLE = True
except ImportError:
    _INFOMAP_AVAILABLE = False


# ---------------------------------------------------------------------------
# 7. METRICHE TOPOLOGICHE INTRACLUSTER
# ---------------------------------------------------------------------------

def analyze_intracluster_metrics(
    path: str,
    target_gene: str = "KRAS",
    is_full: bool = True,
    coocc_params: dict | None = None,
) -> None:
    """
    Calcola le metriche topologiche (densità, degree, clustering, transitività,
    diametro, peso medio) per ogni cluster della rete. Salva in path/intracluster/.

    Parameters
    ----------
    path : str
        Cartella della coorte (es. "./kras_pancreas").
    target_gene : str
        Gene target usato per costruire la rete filtrata se is_full=False.
    is_full : bool
        True = rete completa; False = rete 1-hop attorno a target_gene.
    coocc_params : dict, optional
        Soglie: {"p_val": 0.05, "log2or": 1.0, "min_cooc": 3}.
    """
    if coocc_params is None:
        coocc_params = {"p_val": 0.05, "log2or": 1.0, "min_cooc": 3}

    cohort_name = _cohort_name(path)
    net_type = "FULL" if is_full else "FILTERED"
    print(f"\n{'='*80}")
    print(f"📊 METRICHE INTRACLUSTER: {cohort_name.upper()} ({net_type})")
    print("="*80)

    stats_file   = os.path.join(path, "stats",    f"Full_Cooccurrence_Stats_{cohort_name}.tsv")
    cluster_file = os.path.join(path, "networks", f"Cluster_Genes_{net_type}_{cohort_name}.tsv")

    if not os.path.exists(stats_file) or not os.path.exists(cluster_file):
        print(f"[!] File mancanti per l'analisi di {cohort_name} ({net_type}).")
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

    if is_full:
        G_work = G_full
    else:
        if target_gene not in G_full.nodes():
            print(f"[!] Target {target_gene} non presente nella rete, salto calcolo.")
            return
        neighbors = list(G_full.neighbors(target_gene))
        G_work = G_full.subgraph(neighbors + [target_gene]).copy()

    df_clusters = pd.read_csv(cluster_file, sep="\t")
    cluster_ids = sorted(df_clusters["Cluster_ID"].unique())

    metrics_list = []
    for c_id in cluster_ids:
        cluster_genes = df_clusters[df_clusters["Cluster_ID"] == c_id]["Gene"].tolist()
        valid_genes   = [g for g in cluster_genes if g in G_work.nodes()]
        if len(valid_genes) < 2:
            continue

        subG          = G_work.subgraph(valid_genes)
        n_nodes       = subG.number_of_nodes()
        n_edges       = subG.number_of_edges()
        density       = nx.density(subG)
        avg_degree    = (2 * n_edges) / n_nodes if n_nodes > 0 else 0
        avg_clustering = nx.average_clustering(subG, weight="weight")
        transitivity  = nx.transitivity(subG)
        n_components  = nx.number_connected_components(subG)
        avg_weight    = (
            sum(d.get("weight", 1) for _, _, d in subG.edges(data=True)) / n_edges
            if n_edges > 0 else 0
        )
        lcc_nodes = max(nx.connected_components(subG), key=len)
        lcc       = subG.subgraph(lcc_nodes)
        diameter  = nx.diameter(lcc) if lcc.number_of_nodes() > 1 else 0

        metrics_list.append({
            "Cluster_ID":      c_id,
            "N_Nodes":         n_nodes,
            "N_Edges":         n_edges,
            "N_Components":    n_components,
            "Density":         round(density, 3),
            "Avg_Degree":      round(avg_degree, 2),
            "Avg_Clustering":  round(avg_clustering, 3),
            "Transitivity":    round(transitivity, 3),
            "Diameter_LCC":    diameter,
            "Avg_Edge_Weight": round(avg_weight, 2),
        })

    if metrics_list:
        df_metrics = pd.DataFrame(metrics_list)
        out_dir  = _out(path, "intracluster")
        out_file = os.path.join(out_dir, f"Intracluster_Metrics_{net_type}_{cohort_name}.tsv")
        df_metrics.to_csv(out_file, sep="\t", index=False)
        print(df_metrics.to_string(index=False))
        print(f"\n✅ Metriche salvate in: {out_file}\n")
    else:
        print("[-] Nessun cluster valido per calcolare le metriche.\n")


# ---------------------------------------------------------------------------
# 8. HUB INTRACLUSTER (CENTRALITÀ)
# ---------------------------------------------------------------------------

def find_intracluster_hubs(
    path: str,
    target_gene: str = "KRAS",
    is_full: bool = True,
    coocc_params: dict | None = None,
) -> None:
    """
    Calcola degree / betweenness / closeness centrality per ogni nodo,
    cluster per cluster. Salva TSV e report testuale in path/intracluster/.
    """
    if coocc_params is None:
        coocc_params = {"p_val": 0.05, "log2or": 1.0, "min_cooc": 3}

    cohort_name = _cohort_name(path)
    net_type    = "FULL" if is_full else "FILTERED"
    print(f"\n{'='*80}")
    print(f"🎯 RICERCA HUB INTRACLUSTER: {cohort_name.upper()} ({net_type})")
    print("="*80)

    stats_file   = os.path.join(path, "stats",    f"Full_Cooccurrence_Stats_{cohort_name}.tsv")
    cluster_file = os.path.join(path, "networks", f"Cluster_Genes_{net_type}_{cohort_name}.tsv")

    if not os.path.exists(stats_file) or not os.path.exists(cluster_file):
        print(f"[!] File mancanti per {cohort_name} ({net_type}).")
        return

    df_stats = pd.read_csv(stats_file, sep="\t")
    p_thresh, log_thresh, min_cooc = (
        coocc_params["p_val"], coocc_params["log2or"], coocc_params["min_cooc"]
    )
    valid_edges = df_stats[
        (df_stats["P_Value"] <= p_thresh) &
        (df_stats["Log2OR"] >= log_thresh) &
        (df_stats["Co_Occurrence_Count"] >= min_cooc)
    ]
    G_full = nx.Graph()
    for _, row in valid_edges.iterrows():
        G_full.add_edge(row["Gene_A"], row["Gene_B"])   # unweighted per centralita' topologica

    if is_full:
        G_work = G_full
    else:
        if target_gene not in G_full.nodes():
            print(f"[!] Target {target_gene} non presente nella rete, salto calcolo.")
            return
        neighbors = list(G_full.neighbors(target_gene))
        G_work = G_full.subgraph(neighbors + [target_gene]).copy()

    df_clusters = pd.read_csv(cluster_file, sep="\t")
    cluster_ids = sorted(df_clusters["Cluster_ID"].unique())

    all_nodes_data = []
    for c_id in cluster_ids:
        cluster_genes = df_clusters[df_clusters["Cluster_ID"] == c_id]["Gene"].tolist()
        valid_genes   = [g for g in cluster_genes if g in G_work.nodes()]
        if len(valid_genes) < 3:
            continue
        subG     = G_work.subgraph(valid_genes)
        deg_cent = nx.degree_centrality(subG)
        bet_cent = nx.betweenness_centrality(subG)
        clo_cent = nx.closeness_centrality(subG)
        for gene in valid_genes:
            all_nodes_data.append({
                "Cluster_ID":             c_id,
                "Gene":                   gene,
                "Degree_Centrality":      round(deg_cent[gene], 4),
                "Betweenness_Centrality": round(bet_cent[gene], 4),
                "Closeness_Centrality":   round(clo_cent[gene], 4),
            })

    if not all_nodes_data:
        print("[-] Impossibile calcolare centralita' (cluster troppo piccoli o non validi).\n")
        return

    df_centrality = pd.DataFrame(all_nodes_data).sort_values(
        by=["Cluster_ID", "Degree_Centrality"], ascending=[True, False]
    )
    out_dir = _out(path, "intracluster")
    out_tsv = os.path.join(out_dir, f"Intracluster_Centrality_{net_type}_{cohort_name}.tsv")
    df_centrality.to_csv(out_tsv, sep="\t", index=False)
    print(f"✅ Tabella centralita' completa salvata in: {out_tsv}")

    out_txt = os.path.join(out_dir, f"Report_Prof_TopHubs_{net_type}_{cohort_name}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        titolo = f"--- TOP 3 HUB GENES PER OGNI CLUSTER ({cohort_name.upper()} - {net_type}) ---"
        f.write(f"{titolo}\n")
        for c_id in df_centrality["Cluster_ID"].unique():
            cluster_df = df_centrality[df_centrality["Cluster_ID"] == c_id]
            sep  = "-" * 60
            head = f"\n{sep}\nCLUSTER {c_id} | Totale geni nel network: {len(cluster_df)}\n{sep}\n"
            f.write(head)
            for _, row in cluster_df.head(3).iterrows():
                f.write(f"HUB: {row['Gene']}\n")
                f.write(f"   - Degree Centrality:      {row['Degree_Centrality']:.4f}\n")
                f.write(f"   - Betweenness Centrality: {row['Betweenness_Centrality']:.4f}\n")

    print(f"📄 Report Top Hubs salvato in: {out_txt}\n")
    print(" Top 3 Hub per ogni cluster:")
    for c_id in df_centrality["Cluster_ID"].unique():
        cluster_df = df_centrality[df_centrality["Cluster_ID"] == c_id]
        print(f"\n  CLUSTER {c_id} | Totale geni: {len(cluster_df)}")
        for _, row in cluster_df.head(3).iterrows():
            print(f"   HUB: {row['Gene']} - Degree Centrality: {row['Degree_Centrality']:.4f}")


# ---------------------------------------------------------------------------
# 9. ENRICHMENT HUB GLOBALI DELLA RETE
# ---------------------------------------------------------------------------

def enrich_network_hubs(
    path: str,
    target_gene: str = "KRAS",
    is_full: bool = True,
) -> None:
    """
    Estrae i top-3 hub per cluster dalla centralita' pre-calcolata,
    poi lancia Enrichr. Salva TSV e report in path/intracluster/enrichment/.
    """
    import gseapy as gp
    import time

    cohort_name    = _cohort_name(path)
    net_type       = "FULL" if is_full else "FILTERED"
    print(f"\n{'='*80}")
    print(f"ENRICHMENT DEGLI HUB: {cohort_name.upper()} ({net_type})")
    print("="*80)

    intracluster_dir = os.path.join(path, "intracluster")
    enrichment_dir   = _out(intracluster_dir, "enrichment")
    centrality_file  = os.path.join(intracluster_dir, f"Intracluster_Centrality_{net_type}_{cohort_name}.tsv")

    if not os.path.exists(centrality_file):
        print(f"[!] File centralita' non trovato. Eseguire prima find_intracluster_hubs().")
        return

    df_centrality = pd.read_csv(centrality_file, sep="\t")
    hub_genes = []
    for c_id in df_centrality["Cluster_ID"].unique():
        hub_genes.extend(df_centrality[df_centrality["Cluster_ID"] == c_id].head(3)["Gene"].tolist())
    hub_genes = list(set(hub_genes))

    if len(hub_genes) < 2:
        print(f"[-] Solo {len(hub_genes)} Hub trovati. Troppo pochi per un enrichment valido.")
        return

    print(f"Estratti {len(hub_genes)} Hub unici. Avvio Enrichment...")
    databases = ["KEGG_2021_Human", "GO_Biological_Process_2021", "Reactome_2022"]

    try:
        enr     = gp.enrichr(gene_list=hub_genes, gene_sets=databases, organism="human", outdir=None)
        sig_res = enr.results[enr.results["P-value"] < 0.05].copy()
        time.sleep(1)
    except Exception as e:
        print(f"[!] Errore Enrichr: {e}")
        return

    if sig_res.empty:
        print(f"[-] Nessun pathway significativo trovato per i {len(hub_genes)} Hub.")
        return

    sig_res = sig_res.sort_values("P-value")
    out_tsv = os.path.join(enrichment_dir, f"Enrichment_NetworkHubs_{net_type}_{cohort_name}.tsv")
    sig_res.to_csv(out_tsv, sep="\t", index=False)

    out_txt = os.path.join(enrichment_dir, f"Report_NetworkHubs_Top5perDB_{net_type}_{cohort_name}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        titolo = (f"--- TOP 5 PATHWAY HUB GLOBALI DELLA RETE "
                  f"({cohort_name.upper()} - {net_type}) ---")
        f.write(f"{titolo}\n")
        f.write(f"Geni Hub analizzati ({len(hub_genes)}): {', '.join(hub_genes)}\n")
        f.write("="*80 + "\n\n")
        for db_label, db_keyword in [("GO", "GO_"), ("KEGG", "KEGG_"), ("Reactome", "Reactome")]:
            db_top5 = sig_res[sig_res["Gene_set"].str.contains(db_keyword, case=False)].head(5)
            if db_top5.empty:
                continue
            f.write(f"\n  [{db_label}]\n")
            for _, row in db_top5.iterrows():
                f.write(f"  [{row['Gene_set']}] {row['Term']}\n")
                f.write(f"     - P-value: {row['P-value']:.2e} | Adjusted: {row['Adjusted P-value']:.2e}\n")
                f.write(f"     - Overlap: {row['Overlap']}\n")
                f.write(f"     - Hubs coinvolti: {row['Genes']}\n\n")

    print(f"✅ Risultati salvati in: {out_tsv}")
    print(" Top 5 pathway significativi per database:")
    for db_label, db_keyword in [("GO", "GO_"), ("KEGG", "KEGG_"), ("Reactome", "Reactome")]:
        db_top5 = sig_res[sig_res["Gene_set"].str.contains(db_keyword, case=False)].head(5)
        if not db_top5.empty:
            print(f"  [{db_label}]")
            for _, row in db_top5.iterrows():
                print(f"   [{row['Gene_set']}] {row['Term']}")


# ---------------------------------------------------------------------------
# 10. BARPLOT ENRICHMENT DEGLI HUB
# ---------------------------------------------------------------------------

def plot_hub_enrichment(
    path: str,
    target_gene: str = "KRAS",
    is_full: bool = True,
) -> None:
    """
    Genera il barplot dei top-10 pathway dall'enrichment degli hub globali.
    Salva il PNG in path/plots/.
    """
    cohort_name = _cohort_name(path)
    net_type    = "FULL" if is_full else "FILTERED"

    enrichment_file = os.path.join(
        path, "intracluster", "enrichment",
        f"Enrichment_NetworkHubs_{net_type}_{cohort_name}.tsv",
    )
    if not os.path.exists(enrichment_file):
        return

    df_enr = pd.read_csv(enrichment_file, sep="\t")
    if df_enr.empty:
        return

    print(f"📊 Generazione grafico enrichment per: {cohort_name.upper()} ({net_type})...")

    top_10 = df_enr.sort_values("P-value").head(10).copy()
    top_10["Minus_Log10_Pval"] = -np.log10(top_10["P-value"])
    top_10["Term"] = top_10["Term"].apply(lambda x: (x[:45] + "...") if len(x) > 45 else x)

    plt.figure(figsize=(10, 6))
    palette_color = "Reds_r" if ("FILTRATA" in net_type or "kras_" in cohort_name) else "Blues_r"
    sns.barplot(
        x="Minus_Log10_Pval", y="Term", data=top_10,
        palette=palette_color, hue="Term", legend=False,
    )
    plt.title(
        f"Top 10 Pathway degli Hub\n{cohort_name.upper()} - {net_type}",
        fontsize=14, pad=15, fontweight="bold",
    )
    plt.xlabel("-Log10(P-value)", fontsize=12)
    plt.ylabel("")
    plt.axvline(x=1.3, color="red", linestyle="--", alpha=0.7, label="P-value = 0.05")
    plt.legend(loc="lower right")
    plt.tight_layout()

    out_dir = _out(path, "plots")
    out_png = os.path.join(out_dir, f"Barplot_Hub_Enrichment_{net_type}_{cohort_name}.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# 11. ENRICHMENT PER OGNI CLUSTER
# ---------------------------------------------------------------------------

def run_cluster_enrichment(
    path: str,
    target_gene: str = "KRAS",
    is_full: bool = True,
) -> None:
    """
    Lancia Enrichr su ogni cluster (>=4 geni) e salva TSV + report testuale
    in path/networks/enrichment/.
    """
    import gseapy as gp
    import time

    cohort_name    = _cohort_name(path)
    net_type       = "FULL" if is_full else "FILTERED"
    print(f"\n--- AVVIO ENRICHMENT AUTOMATICO: {cohort_name.upper()} ({net_type}) ---")

    cluster_dir    = os.path.join(path, "networks")
    enrichment_dir = _out(os.path.join(path, "networks"), "enrichment")
    cluster_file   = os.path.join(cluster_dir, f"Cluster_Genes_{net_type}_{cohort_name}.tsv")

    if not os.path.exists(cluster_file):
        print(f"[!] File cluster non trovato: {cluster_file}")
        return

    df_clusters = pd.read_csv(cluster_file, sep="\t", low_memory=False)

    available_libs  = gp.get_library_name(organism="human")
    reactome_vers   = sorted(lib for lib in available_libs if "Reactome" in lib)
    latest_reactome = reactome_vers[-1] if reactome_vers else "Reactome_2016"
    print(f"[*] Auto-detect Reactome: '{latest_reactome}'")

    databases = [
        "KEGG_2021_Human", "GO_Biological_Process_2021",
        "GO_Cellular_Component_2021", "GO_Molecular_Function_2021",
        latest_reactome,
    ]

    all_results = []
    for c_id in df_clusters["Cluster_ID"].unique():
        genes = df_clusters[df_clusters["Cluster_ID"] == c_id]["Gene"].tolist()
        if len(genes) < 4:
            continue
        print(f"Analizzo Cluster {c_id} ({len(genes)} geni)...")
        try:
            enr     = gp.enrichr(gene_list=genes, gene_sets=databases, organism="human", outdir=None)
            sig_res = enr.results[enr.results["P-value"] < 0.05].copy()
            if not sig_res.empty:
                sig_res["Cluster_ID"] = c_id
                all_results.append(sig_res)
            time.sleep(1)
        except Exception as e:
            print(f"Errore nel Cluster {c_id}: {e}")

    if not all_results:
        print("[-] Nessun pathway significativo trovato per questi cluster.\n")
        return

    final_df = pd.concat(all_results, ignore_index=True)
    cols     = ["Cluster_ID", "Gene_set", "Term", "Overlap", "P-value", "Adjusted P-value", "Genes"]
    final_df = final_df[cols].sort_values(by=["Cluster_ID", "P-value"], ascending=[True, True])

    out_tsv = os.path.join(enrichment_dir, f"Enrichment_Clusters_{net_type}_{cohort_name}.tsv")
    final_df.to_csv(out_tsv, sep="\t", index=False)
    print(f"✅ Tabella dati salvata in: {out_tsv}")

    out_txt = os.path.join(enrichment_dir, f"Report_Clusters_Top5perDB_{net_type}_{cohort_name}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        titolo = (f"--- TOP 5 PATHWAY PER DATABASE PER OGNI CLUSTER "
                  f"({cohort_name.upper()} - {net_type}) ---")
        print(f"\n{titolo}")
        f.write(f"{titolo}\n")
        for c_id in final_df["Cluster_ID"].unique():
            cluster_genes = df_clusters[df_clusters["Cluster_ID"] == c_id]["Gene"].tolist()
            genes_str = ", ".join(cluster_genes)
            sep   = "=" * 80
            head1 = f"CLUSTER {c_id}  |  DIMENSIONE: {len(cluster_genes)} geni"
            head2 = f"TUTTI I GENI: {genes_str}"
            print(f"\n{sep}\n{head1}\n{head2}\n{sep}")
            f.write(f"\n{sep}\n{head1}\n{head2}\n{sep}\n")
            cluster_data = final_df[final_df["Cluster_ID"] == c_id]
            for db_label, db_keyword in [("GO", "GO_"), ("KEGG", "KEGG_"), ("Reactome", "Reactome")]:
                db_top5 = cluster_data[cluster_data["Gene_set"].str.contains(db_keyword, case=False)].head(5)
                if db_top5.empty:
                    continue
                db_header = f"  [{db_label}]"
                print(db_header)
                f.write(f"\n{db_header}\n")
                for _, row in db_top5.iterrows():
                    line1 = f"  [{row['Gene_set']}] {row['Term']}"
                    line2 = f"     P-value: {row['P-value']:.2e}  |  Overlap: {row['Overlap']}"
                    line3 = f"     Geni nel pathway: {row['Genes']}"
                    print(f"{line1}\n{line2}\n{line3}")
                    f.write(f"{line1}\n{line2}\n{line3}\n")

    print(f"\n📄 REPORT SALVATO IN: {out_txt}\n")


# ---------------------------------------------------------------------------
# 12. ENRICHMENT HUB LOCALI PER OGNI SINGOLO CLUSTER
# ---------------------------------------------------------------------------

def enrich_cluster_hubs(
    path: str,
    target_gene: str = "KRAS",
    is_full: bool = True,
) -> None:
    """
    Per ogni cluster prende i top-5 hub locali (dalla centralita') e lancia
    Enrichr su di essi. Report testuale in path/intracluster/enrichment/.
    """
    import gseapy as gp
    import time

    cohort_name      = _cohort_name(path)
    net_type         = "FULL" if is_full else "FILTERED"
    print(f"\n{'='*80}")
    print(f"ENRICHMENT HUB DEI SINGOLI CLUSTER: {cohort_name.upper()} ({net_type})")
    print("="*80)

    intracluster_dir = os.path.join(path, "intracluster")
    enrichment_dir   = _out(intracluster_dir, "enrichment")
    centrality_file  = os.path.join(intracluster_dir, f"Intracluster_Centrality_{net_type}_{cohort_name}.tsv")

    if not os.path.exists(centrality_file):
        print(f"[!] File centralita' non trovato. Eseguire prima find_intracluster_hubs().")
        return

    df_centrality = pd.read_csv(centrality_file, sep="\t")
    databases     = ["KEGG_2021_Human", "GO_Biological_Process_2021", "Reactome_2022"]

    out_txt = os.path.join(enrichment_dir, f"Report_ClusterHubs_Top5perDB_{net_type}_{cohort_name}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        titolo = (f"--- TOP 5 PATHWAY HUB LOCALI PER OGNI CLUSTER "
                  f"({cohort_name.upper()} - {net_type}) ---")
        f.write(f"{titolo}\n\n")
        for c_id in df_centrality["Cluster_ID"].unique():
            cluster_df = df_centrality[df_centrality["Cluster_ID"] == c_id]
            top_hubs   = cluster_df.head(5)["Gene"].tolist()
            f.write("="*80 + "\n")
            f.write(f"CLUSTER {c_id}  |  TOP HUB ANALIZZATI: {len(top_hubs)}\n")
            f.write(f"GENI HUB: {', '.join(top_hubs)}\n")
            f.write("="*80 + "\n\n")
            if len(top_hubs) < 2:
                f.write("[-] Troppi pochi hub per un enrichment.\n\n")
                print(f" Cluster {c_id}: Saltato (troppi pochi hub).")
                continue
            print(f" Analizzo Hub del Cluster {c_id}...")
            try:
                enr     = gp.enrichr(gene_list=top_hubs, gene_sets=databases, organism="human", outdir=None)
                sig_res = enr.results[enr.results["P-value"] < 0.05].copy()
                time.sleep(1)
            except Exception:
                f.write("[-] Errore Enrichr per questo cluster.\n\n")
                continue
            if not sig_res.empty:
                sig_res = sig_res.sort_values("P-value")
                for db_label, db_keyword in [("GO", "GO_"), ("KEGG", "KEGG_"), ("Reactome", "Reactome")]:
                    db_top5 = sig_res[sig_res["Gene_set"].str.contains(db_keyword, case=False)].head(5)
                    if db_top5.empty:
                        continue
                    f.write(f"  [{db_label}]\n")
                    for _, row in db_top5.iterrows():
                        f.write(f"  [{row['Gene_set']}] {row['Term']}\n")
                        f.write(f"     - P-value: {row['P-value']:.2e} | Overlap: {row['Overlap']}\n")
                        f.write(f"     - Geni nel pathway: {row['Genes']}\n\n")
            else:
                f.write("[-] Nessun pathway significativo trovato.\n\n")

    print(f"✅ Report salvato: {out_txt}")


# ---------------------------------------------------------------------------
# 13. CONFRONTO METODI DI CLUSTERING
# ---------------------------------------------------------------------------

# Utility interne

def _nx_to_igraph(G: nx.Graph):
    nodes    = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    edges    = [(node_idx[u], node_idx[v]) for u, v in G.edges()]
    weights  = [G[u][v].get("weight", 1) for u, v in G.edges()]
    g = ig.Graph(n=len(nodes), edges=edges, directed=False)
    g.es["weight"] = weights
    g.vs["name"]   = nodes
    return g, nodes


def _run_louvain(G: nx.Graph):
    return nx_comm.louvain_communities(G, weight="weight", seed=42)


def _run_leiden(G: nx.Graph):
    if not _LEIDEN_AVAILABLE:
        return None
    g_ig, nodes = _nx_to_igraph(G)
    partition = leidenalg.find_partition(
        g_ig, leidenalg.ModularityVertexPartition, weights="weight", seed=42
    )
    return [set(nodes[i] for i in part) for part in partition]


def _run_girvan_newman(G: nx.Graph, max_communities: int = 20):
    comp = nx_comm.girvan_newman(G)
    best_comms, best_mod = None, -1
    for comms_tuple in comp:
        comms = list(comms_tuple)
        if len(comms) > max_communities:
            break
        mod = nx_comm.modularity(G, comms, weight="weight")
        if mod > best_mod:
            best_mod, best_comms = mod, comms
    return [set(c) for c in best_comms] if best_comms else None


def _run_infomap(G: nx.Graph):
    if not _INFOMAP_AVAILABLE:
        return None
    from collections import defaultdict
    im       = im_lib.Infomap(silent=True)
    nodes    = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    for u, v, d in G.edges(data=True):
        im.add_link(node_idx[u], node_idx[v], d.get("weight", 1))
    im.run()
    groups: dict = defaultdict(set)
    for node in im.tree:
        if node.is_leaf:
            groups[node.module_id].add(nodes[node.node_id])
    return list(groups.values())


def _comms_to_labels(G: nx.Graph, communities) -> dict:
    label_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            label_map[node] = i
    return label_map


def _labels_vector(G: nx.Graph, label_map: dict) -> np.ndarray:
    return np.array([label_map.get(n, -1) for n in G.nodes()])


def _compute_silhouette(G: nx.Graph, label_map: dict) -> float:
    from sklearn.metrics import silhouette_score as _sil
    nodes  = list(G.nodes())
    labels = [label_map.get(n, -1) for n in nodes]
    n      = len(nodes)
    dist   = np.ones((n, n))
    idx    = {nd: i for i, nd in enumerate(nodes)}
    for u, v, d in G.edges(data=True):
        w  = d.get("weight", 1)
        dv = 1.0 / (w + 1e-9)
        dist[idx[u]][idx[v]] = dist[idx[v]][idx[u]] = dv
    np.fill_diagonal(dist, 0)
    if len(set(labels)) < 2:
        return float("nan")
    try:
        return round(_sil(dist, labels, metric="precomputed"), 4)
    except Exception:
        return float("nan")


def _plot_clustering_comparison(G: nx.Graph, results: dict, cohort_name: str, out_dir: str) -> None:
    import matplotlib.gridspec as gridspec
    valid_methods = [(m, r) for m, r in results.items() if r is not None]
    n_methods = len(valid_methods)
    if n_methods == 0:
        return

    fig = plt.figure(figsize=(5 * n_methods, 14))
    gs  = gridspec.GridSpec(2, n_methods, height_ratios=[2, 1], hspace=0.4, wspace=0.3)
    pos  = nx.spring_layout(G, k=0.6, iterations=40, seed=42)
    cmap = plt.cm.tab20

    for col, (method_name, r) in enumerate(valid_methods):
        ax          = fig.add_subplot(gs[0, col])
        label_map   = r["label_map"]
        n_comm      = r["n_communities"]
        node_colors = [cmap(label_map.get(n, 0) / max(n_comm - 1, 1)) for n in G.nodes()]
        node_sizes  = [min(max(80, int(G.degree(n, weight="weight")) * 3), 500) for n in G.nodes()]
        edge_weights = [max(0.3, np.log2(G[u][v]["weight"] + 1) * 0.8) for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, width=edge_weights, edge_color="gray")
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
                               alpha=0.85, edgecolors="white", linewidths=0.8)
        if G.number_of_nodes() <= 60:
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=6, font_weight="bold")
        ax.set_title(f"{method_name}\nK={n_comm}  Q={r['modularity']:.3f}",
                     fontsize=10, fontweight="bold", pad=6)
        ax.axis("off")

    metric_configs = [
        ("modularity",     "Modularity (Q)",   "steelblue"),
        ("silhouette",     "Silhouette score", "seagreen"),
        ("NMI_vs_Louvain", "NMI vs Louvain",  "darkorange"),
    ]
    ax_m  = fig.add_subplot(gs[1, :])
    x     = np.arange(n_methods)
    width = 0.22
    names = [m for m, _ in valid_methods]
    for i, (metric_key, metric_label, color) in enumerate(metric_configs):
        vals = [r.get(metric_key, float("nan")) if r else float("nan") for _, r in valid_methods]
        bars = ax_m.bar(x + i * width, vals, width, label=metric_label,
                        color=color, alpha=0.82, edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax_m.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                          f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax_m.set_xticks(x + width)
    ax_m.set_xticklabels(names, fontsize=9, rotation=15, ha="right")
    ax_m.set_ylabel("Score", fontsize=10)
    ax_m.set_ylim(0, 1.15)
    ax_m.legend(fontsize=9, loc="upper right")
    ax_m.set_title("Confronto metriche di qualita' clustering", fontsize=11, fontweight="bold")
    ax_m.axhline(0.3, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax_m.grid(axis="y", alpha=0.3)
    ax_m.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Confronto metodi di community detection - {cohort_name.upper()}",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    img_path = os.path.join(out_dir, f"Clustering_Comparison_{cohort_name}.png")
    plt.savefig(img_path, dpi=180, bbox_inches="tight")
    plt.show()
    print(f"✅ Plot comparativo salvato: {img_path}")


def compare_clustering_methods(
    path: str,
    target_gene: str = "KRAS",
    coocc_params: dict | None = None,
) -> tuple | None:
    """
    Esegue Louvain, Leiden, Girvan-Newman e Infomap sulla rete di co-occorrenze
    e confronta le metriche (modularity, silhouette, NMI). Salva TSV, cluster
    per ogni metodo e plot comparativo in path/networks/.

    Returns
    -------
    (df_summary, results) oppure None se la rete e' vuota.
    """
    import time
    from sklearn.metrics import normalized_mutual_info_score as _nmi

    if coocc_params is None:
        coocc_params = {"p_val": 0.05, "log2or": 1.0, "min_cooc": 3}

    cohort_name = _cohort_name(path)
    print(f"\n{'='*55}")
    print(f"  CONFRONTO CLUSTERING - {cohort_name.upper()}")
    print(f"{'='*55}")

    stats_file = os.path.join(path, "stats",    f"Full_Cooccurrence_Stats_{cohort_name}.tsv")
    coocc_file = os.path.join(path, "matrices", f"M_cooccurrence_{cohort_name}.tsv")

    if not os.path.exists(stats_file) or not os.path.exists(coocc_file):
        print(f"[!] File mancanti per {cohort_name}.")
        return None

    df_stats = pd.read_csv(stats_file, sep="\t", low_memory=False)
    p_thresh, log_thresh, min_cooc = (
        coocc_params["p_val"], coocc_params["log2or"], coocc_params["min_cooc"]
    )
    valid_edges = df_stats[
        (df_stats["P_Value"] <= p_thresh) &
        (df_stats["Log2OR"] >= log_thresh) &
        (df_stats["Co_Occurrence_Count"] >= min_cooc)
    ]
    G = nx.Graph()
    for _, row in valid_edges.iterrows():
        G.add_edge(row["Gene_A"], row["Gene_B"], weight=int(row["Co_Occurrence_Count"]))

    if G.number_of_nodes() == 0:
        print(f"[-] Rete vuota per {cohort_name}.")
        return None

    print(f"  Rete: {G.number_of_nodes()} nodi, {G.number_of_edges()} archi")

    methods = {
        "Louvain":       _run_louvain,
        "Leiden":        _run_leiden,
        "Girvan-Newman": lambda g: _run_girvan_newman(g, max_communities=25),
        "Infomap":       _run_infomap,
    }

    results: dict    = {}
    all_labels: dict = {}
    louvain_vec      = None

    for method_name, method_fn in methods.items():
        print(f"\n  {method_name}...", end=" ", flush=True)
        t0 = time.time()
        try:
            comms = method_fn(G)
        except Exception as e:
            print(f"ERRORE ({e})")
            results[method_name] = None
            continue
        elapsed = time.time() - t0

        if comms is None:
            print("non disponibile")
            results[method_name] = None
            continue

        label_map  = _comms_to_labels(G, comms)
        vec        = _labels_vector(G, label_map)
        all_labels[method_name] = vec
        modularity = nx_comm.modularity(G, comms, weight="weight")
        sil        = _compute_silhouette(G, label_map)
        if method_name == "Louvain":
            louvain_vec = vec
        results[method_name] = {
            "n_communities": len(comms),
            "modularity":    round(modularity, 4),
            "silhouette":    sil,
            "time_s":        round(elapsed, 2),
            "label_map":     label_map,
            "communities":   comms,
        }
        print(f"ok  Q={modularity:.3f}  k={len(comms)}  sil={sil}  ({elapsed:.1f}s)")

    for method_name, vec in all_labels.items():
        if method_name == "Louvain":
            results[method_name]["NMI_vs_Louvain"] = 1.0
        elif louvain_vec is not None:
            try:
                results[method_name]["NMI_vs_Louvain"] = round(_nmi(louvain_vec, vec), 4)
            except Exception:
                results[method_name]["NMI_vs_Louvain"] = float("nan")
        else:
            if results.get(method_name):
                results[method_name]["NMI_vs_Louvain"] = float("nan")

    rows = []
    for m, r in results.items():
        if r is None:
            rows.append({"Metodo": m, "K": "-", "Modularity": "-",
                         "Silhouette": "-", "NMI vs Louvain": "-", "Tempo (s)": "-"})
        else:
            rows.append({
                "Metodo":         m,
                "K":              r["n_communities"],
                "Modularity":     r["modularity"],
                "Silhouette":     r["silhouette"],
                "NMI vs Louvain": r.get("NMI_vs_Louvain", float("nan")),
                "Tempo (s)":      r["time_s"],
            })
    df_summary = pd.DataFrame(rows)
    print(f"\n{'─'*55}")
    print(df_summary.to_string(index=False))
    print(f"{'─'*55}")

    out_dir      = _out(path, "networks")
    summary_path = os.path.join(out_dir, f"Clustering_Comparison_{cohort_name}.tsv")
    df_summary.to_csv(summary_path, sep="\t", index=False)
    print(f"📊 Tabella metriche salvata: {summary_path}")

    for method_name, r in results.items():
        if r is None:
            continue
        rows_cl = [
            {"Metodo": method_name, "Cluster_ID": cid, "Gene": gene}
            for cid, comm in enumerate(r["communities"])
            for gene in comm
        ]
        safe_name = method_name.replace(" ", "_").replace("-", "")
        pd.DataFrame(rows_cl).to_csv(
            os.path.join(out_dir, f"Clusters_{safe_name}_{cohort_name}.tsv"),
            sep="\t", index=False,
        )

    _plot_clustering_comparison(G, results, cohort_name, out_dir)
    return df_summary, results


# ---------------------------------------------------------------------------
# 14. ANALISI PAN-CANCER
# ---------------------------------------------------------------------------

def compute_pancancer_superhubs_robust(
    paths: list,
    output_dir: str,
) -> None:
    """
    Identifica Super-Hub universali (logica Weakest Link) su una lista di
    coorti. Salva il TSV in output_dir.

    Parameters
    ----------
    paths : list[str]
        Lista di path delle coorti (es. ["./kras_pancreas", "./kras_lung", ...]).
    output_dir : str
        Cartella di output per i risultati pan-cancer.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("Identificazione Super-Hub Universali (Logica: Weakest Link)...")
    dfs = []

    for path in paths:
        cohort_name = _cohort_name(path)
        filepath = os.path.join(path, "intracluster", f"Intracluster_Centrality_FULL_{cohort_name}.tsv")
        if not os.path.exists(filepath):
            print(f"File mancante per {cohort_name}, saltato.")
            continue
        df = pd.read_csv(filepath, sep="\t")
        col_cluster = "Cluster_ID" if "Cluster_ID" in df.columns else "Cluster"
        if col_cluster not in df.columns:
            continue
        df = df.sort_values("Degree_Centrality", ascending=False).drop_duplicates("Gene", keep="first")
        df = df[["Gene", col_cluster, "Degree_Centrality"]].rename(columns={
            "Degree_Centrality": f"Centr_{cohort_name}",
            col_cluster:         f"ClusterID_{cohort_name}",
        })
        dfs.append(df)

    if len(dfs) < 2:
        print("File insufficienti, impossibile calcolare Super-Hub.")
        return

    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="Gene", how="inner")

    centrality_cols = [c for c in merged.columns if c.startswith("Centr_")]
    n = len(centrality_cols)
    merged["Min_Centrality"] = merged[centrality_cols].min(axis=1)
    merged["Geometric_Mean"] = np.prod(merged[centrality_cols].values, axis=1) ** (1.0 / n)
    merged = merged.sort_values(["Min_Centrality", "Geometric_Mean"], ascending=[False, False])

    out_file = os.path.join(output_dir, "PanCancer_SuperHubs_Robust.tsv")
    merged.to_csv(out_file, sep="\t", index=False)
    print(f"✅ Salvati {len(merged)} Super-Hub in: {out_file}")
    if not merged.empty:
        print("TOP 3 SUPER-HUBS:")
        print(merged[["Gene", "Min_Centrality", "Geometric_Mean"]].head(3).to_string(index=False))
    print("-" * 60)


def find_consensus_clusters(
    paths: list,
    output_dir: str,
    min_overlap: int = 3,
) -> None:
    """
    Cerca moduli consensuati (geni condivisi fra cluster di coorti diverse).
    Salva il TSV in output_dir. Funziona con N coorti (non solo 3).

    Parameters
    ----------
    paths : list[str]
        Lista di path delle coorti.
    output_dir : str
        Cartella di output per i risultati pan-cancer.
    min_overlap : int
        Numero minimo di geni condivisi per definire un modulo consensuato.
    """
    from itertools import product as _product
    os.makedirs(output_dir, exist_ok=True)
    print("Ricerca Consensus Modules...")

    cluster_data: dict = {}
    for path in paths:
        cohort_name = _cohort_name(path)
        filepath = os.path.join(path, "intracluster", f"Intracluster_Centrality_FULL_{cohort_name}.tsv")
        if not os.path.exists(filepath):
            print(f"File mancante per {cohort_name}, saltato.")
            continue
        df = pd.read_csv(filepath, sep="\t")
        col_cluster = "Cluster_ID" if "Cluster_ID" in df.columns else "Cluster"
        if col_cluster in df.columns:
            cluster_data[cohort_name] = df.groupby(col_cluster)["Gene"].apply(set).to_dict()

    if len(cluster_data) < 2:
        print("Dati insufficienti per il confronto.")
        return

    tissue_names = list(cluster_data.keys())
    all_clusters = [list(cluster_data[t].items()) for t in tissue_names]

    def _intersect_all(gene_sets):
        result = gene_sets[0]
        for s in gene_sets[1:]:
            result = result & s
        return result

    consensus_list = []
    for combo in _product(*all_clusters):
        gene_sets     = [genes for _, genes in combo]
        final_overlap = _intersect_all(gene_sets)
        if len(final_overlap) >= min_overlap:
            entry = {
                "Consensus_ID": f"M_{len(consensus_list)}",
                "Size":         len(final_overlap),
                "Genes":        ";".join(sorted(final_overlap)),
            }
            for tissue, (cid, _) in zip(tissue_names, combo):
                entry[f"ClusterID_{tissue}"] = cid
            consensus_list.append(entry)

    consensus_df = pd.DataFrame(consensus_list).sort_values("Size", ascending=False)
    out_file = os.path.join(output_dir, "PanCancer_Consensus_Modules.tsv")
    consensus_df.to_csv(out_file, sep="\t", index=False)
    print(f"✅ Trovati {len(consensus_df)} Moduli Consensuati in: {out_file}")
    if not consensus_df.empty:
        print(f"Modulo piu' grande: {consensus_df.iloc[0]['Size']} geni condivisi.")
    print("-" * 60)


def run_all_aggregations(
    paths: list,
    output_dir: str,
) -> None:
    """
    Aggrega i risultati di enrichment per cluster usando la gerarchia nativa
    KEGG (via API) e GO (via goatools). Salva i TSV in output_dir.

    Parameters
    ----------
    paths : list[str]
        Lista di path delle coorti.
    output_dir : str
        Cartella di output per i risultati pan-cancer.
    """
    import requests
    import re
    try:
        from goatools.obo_parser import GODag
        _GOATOOLS = True
    except ImportError:
        _GOATOOLS = False
        print("goatools non disponibile. L'aggregazione GO sara' limitata.")

    os.makedirs(output_dir, exist_ok=True)

    # Setup KEGG
    print("Connessione alle API KEGG...")
    kegg_mapping: dict = {}
    try:
        kegg_json = requests.get("http://rest.kegg.jp/get/br:hsa00001/json").json()

        def _parse_kegg(node, depth, current_sub):
            name      = node.get("name", "")
            children  = node.get("children", [])
            clean     = re.sub(r"^[\d\w]+\s+", "", name).split(" [PATH")[0].strip()
            if depth == 2:
                current_sub = clean
            elif depth == 3:
                kegg_mapping[clean.lower()] = f"KEGG: {current_sub}"
            for child in children:
                _parse_kegg(child, depth + 1, current_sub)

        _parse_kegg(kegg_json, 0, "")
        print(f"✅ Gerarchia KEGG: {len(kegg_mapping)} pathway.")
    except Exception as e:
        print(f"Errore API KEGG: {e}")

    # Setup GO DAG
    go_dag = None
    if _GOATOOLS:
        obo_file = os.path.join(output_dir, "go-basic.obo")
        if not os.path.exists(obo_file):
            with open(obo_file, "wb") as f:
                f.write(requests.get("http://purl.obolibrary.org/obo/go/go-basic.obo").content)
        go_dag = GODag(obo_file)
        print("✅ DAG GO costruito.")

    GO_BLACKLIST = {
        "biological_process", "cellular process", "regulation of biological process",
        "regulation of cellular process", "metabolic process", "protein binding",
        "binding", "catalytic activity", "cellular_component", "molecular_function",
        "system development", "anatomical structure development",
        "multicellular organismal process", "regulation of response to stimulus",
        "response to stimulus", "cellular component organization",
    }

    def _get_macro_sector(term: str, db_source: str) -> str:
        clean = re.sub(r"\s*-\s*Homo sapiens.*$", "", str(term), flags=re.IGNORECASE).strip().lower()
        if "KEGG" in db_source:
            if clean in kegg_mapping:
                return kegg_mapping[clean]
            for kp, ms in kegg_mapping.items():
                if kp in clean or clean in kp:
                    return ms
            return "KEGG: Unclassified / Specific Disease"
        elif "GO_" in db_source:
            if go_dag:
                m = re.search(r"(GO:\d{7})", str(term))
                if m and m.group(1) in go_dag:
                    paths_top = go_dag.paths_to_top(m.group(1))
                    if paths_top:
                        pfr = sorted(paths_top, key=len)[0][::-1]
                        macro = pfr[-1]
                        for node in pfr[1:]:
                            if node.name.lower() not in GO_BLACKLIST:
                                macro = node
                                break
                        return f"GO: {macro.name.capitalize()}"
            return f"GO: {clean.capitalize()} (No ID match)"
        elif "Reactome" in db_source:
            return f"Reactome: {re.sub(r'R-HSA-[0-9]+', '', str(term)).strip().capitalize()}"
        return "Other Database"

    def _process_enrichment_file(filepath, out_prefix, group_col, cluster_genes_map):
        if not os.path.exists(filepath):
            return
        df = pd.read_csv(filepath, sep="\t")
        if df.empty or group_col not in df.columns:
            return

        df["Macro_Sector"] = df.apply(lambda r: _get_macro_sector(r["Term"], r["Gene_set"]), axis=1)
        df["Database"]     = df["Macro_Sector"].apply(
            lambda x: "KEGG" if x.startswith("KEGG") else ("GO" if x.startswith("GO") else "Reactome")
        )

        all_macro_stats = []
        for group_id in sorted(df[group_col].unique()):
            group_df     = df[df[group_col] == group_id].copy()
            cluster_genes = cluster_genes_map.get(group_id, [])
            total_genes   = len(cluster_genes) if cluster_genes else None

            for sector in group_df["Macro_Sector"].unique():
                sector_df    = group_df[group_df["Macro_Sector"] == sector].copy()
                all_genes    = set()
                for genes_str in sector_df["Genes"].dropna():
                    all_genes.update(str(genes_str).split(";"))
                unique_count = len(all_genes)
                coverage_pct = (unique_count / total_genes * 100) if total_genes else None
                all_macro_stats.append({
                    group_col:               group_id,
                    "Database":              sector_df["Database"].iloc[0],
                    "Native_Macro_Sector":   sector,
                    "N_Pathways_Aggregated": len(sector_df),
                    "Best_P-value":          sector_df["P-value"].min(),
                    "Unique_Genes_Count":    unique_count,
                    "Coverage_%":            round(coverage_pct, 2) if coverage_pct else "N/A",
                    "Total_Input_Genes":     total_genes if total_genes else "N/A",
                    "Original_Pathways":     " | ".join(sector_df["Term"].tolist()),
                    "All_Unique_Genes":      ";".join(sorted(all_genes)),
                })

        meta_df = pd.DataFrame(all_macro_stats)
        if meta_df.empty:
            return
        meta_df = meta_df[
            ~meta_df["Native_Macro_Sector"].str.contains("Unclassified") |
            (meta_df["N_Pathways_Aggregated"] > 3)
        ]
        meta_df = meta_df.sort_values(
            by=[group_col, "Database", "N_Pathways_Aggregated", "Best_P-value"],
            ascending=[True, True, False, True],
        )
        meta_df.to_csv(os.path.join(output_dir, f"NativeDB_{out_prefix}_FULL.tsv"), sep="\t", index=False)
        top5_df = meta_df.groupby([group_col, "Database"]).head(5)
        top5_df.to_csv(os.path.join(output_dir, f"Top5_NativeDB_{out_prefix}.tsv"), sep="\t", index=False)

        print(f"\n--- TOP 5 {out_prefix.upper()} ---")
        for group_id in top5_df[group_col].unique():
            print(f"\n{group_col}: {group_id}")
            if cluster_genes_map and group_id in cluster_genes_map:
                print(f"   GENI DEL CLUSTER ({len(cluster_genes_map[group_id])}): "
                      f"{', '.join(sorted(cluster_genes_map[group_id]))}")
            group_top5 = top5_df[top5_df[group_col] == group_id]
            for db in ["GO", "KEGG", "Reactome"]:
                db_df = group_top5[group_top5["Database"] == db]
                if not db_df.empty:
                    print(f"  [{db}]")
                    for _, row in db_df.iterrows():
                        pct = f" ({row['Coverage_%']}%)" if row["Coverage_%"] != "N/A" else ""
                        print(f"   - {row['Native_Macro_Sector']} "
                              f"(P-value: {row['Best_P-value']:.2e}) -> "
                              f"{row['Unique_Genes_Count']} geni unici{pct}")
                        print(f"      Geni: {row['All_Unique_Genes'].replace(';', ', ')}")
        print("=" * 80)

    # Esecuzione per ogni coorte
    print("\nAvvio Aggregazione Stratificata per Clusters...")
    for path in paths:
        cohort_name = _cohort_name(path)
        print(f"\n\n{'#'*60}\n# COORTE: {cohort_name.upper()}\n{'#'*60}")

        cluster_file = os.path.join(path, "networks", f"Cluster_Genes_FULL_{cohort_name}.tsv")
        cluster_genes_map: dict = {}
        col_c = "Cluster_ID"
        if os.path.exists(cluster_file):
            df_c  = pd.read_csv(cluster_file, sep="\t")
            col_c = "Cluster_ID" if "Cluster_ID" in df_c.columns else "Cluster"
            cluster_genes_map = df_c.groupby(col_c)["Gene"].apply(lambda x: list(set(x))).to_dict()

        enrich_file = os.path.join(
            path, "networks", "enrichment", f"Enrichment_Clusters_FULL_{cohort_name}.tsv"
        )
        _process_enrichment_file(enrich_file, f"Clusters_{cohort_name}", col_c, cluster_genes_map)