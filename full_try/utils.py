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

import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.community as nx_comm
import numpy as np
import pandas as pd
import seaborn as sns
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
    print("=" * 60)

    mut_file = os.path.join(path, "F_data_mutations.txt")
    if not os.path.exists(mut_file):
        print(f"[!] File {mut_file} non trovato. Salto.")
        return

    df_mut = pd.read_csv(mut_file, sep="\t")

    functional = [
        "Missense_Mutation",
        "Nonsense_Mutation",
        "Frame_Shift_Del",
        "Frame_Shift_Ins",
        "In_Frame_Del",
        "In_Frame_Ins",
        "Splice_Site",
    ]

    # --- Statistiche "prima" (baseline) ---
    df_mut_func_all = df_mut[df_mut["Variant_Classification"].isin(functional)]
    pazienti_totali_iniziali = df_mut["Sample_Id"].nunique()
    muts_per_patient_before = df_mut_func_all.groupby("Sample_Id").size()
    tutti_i_pazienti_iniziali = pd.Series(0, index=df_mut["Sample_Id"].unique())
    muts_per_patient_before = muts_per_patient_before.combine_first(
        tutti_i_pazienti_iniziali
    )

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
                stable_samples = df_clin[df_clin["MSI Type"] == "Stable"][
                    "Sample_Id"
                ].tolist()
                df_mut_filtered = df_mut[df_mut["Sample_Id"].isin(stable_samples)]
                pazienti_rimasti = df_mut_filtered["Sample_Id"].nunique()

                df_mut_func_filtered = df_mut_filtered[
                    df_mut_filtered["Variant_Classification"].isin(functional)
                ]
                muts_per_patient_after = df_mut_func_filtered.groupby(
                    "Sample_Id"
                ).size()
                pazienti_validi_series = pd.Series(
                    0, index=df_mut_filtered["Sample_Id"].unique()
                )
                muts_per_patient_after = muts_per_patient_after.combine_first(
                    pazienti_validi_series
                )

                print("\n⚖️ --- IMPATTO DEL FILTRO MSI (MSS vs ALL) ---")
                print(
                    f"  • Pazienti:        {pazienti_totali_iniziali} -> {pazienti_rimasti} "
                    f"(Rimossi {pazienti_totali_iniziali - pazienti_rimasti} pazienti ipermutati)"
                )
                print(
                    f"  • Tot. Mutazioni:  {int(muts_per_patient_before.sum())} -> "
                    f"{int(muts_per_patient_after.sum())}"
                )
                print(
                    f"  • Media Mut/Paz:   {muts_per_patient_before.mean():.1f} -> "
                    f"{muts_per_patient_after.mean():.1f}"
                )
                print(
                    f"  • Mediana Mut/Paz: {muts_per_patient_before.median():.1f} -> "
                    f"{muts_per_patient_after.median():.1f}"
                )
                print(
                    f"  • MAX Mut/Paz:     {int(muts_per_patient_before.max())} -> "
                    f"{int(muts_per_patient_after.max())}  <-- Il dato che sbroglia il gomitolo!"
                )
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
    print(
        f"  • Densità matrice: "
        f"{(binary_matrix.sum().sum() / (num_samples * num_genes)) * 100:.2f}%"
    )

    top_genes = events_per_gene.sort_values(ascending=False).head(5)
    print("\n  🔥 Top 5 Geni Driver (SNV):")
    for g, count in top_genes.items():
        print(f"    - {g}: {count} pazienti ({(count/num_samples)*100:.1f}%)")
    print("====================================================\n")

    co_occ_matrix = binary_matrix.T.dot(binary_matrix)

    out_dir = _out(path, "matrices")
    binary_matrix.to_csv(os.path.join(out_dir, f"M_binary_{cohort_name}.tsv"), sep="\t")
    co_occ_matrix.to_csv(
        os.path.join(out_dir, f"M_cooccurrence_{cohort_name}.tsv"), sep="\t"
    )
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
        print(
            f"[!] File binario non trovato: {bin_file}. Eseguire prima generate_matrices()."
        )
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

        full_results.append(
            {
                "Gene_A": g1,
                "Gene_B": g2,
                "Co_Occurrence_Count": a,
                "P_Value": p_value,
                "Log2OR": log2or,
            }
        )

    if full_results:
        df_full = pd.DataFrame(full_results)
        df_full["P_Adj"] = multipletests(df_full["P_Value"], method="fdr_bh")[1]
        df_full["Log2OR"] = df_full["Log2OR"].replace([np.inf, -np.inf], [10.0, -10.0])
        df_full.to_csv(
            os.path.join(out_dir, f"Full_Cooccurrence_Stats_{cohort_name}.tsv"),
            sep="\t",
            index=False,
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

        both_mut = ((target_mut == 1) & (gene_mut == 1)).sum()
        only_target = ((target_mut == 1) & (gene_mut == 0)).sum()
        only_gene = ((target_mut == 0) & (gene_mut == 1)).sum()
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

        me_results.append(
            {
                "Gene_A": target_gene,
                "Gene_B": gene,
                "Co_Occurrence_Count": both_mut,
                "Only_KRAS": only_target,
                "Only_Gene_B": only_gene,
                "Neither": neither_mut,
                "P_Value": p_value,
                "Log2OR": log2or,
            }
        )

    if me_results:
        df_me = pd.DataFrame(me_results)
        df_me["P_Adj"] = multipletests(df_me["P_Value"], method="fdr_bh")[1]
        df_me = df_me.sort_values("P_Value")
        df_me.to_csv(
            os.path.join(out_dir, f"Mutual_Exclusivity_Stats_{cohort_name}.tsv"),
            sep="\t",
            index=False,
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

    out_dir = _out(path, "plots")
    stats_dir = os.path.join(path, "stats")

    # --- Plot Co-occorrenze ---
    full_file = os.path.join(stats_dir, f"Full_Cooccurrence_Stats_{cohort_name}.tsv")
    if os.path.exists(full_file):
        df = pd.read_csv(full_file, sep="\t")
        df_kras = df[
            (df["Gene_A"] == target_gene) | (df["Gene_B"] == target_gene)
        ].copy()

        if not df_kras.empty:
            p_thresh = coocc_params["p_val"]
            log_thresh = coocc_params["log2or"]

            df_kras["Partner"] = df_kras.apply(
                lambda row: (
                    row["Gene_B"] if row["Gene_A"] == target_gene else row["Gene_A"]
                ),
                axis=1,
            )
            df_kras["neg_log10_p_adj"] = -np.log10(df_kras["P_Adj"] + 1e-300)
            cond_sig = (df_kras["Log2OR"] >= log_thresh) & (
                df_kras["P_Adj"] <= p_thresh
            )
            df_kras["Stato"] = np.where(cond_sig, "Significativo", "Non Significativo")

            plt.figure(figsize=(12, 7))
            sns.scatterplot(
                data=df_kras,
                x="Log2OR",
                y="neg_log10_p_adj",
                hue="Stato",
                palette={"Significativo": "#d62728", "Non Significativo": "#b0b0b0"},
                alpha=0.7,
                s=60,
                edgecolor=None,
            )
            plt.axvline(
                x=log_thresh,
                color="red",
                linestyle="--",
                alpha=0.5,
                label=f"Soglia Log2OR (>= {log_thresh})",
            )
            plt.axhline(
                y=-np.log10(p_thresh),
                color="blue",
                linestyle="--",
                alpha=0.5,
                label=f"Soglia P-Adj (<= {p_thresh})",
            )

            for _, row in df_kras[cond_sig].iterrows():
                plt.text(
                    row["Log2OR"] + 0.05,
                    row["neg_log10_p_adj"] + 0.05,
                    row["Partner"],
                    fontsize=9,
                    weight="bold",
                    color="black",
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
                os.path.join(out_dir, f"Volcano_Cooccurrence_{cohort_name}.png"),
                dpi=150,
            )
            plt.show()
            print("✅ Volcano Plot Co-occorrenze (P-Adj) salvato e visualizzato.\n")

    # --- Plot Mutua Esclusione ---
    me_file = os.path.join(stats_dir, f"Mutual_Exclusivity_Stats_{cohort_name}.tsv")
    if os.path.exists(me_file):
        df_me = pd.read_csv(me_file, sep="\t")
        p_thresh_me = me_params["p_val"]
        log_thresh_me = me_params["log2or"]

        df_me["neg_log10_p_adj"] = -np.log10(df_me["P_Adj"] + 1e-300)
        cond_sig_me = (df_me["Log2OR"] <= log_thresh_me) & (
            df_me["P_Adj"] <= p_thresh_me
        )
        df_me["Stato"] = np.where(cond_sig_me, "Significativo", "Non Significativo")

        plt.figure(figsize=(12, 7))
        sns.scatterplot(
            data=df_me,
            x="Log2OR",
            y="neg_log10_p_adj",
            hue="Stato",
            palette={"Significativo": "#1f77b4", "Non Significativo": "#b0b0b0"},
            alpha=0.7,
            s=60,
            edgecolor=None,
        )
        plt.axvline(
            x=log_thresh_me,
            color="red",
            linestyle="--",
            alpha=0.5,
            label=f"Soglia Log2OR (<= {log_thresh_me})",
        )
        plt.axhline(
            y=-np.log10(p_thresh_me),
            color="blue",
            linestyle="--",
            alpha=0.5,
            label=f"Soglia P-Adj (<= {p_thresh_me})",
        )

        for _, row in df_me[cond_sig_me].iterrows():
            plt.text(
                row["Log2OR"] - 0.05,
                row["neg_log10_p_adj"] + 0.05,
                row["Gene_B"],
                fontsize=9,
                weight="bold",
                color="black",
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
            os.path.join(out_dir, f"Volcano_Mutual_Exclusivity_{cohort_name}.png"),
            dpi=150,
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
            import igraph as ig
            import leidenalg
            from sklearn.metrics import silhouette_score

            nodes = list(G.nodes())
            node_idx = {n: i for i, n in enumerate(nodes)}
            edges_ig = [(node_idx[u], node_idx[v]) for u, v in G.edges()]
            weights = [G[u][v].get("weight", 1) for u, v in G.edges()]

            g_ig = ig.Graph(n=len(nodes), edges=edges_ig, directed=False)
            g_ig.es["weight"] = weights
            g_ig.vs["name"] = nodes

            partition = leidenalg.find_partition(
                g_ig,
                leidenalg.ModularityVertexPartition,
                weights="weight",
                seed=42,
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
                    silhouette = silhouette_score(
                        dist_matrix, labels, metric="precomputed"
                    )
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
    cluster_data = [
        {"Cluster_ID": c_id, "Gene": node} for node, c_id in cluster_map.items()
    ]
    df_cluster = pd.DataFrame(cluster_data).sort_values(by=["Cluster_ID", "Gene"])

    net_dir = os.path.join(out_base, "networks")
    os.makedirs(net_dir, exist_ok=True)
    net_suffix = "FULL" if is_full else "FILTERED"
    cluster_path = os.path.join(
        net_dir, f"Cluster_Genes_{net_suffix}_{cohort_name}.tsv"
    )
    df_cluster.to_csv(cluster_path, sep="\t", index=False)
    print(f"📁 File cluster salvato in: {cluster_path}")

    # --- Grafico ---
    plt.figure(figsize=(18, 14) if is_full else (14, 10))
    pos = nx.spring_layout(G, k=0.5 if is_full else 0.9, iterations=50, seed=42)

    if target_gene in G.nodes():
        partner_nodes = [n for n in G.nodes() if n != target_gene]
        target_nodes = [target_gene]
    else:
        partner_nodes = list(G.nodes())
        target_nodes = []

    partner_colors = [cluster_map.get(n, 0) for n in partner_nodes]
    partner_sizes = [
        min(
            max(
                150,
                int(coocc_matrix.loc[n, n]) * 25 if n in coocc_matrix.index else 150,
            ),
            3000,
        )
        for n in partner_nodes
    ]

    edge_list = list(G.edges())
    weights = [max(1.0, np.log2(G[u][v]["weight"]) * 1.2) for u, v in edge_list]
    nx.draw_networkx_edges(G, pos, alpha=0.25, width=weights, edge_color="gray")

    if partner_nodes:
        vmax_val = max(max(partner_colors), 1) if partner_colors else 1
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=partner_nodes,
            node_color=partner_colors,
            cmap=plt.cm.tab20,
            vmin=0,
            vmax=vmax_val,
            node_size=partner_sizes,
            alpha=0.9,
            edgecolors="white",
            linewidths=1.5,
        )

    if target_nodes:
        target_tot = (
            int(coocc_matrix.loc[target_gene, target_gene])
            if target_gene in coocc_matrix.index
            else 50
        )
        target_size = min(max(800, target_tot * 25), 4000)
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=target_nodes,
            node_color="#d62728",
            node_shape="*",
            node_size=target_size,
            edgecolors="black",
            linewidths=1.5,
        )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=8 if is_full else 10,
        font_weight="bold",
        font_color="black",
    )

    net_type = "FULL" if is_full else f"FILTRATA ({target_gene})"
    plt.title(
        f"Rete Co-occorrenze {net_type} - {cohort_name.upper()}\n"
        f"Cluster identificati: {len(communities)}",
        fontsize=16,
        fontweight="bold",
    )

    if target_nodes:
        plt.plot(
            [],
            [],
            "*",
            color="#d62728",
            markersize=15,
            label=f"Target ({target_gene})",
            markeredgecolor="black",
        )
    plt.plot(
        [],
        [],
        "o",
        color="gray",
        markersize=10,
        label="Geni Partner (colore = cluster)",
    )
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

    stats_file = os.path.join(
        path, "stats", f"Full_Cooccurrence_Stats_{cohort_name}.tsv"
    )
    coocc_file = os.path.join(path, "matrices", f"M_cooccurrence_{cohort_name}.tsv")

    if not os.path.exists(stats_file) or not os.path.exists(coocc_file):
        print(
            f"[!] File mancanti per {cohort_name}. Eseguire prima le fasi precedenti."
        )
        return

    df_stats = pd.read_csv(stats_file, sep="\t")
    coocc_matrix = pd.read_csv(coocc_file, sep="\t", index_col=0)

    p_thresh = coocc_params["p_val"]
    log_thresh = coocc_params["log2or"]
    min_cooc = coocc_params["min_cooc"]

    valid_edges = df_stats[
        (df_stats["P_Value"] <= p_thresh)
        & (df_stats["Log2OR"] >= log_thresh)
        & (df_stats["Co_Occurrence_Count"] >= min_cooc)
    ]

    G_full = nx.Graph()
    for _, row in valid_edges.iterrows():
        G_full.add_edge(
            row["Gene_A"], row["Gene_B"], weight=int(row["Co_Occurrence_Count"])
        )

    if G_full.number_of_nodes() == 0:
        print(f"[-] Nessuna rete formata per {cohort_name}.")
        return

    # Rete FULL
    _draw_static_network(
        G_full, coocc_matrix, cohort_name, path, target_gene, is_full=True
    )

    # Rete FILTRATA (1-hop)
    if target_gene in G_full.nodes():
        neighbors = list(G_full.neighbors(target_gene))
        G_filtered = G_full.subgraph(neighbors + [target_gene]).copy()
        _draw_static_network(
            G_filtered, coocc_matrix, cohort_name, path, target_gene, is_full=False
        )
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

    stats_file = os.path.join(
        path, "stats", f"Full_Cooccurrence_Stats_{cohort_name}.tsv"
    )
    out_dir = _out(path, "stats")

    if not os.path.exists(stats_file):
        print(f"[!] File stats mancante per {cohort_name}.")
        return

    df_stats = pd.read_csv(stats_file, sep="\t")

    p_thresh = coocc_params["p_val"]
    log_thresh = coocc_params["log2or"]
    min_cooc = coocc_params["min_cooc"]

    valid_edges = df_stats[
        (df_stats["P_Value"] <= p_thresh)
        & (df_stats["Log2OR"] >= log_thresh)
        & (df_stats["Co_Occurrence_Count"] >= min_cooc)
    ]

    G_full = nx.Graph()
    for _, row in valid_edges.iterrows():
        G_full.add_edge(
            row["Gene_A"], row["Gene_B"], weight=int(row["Co_Occurrence_Count"])
        )

    if target_gene not in G_full.nodes():
        print(f"[-] {target_gene} non presente, calcolo metriche annullato.")
        return

    neighbors = list(G_full.neighbors(target_gene))
    G = G_full.subgraph(neighbors + [target_gene]).copy()

    if G.number_of_nodes() == 0:
        return

    degree_cent = nx.degree_centrality(G)
    betweenness_cent = nx.betweenness_centrality(G)
    closeness_cent = nx.closeness_centrality(G)
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
        sep="\t",
        index=False,
    )

    print("✅ Metriche salvate. Top 3 Hub identificati:")
    print(
        metrics_df[["Gene", "Degree", "Betweenness_Centrality"]]
        .head(3)
        .to_string(index=False)
    )


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

    global_cohort = _cohort_name(path_global)
    filtered_cohort = _cohort_name(path_filtered)
    print(
        f"\n--- 🔀 CONFRONTO RETI: {global_cohort.upper()} vs {filtered_cohort.upper()} ---"
    )

    stats_global = os.path.join(
        path_global, "stats", f"Full_Cooccurrence_Stats_{global_cohort}.tsv"
    )
    stats_filtered = os.path.join(
        path_filtered, "stats", f"Full_Cooccurrence_Stats_{filtered_cohort}.tsv"
    )

    if not os.path.exists(stats_global) or not os.path.exists(stats_filtered):
        print("[!] Dati mancanti per il confronto.")
        return

    df_g = pd.read_csv(stats_global, sep="\t", low_memory=False)
    df_f = pd.read_csv(stats_filtered, sep="\t", low_memory=False)

    def _filter(df: pd.DataFrame, params: dict) -> pd.DataFrame:
        return df[
            (df["P_Value"] <= params["p_val"])
            & (df["Log2OR"] >= params["log2or"])
            & (df["Co_Occurrence_Count"] >= params["min_cooc"])
        ]

    edges_g = _filter(df_g, coocc_params_global)
    edges_f = _filter(df_f, coocc_params_filtered)

    set_g = {tuple(sorted([r["Gene_A"], r["Gene_B"]])) for _, r in edges_g.iterrows()}
    set_f = {tuple(sorted([r["Gene_A"], r["Gene_B"]])) for _, r in edges_f.iterrows()}

    common = set_g & set_f
    unique_g = set_g - set_f
    unique_f = set_f - set_g

    print(f"Archi totali in {global_cohort.upper()}: {len(set_g)}")
    print(f"Archi totali in {filtered_cohort.upper()}: {len(set_f)}")
    print(f"Archi in comune: {len(common)}")
    print(f"Archi persi (presenti solo in {global_cohort.upper()}): {len(unique_g)}")
    print(
        f"Archi guadagnati (presenti solo in {filtered_cohort.upper()}): {len(unique_f)}\n"
    )

    if unique_f:
        print("Top 5 archi guadagnati (per Log2OR):")
        new_edges_df = edges_f[
            edges_f.apply(
                lambda r: tuple(sorted([r["Gene_A"], r["Gene_B"]])) in unique_f, axis=1
            )
        ]
        print(
            new_edges_df.sort_values(by="Log2OR", ascending=False)[
                ["Gene_A", "Gene_B", "Log2OR", "P_Adj"]
            ]
            .head()
            .to_string(index=False)
        )
        print()


import logging
import os
import time

import gseapy as gp
import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.community as nx_comm
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

try:
    import igraph as ig
    import leidenalg

    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False

try:
    import infomap as im_lib

    INFOMAP_AVAILABLE = True
except ImportError:
    INFOMAP_AVAILABLE = False

# =============================================================================
# INTRACLUSTER E HUB ANALYSIS
# =============================================================================


def analyze_intracluster_metrics(
    path: str,
    target_gene: str = "KRAS",
    is_full: bool = True,
    coocc_params: dict = None,
) -> None:
    """
    Calcola metriche topologiche dettagliate (densità, degree, transitivity, ecc.)
    per ogni singolo cluster identificato nella rete. Salva i risultati in TSV.

    Args:
        path (str): Cartella di output/input della coorte (es. './outputs_mut/kras_pancreas').
        target_gene (str): Gene target utilizzato per reti filtrate (default 'KRAS').
        is_full (bool): Se True analizza la rete FULL, altrimenti quella FILTERED.
        coocc_params (dict): Parametri di filtro archi (es. {'p_val':0.05, 'log2or':1.0, 'min_cooc':3}).
    """
    if coocc_params is None:
        coocc_params = {"p_val": 0.05, "log2or": 1.0, "min_cooc": 3}

    cohort_name = os.path.basename(os.path.normpath(path))
    net_type = "FULL" if is_full else "FILTERED"
    print(f"\n📊 METRICHE INTRACLUSTER: {cohort_name.upper()} ({net_type})")

    stats_file = os.path.join(
        path, "stats", f"Full_Cooccurrence_Stats_{cohort_name}.tsv"
    )
    cluster_file = os.path.join(
        path, "networks", f"Cluster_Genes_{net_type}_{cohort_name}.tsv"
    )

    if not os.path.exists(stats_file) or not os.path.exists(cluster_file):
        print(f"[!] File mancanti per l'analisi di {cohort_name} ({net_type}).")
        return

    df_stats = pd.read_csv(stats_file, sep="\t")

    valid_edges = df_stats[
        (df_stats["P_Value"] <= coocc_params["p_val"])
        & (df_stats["Log2OR"] >= coocc_params["log2or"])
        & (df_stats["Co_Occurrence_Count"] >= coocc_params["min_cooc"])
    ]

    G_full = nx.Graph()
    for _, row in valid_edges.iterrows():
        G_full.add_edge(
            row["Gene_A"], row["Gene_B"], weight=int(row["Co_Occurrence_Count"])
        )

    if is_full:
        G_work = G_full
    else:
        if target_gene in G_full.nodes():
            neighbors = list(G_full.neighbors(target_gene))
            G_work = G_full.subgraph(neighbors + [target_gene]).copy()
        else:
            print(f"[!] Target {target_gene} non presente nella rete.")
            return

    df_clusters = pd.read_csv(cluster_file, sep="\t")
    cluster_ids = sorted(df_clusters["Cluster_ID"].unique())

    metrics_list = []
    for c_id in cluster_ids:
        cluster_genes = df_clusters[df_clusters["Cluster_ID"] == c_id]["Gene"].tolist()
        valid_genes = [g for g in cluster_genes if g in G_work.nodes()]

        if len(valid_genes) < 2:
            continue

        subG = G_work.subgraph(valid_genes)

        n_nodes = subG.number_of_nodes()
        n_edges = subG.number_of_edges()

        density = nx.density(subG)
        avg_degree = (2 * n_edges) / n_nodes if n_nodes > 0 else 0
        avg_clustering = nx.average_clustering(subG, weight="weight")
        transitivity = nx.transitivity(subG)
        n_components = nx.number_connected_components(subG)

        avg_weight = (
            sum([d.get("weight", 1) for _, _, d in subG.edges(data=True)]) / n_edges
            if n_edges > 0
            else 0
        )

        lcc_nodes = max(nx.connected_components(subG), key=len)
        lcc = subG.subgraph(lcc_nodes)
        diameter = nx.diameter(lcc) if lcc.number_of_nodes() > 1 else 0

        metrics_list.append(
            {
                "Cluster_ID": c_id,
                "N_Nodes": n_nodes,
                "N_Edges": n_edges,
                "N_Components": n_components,
                "Density": round(density, 3),
                "Avg_Degree": round(avg_degree, 2),
                "Avg_Clustering": round(avg_clustering, 3),
                "Transitivity": round(transitivity, 3),
                "Diameter_LCC": diameter,
                "Avg_Edge_Weight": round(avg_weight, 2),
            }
        )

    if metrics_list:
        df_metrics = pd.DataFrame(metrics_list)
        out_file = os.path.join(
            path, "intracluster", f"Intracluster_Metrics_{net_type}_{cohort_name}.tsv"
        )
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        df_metrics.to_csv(out_file, sep="\t", index=False)
        print(f"✅ Metriche salvate in: {out_file}")
    else:
        print("[-] Nessun cluster valido per calcolare le metriche.")


def find_intracluster_hubs(
    path: str,
    target_gene: str = "KRAS",
    is_full: bool = True,
    coocc_params: dict = None,
) -> None:
    """
    Identifica gli hub locali per ogni cluster calcolando le centralità
    topologiche (degree, betweenness, closeness). Salva file TSV e report TXT.

    Args:
        path (str): Cartella di output/input della coorte.
        target_gene (str): Gene target per reti filtrate.
        is_full (bool): Se True analizza rete FULL.
        coocc_params (dict): Dizionario con i parametri di co-occorrenza.
    """
    if coocc_params is None:
        coocc_params = {"p_val": 0.05, "log2or": 1.0, "min_cooc": 3}

    cohort_name = os.path.basename(os.path.normpath(path))
    net_type = "FULL" if is_full else "FILTERED"
    print(f"\n🎯 RICERCA HUB INTRACLUSTER: {cohort_name.upper()} ({net_type})")

    base_dir = path
    stats_file = os.path.join(
        base_dir, "stats", f"Full_Cooccurrence_Stats_{cohort_name}.tsv"
    )
    cluster_file = os.path.join(
        base_dir, "networks", f"Cluster_Genes_{net_type}_{cohort_name}.tsv"
    )

    if not os.path.exists(stats_file) or not os.path.exists(cluster_file):
        print(f"[!] File mancanti per l'analisi di {cohort_name} ({net_type}).")
        return

    df_stats = pd.read_csv(stats_file, sep="\t")

    valid_edges = df_stats[
        (df_stats["P_Value"] <= coocc_params["p_val"])
        & (df_stats["Log2OR"] >= coocc_params["log2or"])
        & (df_stats["Co_Occurrence_Count"] >= coocc_params["min_cooc"])
    ]

    G_full = nx.Graph()
    for _, row in valid_edges.iterrows():
        G_full.add_edge(row["Gene_A"], row["Gene_B"])

    if is_full:
        G_work = G_full
    else:
        if target_gene in G_full.nodes():
            neighbors = list(G_full.neighbors(target_gene))
            G_work = G_full.subgraph(neighbors + [target_gene]).copy()
        else:
            return

    df_clusters = pd.read_csv(cluster_file, sep="\t")
    cluster_ids = sorted(df_clusters["Cluster_ID"].unique())
    all_nodes_data = []

    for c_id in cluster_ids:
        cluster_genes = df_clusters[df_clusters["Cluster_ID"] == c_id]["Gene"].tolist()
        valid_genes = [g for g in cluster_genes if g in G_work.nodes()]

        if len(valid_genes) < 3:
            continue

        subG = G_work.subgraph(valid_genes)
        deg_cent = nx.degree_centrality(subG)
        bet_cent = nx.betweenness_centrality(subG)
        clo_cent = nx.closeness_centrality(subG)

        for gene in valid_genes:
            all_nodes_data.append(
                {
                    "Cluster_ID": c_id,
                    "Gene": gene,
                    "Degree_Centrality": round(deg_cent[gene], 4),
                    "Betweenness_Centrality": round(bet_cent[gene], 4),
                    "Closeness_Centrality": round(clo_cent[gene], 4),
                }
            )

    if all_nodes_data:
        df_centrality = pd.DataFrame(all_nodes_data)
        df_centrality = df_centrality.sort_values(
            by=["Cluster_ID", "Degree_Centrality"], ascending=[True, False]
        )

        out_dir = os.path.join(base_dir, "intracluster")
        os.makedirs(out_dir, exist_ok=True)

        out_tsv = os.path.join(
            out_dir, f"Intracluster_Centrality_{net_type}_{cohort_name}.tsv"
        )
        df_centrality.to_csv(out_tsv, sep="\t", index=False)

        out_txt = os.path.join(
            out_dir, f"Report_Prof_TopHubs_{net_type}_{cohort_name}.txt"
        )
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(
                f"--- 👑 TOP 3 HUB GENES PER OGNI CLUSTER ({cohort_name.upper()} - {net_type}) ---\n"
            )
            for c_id in df_centrality["Cluster_ID"].unique():
                cluster_df = df_centrality[df_centrality["Cluster_ID"] == c_id]
                f.write(
                    f"\n{'-'*60}\n🎯 CLUSTER {c_id} | Totale geni nel network: {len(cluster_df)}\n{'-'*60}\n"
                )

                top_3 = cluster_df.head(3)
                for _, row in top_3.iterrows():
                    f.write(f"🔸 HUB: {row['Gene']}\n")
                    f.write(
                        f"   - Degree Centrality:      {row['Degree_Centrality']:.4f}\n"
                    )
                    f.write(
                        f"   - Betweenness Centrality: {row['Betweenness_Centrality']:.4f}\n"
                    )

        print(f"✅ Report Top Hubs salvato in: {out_txt}")
    else:
        print("[-] Impossibile calcolare centralità (cluster troppo piccoli).")


# =============================================================================
# ENRICHMENT ANALYSIS
# =============================================================================


def enrich_network_hubs(
    path: str, target_gene: str = "KRAS", is_full: bool = True
) -> None:
    """
    Esegue l'analisi di enrichment biologico (GSEA) sui top Hub globali della rete.

    Args:
        path (str): Cartella della coorte.
        target_gene (str): Gene target.
        is_full (bool): Rete FULL o FILTERED.
    """
    net_type = "FULL" if is_full else "FILTERED"
    cohort_name = os.path.basename(os.path.normpath(path))
    print(f"\n🌟 ENRICHMENT DEGLI HUB: {cohort_name.upper()} ({net_type})")

    enrichment_dir = os.path.join(path, "intracluster", "enrichment")
    os.makedirs(enrichment_dir, exist_ok=True)
    centrality_file = os.path.join(
        path, "intracluster", f"Intracluster_Centrality_{net_type}_{cohort_name}.tsv"
    )

    if not os.path.exists(centrality_file):
        print(f"[!] File centralità non trovato per {cohort_name} ({net_type}).")
        return

    df_centrality = pd.read_csv(centrality_file, sep="\t")

    hub_genes = []
    for c_id in df_centrality["Cluster_ID"].unique():
        cluster_df = df_centrality[df_centrality["Cluster_ID"] == c_id]
        hub_genes.extend(cluster_df.head(3)["Gene"].tolist())

    hub_genes = list(set(hub_genes))

    if len(hub_genes) < 2:
        print("[-] Troppi pochi Hub trovati.")
        return

    databases = ["KEGG_2021_Human", "GO_Biological_Process_2021", "Reactome_2022"]
    try:
        enr = gp.enrichr(
            gene_list=hub_genes, gene_sets=databases, organism="human", outdir=None
        )
        sig_res = enr.results[enr.results["P-value"] < 0.05].copy()
        time.sleep(1)
    except Exception as e:
        print(f"[!] Errore Enrichr: {e}")
        return

    if not sig_res.empty:
        sig_res = sig_res.sort_values(by=["P-value"], ascending=True)
        out_tsv = os.path.join(
            enrichment_dir, f"Enrichment_NetworkHubs_{net_type}_{cohort_name}.tsv"
        )
        sig_res.to_csv(out_tsv, sep="\t", index=False)
        print(f"✅ Analisi completata! Risultati salvati in: {out_tsv}")
    else:
        print("[-] Nessun pathway significativo trovato.")


def plot_hub_enrichment(
    path: str, target_gene: str = "KRAS", is_full: bool = True
) -> None:
    """
    Genera un barplot con i top 10 pathway più significativi per gli hub della rete.

    Args:
        path (str): Cartella della coorte.
        target_gene (str): Gene target.
        is_full (bool): Rete FULL o FILTERED.
    """
    net_type = "FULL" if is_full else "FILTERED"
    cohort_name = os.path.basename(os.path.normpath(path))

    enrichment_file = os.path.join(
        path,
        "intracluster",
        "enrichment",
        f"Enrichment_NetworkHubs_{net_type}_{cohort_name}.tsv",
    )
    plot_dir = os.path.join(path, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    if not os.path.exists(enrichment_file):
        return

    df_enr = pd.read_csv(enrichment_file, sep="\t")
    if df_enr.empty:
        return

    top_10 = df_enr.sort_values("P-value").head(10).copy()
    top_10["Minus_Log10_Pval"] = -np.log10(top_10["P-value"])
    top_10["Term"] = top_10["Term"].apply(
        lambda x: (x[:45] + "...") if len(x) > 45 else x
    )

    plt.figure(figsize=(10, 6))
    palette_color = (
        "Reds_r" if "FILTRATA" in net_type or "kras_" in cohort_name else "Blues_r"
    )

    sns.barplot(
        x="Minus_Log10_Pval",
        y="Term",
        data=top_10,
        palette=palette_color,
        hue="Term",
        legend=False,
    )

    plt.title(
        f"Top 10 Pathway degli Hub\n{cohort_name.upper()} - {net_type}",
        fontsize=14,
        pad=15,
        fontweight="bold",
    )
    plt.xlabel("-Log10(P-value)", fontsize=12)
    plt.ylabel("")
    plt.axvline(x=1.3, color="red", linestyle="--", alpha=0.7, label="P-value = 0.05")
    plt.legend(loc="lower right")
    plt.tight_layout()

    out_plot_png = os.path.join(
        plot_dir, f"Barplot_Hub_Enrichment_{net_type}_{cohort_name}.png"
    )
    plt.savefig(out_plot_png, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"✅ Grafico salvato in {out_plot_png}")


import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

# =============================================================================
# SURVIVAL ANALYSIS & STRATIFICATION
# =============================================================================


def load_mutation_matrix(path: str) -> pd.DataFrame:
    """
    Carica la matrice binaria M_binary (Righe: Sample_id, Colonne: Geni).

    Args:
        path (str): Cartella della coorte.

    Returns:
        pd.DataFrame: DataFrame contenente la matrice binaria delle mutazioni.
    """
    cohort_name = os.path.basename(os.path.normpath(path))
    path_mut = os.path.join(path, "matrices", f"M_binary_{cohort_name}.tsv")
    if os.path.exists(path_mut):
        return pd.read_csv(path_mut, sep="\t", index_col=0)
    else:
        print(f"[-] Matrice mutazionale non trovata: {path_mut}")
        return None


def plot_kaplan_meier(
    df_clinical: pd.DataFrame,
    group_col: str,
    time_col: str = "Overall Survival (Months)",
    status_col: str = "Overall Survival Status",
    ax=None,
) -> None:
    """
    Esegue l'analisi di Kaplan-Meier e calcola il modello di Cox per un gruppo specifico.

    Args:
        df_clinical (pd.DataFrame): DataFrame contenente i dati clinici.
        group_col (str): Colonna su cui stratificare (es. 'Sex', 'Cluster_Status').
        time_col (str): Colonna del tempo di sopravvivenza.
        status_col (str): Colonna dello stato dell'evento (es. '1:DECEASED').
        ax: Asse matplotlib (opzionale) su cui plottare.
    """
    df_clinical.columns = df_clinical.columns.str.strip()
    df_clean = df_clinical.dropna(subset=[time_col, status_col, group_col]).copy()
    df_clean["Time"] = pd.to_numeric(df_clean[time_col], errors="coerce")
    df_clean["Event"] = df_clean[status_col].apply(
        lambda x: 1 if "1:DECEASED" in str(x).upper() else 0
    )
    df_clean = df_clean.dropna(subset=["Time"])

    if df_clean.empty or len(df_clean[group_col].unique()) < 2:
        if ax:
            ax.text(0.5, 0.5, "Dati insufficienti\n(min. 2 gruppi)", ha="center")
        return

    kmf = KaplanMeierFitter()
    if ax is None:
        plt.figure(figsize=(8, 5))
        ax = plt.gca()

    groups = df_clean[group_col].unique()
    durations, events = [], []

    for group in groups:
        mask = df_clean[group_col] == group
        t, e = df_clean[mask]["Time"], df_clean[mask]["Event"]
        if len(t) > 5:
            kmf.fit(t, event_observed=e, label=f"{group} (n={len(t)})")
            kmf.plot_survival_function(ax=ax, ci_show=False)
            durations.append(t)
            events.append(e)

    ax.set_title(f"{group_col}", fontweight="bold")
    ax.set_xlabel("Mesi")
    ax.set_ylabel("Probabilità di Sopravvivenza")
    ax.grid(True, alpha=0.3)
    ax.legend(prop={"size": 9})

    if len(durations) == 2:
        res = logrank_test(
            durations[0],
            durations[1],
            event_observed_A=events[0],
            event_observed_B=events[1],
        )
        df_cox = df_clean[[group_col, "Time", "Event"]].copy()

        g1, g0 = groups[0], groups[1]
        df_cox["Group_Num"] = np.where(df_cox[group_col] == g1, 1, 0)

        cph = CoxPHFitter()
        try:
            cph.fit(
                df_cox[["Group_Num", "Time", "Event"]],
                duration_col="Time",
                event_col="Event",
            )
            hr = cph.summary.loc["Group_Num", "exp(coef)"]
            ci_lower = cph.summary.loc["Group_Num", "exp(coef) lower 95%"]
            ci_upper = cph.summary.loc["Group_Num", "exp(coef) upper 95%"]
            stats_text = (
                f"Log-rank p-value = {res.p_value:.4f}\n"
                f"HR ({g1} vs {g0}) = {hr:.2f}\n"
                f"95% CI: {ci_lower:.2f} - {ci_upper:.2f}"
            )
        except Exception:
            stats_text = f"Log-rank p-value = {res.p_value:.4f}"

        ax.annotate(
            stats_text,
            xy=(0.05, 0.05),
            xycoords="axes fraction",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9),
        )


def survival_by_gene_hub(
    df_clinical: pd.DataFrame, df_mut: pd.DataFrame, gene_name: str, ax=None
) -> None:
    """
    Incrocia i dati clinici con lo stato mutazionale di un gene hub specifico
    e genera il plot di Kaplan-Meier.

    Args:
        df_clinical (pd.DataFrame): DataFrame dati clinici.
        df_mut (pd.DataFrame): DataFrame matrice binaria mutazioni.
        gene_name (str): Nome del gene da testare.
        ax: Asse matplotlib opzionale.
    """
    df_plot = df_clinical.copy()
    df_plot["Sample_Id"] = df_plot["Sample_Id"].astype(str)

    if gene_name not in df_mut.columns:
        if ax:
            ax.text(0.5, 0.5, f"Gene {gene_name}\nnon in matrice", ha="center")
        return

    gene_status = df_mut[gene_name].to_dict()
    df_plot["Gene_Status"] = df_plot["Sample_Id"].map(gene_status)
    df_plot["Gene_Group"] = df_plot["Gene_Status"].map(
        {1.0: f"{gene_name} Mut", 0.0: f"{gene_name} WT"}
    )
    df_plot = df_plot.dropna(subset=["Gene_Group"])

    plot_kaplan_meier(df_plot, group_col="Gene_Group", ax=ax)


def find_significant_features(
    df_clinical: pd.DataFrame,
    features_list: list,
    time_col: str = "Overall Survival (Months)",
    status_col: str = "Overall Survival Status",
) -> None:
    """
    Scansiona una lista di feature cliniche e stampa un report per quelle
    che mostrano una sopravvivenza statisticamente significativa (p < 0.05).

    Args:
        df_clinical (pd.DataFrame): DataFrame dati clinici.
        features_list (list): Lista di colonne (feature) da testare.
        time_col (str): Colonna del tempo di sopravvivenza.
        status_col (str): Colonna dello stato dell'evento.
    """
    df_base = df_clinical.copy()
    df_base.columns = df_base.columns.str.strip()

    if time_col not in df_base.columns or status_col not in df_base.columns:
        print(f"   [!] Colonne di sopravvivenza non trovate.")
        return

    df_base["Time"] = pd.to_numeric(df_base[time_col], errors="coerce")
    df_base["Event"] = df_base[status_col].apply(
        lambda x: 1 if "1:DECEASED" in str(x).upper() else 0
    )
    df_base = df_base.dropna(subset=["Time"])

    significant_found = False

    for feature in features_list:
        if feature not in df_base.columns:
            continue

        df_feat = df_base.dropna(subset=[feature, "Time", "Event"]).copy()

        group_counts = df_feat[feature].value_counts()
        valid_groups = group_counts[group_counts >= 10].index
        df_feat = df_feat[df_feat[feature].isin(valid_groups)]

        groups = df_feat[feature].unique()

        if len(groups) == 2:
            g1, g0 = groups[0], groups[1]
            mask1, mask0 = (df_feat[feature] == g1), (df_feat[feature] == g0)
            t1, e1 = df_feat[mask1]["Time"], df_feat[mask1]["Event"]
            t0, e0 = df_feat[mask0]["Time"], df_feat[mask0]["Event"]

            res = logrank_test(t1, t0, event_observed_A=e1, event_observed_B=e0)

            if res.p_value < 0.05:
                significant_found = True
                df_cox = df_feat[[feature, "Time", "Event"]].copy()
                df_cox["Group_Num"] = np.where(df_cox[feature] == g1, 1, 0)

                try:
                    cph = CoxPHFitter()
                    cph.fit(
                        df_cox[["Group_Num", "Time", "Event"]],
                        duration_col="Time",
                        event_col="Event",
                    )
                    hr = cph.summary.loc["Group_Num", "exp(coef)"]
                    ci_l = cph.summary.loc["Group_Num", "exp(coef) lower 95%"]
                    ci_u = cph.summary.loc["Group_Num", "exp(coef) upper 95%"]
                    hr_text = f"HR = {hr:.2f} (95% CI: {ci_l:.2f} - {ci_u:.2f})"

                    if ci_l > 1.0:
                        interpretazione = (
                            "Rischio aumentato (Fattore prognostico negativo 🔴)"
                        )
                    elif ci_u < 1.0:
                        interpretazione = "Rischio ridotto (Fattore protettivo 🟢)"
                    else:
                        interpretazione = "Nessun effetto netto (Il CI include l'1 ⚪)"

                except Exception:
                    hr_text = "HR non calcolabile"
                    interpretazione = "Non valutabile"

                print(f"✅ FEATURE: {feature}")
                print(f"   ► P-value (Log-rank) : {res.p_value:.5f} ⭐")
                print(f"   ► Rischio (Cox)      : {hr_text} [{g1} vs {g0}]")
                print(f"   ► Effetto            : {interpretazione}")
                print(
                    f"   ► Distribuzione      : {g1} (n={len(t1)}) vs {g0} (n={len(t0)})"
                )
                print("-" * 70)

    if not significant_found:
        print("❌ Nessuna delle feature testate ha mostrato differenze significative.")


import os
import time

import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.community as nx_comm
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

try:
    import igraph as ig
    import leidenalg

    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False

try:
    import infomap as im_lib

    INFOMAP_AVAILABLE = True
except ImportError:
    INFOMAP_AVAILABLE = False

# =============================================================================
# CONFRONTO ALGORITMI DI CLUSTERING
# =============================================================================


def nx_to_igraph(G):
    """Converte un grafo NetworkX pesato in igraph (necessario per Leiden)."""
    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    edges = [(node_idx[u], node_idx[v]) for u, v in G.edges()]
    weights = [G[u][v].get("weight", 1) for u, v in G.edges()]
    g = ig.Graph(n=len(nodes), edges=edges, directed=False)
    g.es["weight"] = weights
    g.vs["name"] = nodes
    return g, nodes


def run_louvain(G):
    return nx_comm.louvain_communities(G, weight="weight", seed=42)


def run_leiden(G):
    if not LEIDEN_AVAILABLE:
        return None
    g_ig, nodes = nx_to_igraph(G)
    partition = leidenalg.find_partition(
        g_ig, leidenalg.ModularityVertexPartition, weights="weight", seed=42
    )
    return [set(nodes[i] for i in part) for part in partition]


def run_girvan_newman(G, max_communities=20):
    comp = nx_comm.girvan_newman(G)
    best_comms = None
    best_mod = -1
    for comms_tuple in comp:
        comms = list(comms_tuple)
        if len(comms) > max_communities:
            break
        mod = nx_comm.modularity(G, comms, weight="weight")
        if mod > best_mod:
            best_mod = mod
            best_comms = comms
    return [set(c) for c in best_comms] if best_comms else None


def run_infomap(G):
    if not INFOMAP_AVAILABLE:
        return None
    im = im_lib.Infomap(silent=True)
    node_list = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(node_list)}
    for u, v, d in G.edges(data=True):
        im.add_link(node_idx[u], node_idx[v], d.get("weight", 1))
    im.run()

    comm_map = {}
    for node in im.tree:
        if node.is_leaf:
            gene = node_list[node.node_id]
            comm_map[gene] = node.module_id

    from collections import defaultdict

    groups = defaultdict(set)
    for gene, mid in comm_map.items():
        groups[mid].add(gene)
    return list(groups.values())


def comms_to_labels(G, communities):
    label_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            label_map[node] = i
    return label_map


def labels_vector(G, label_map):
    return np.array([label_map.get(n, -1) for n in G.nodes()])


def compute_silhouette(G, label_map):
    nodes = list(G.nodes())
    labels = [label_map.get(n, -1) for n in nodes]
    n = len(nodes)

    dist = np.ones((n, n))
    node_idx = {nd: i for i, nd in enumerate(nodes)}
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1)
        dist_val = 1.0 / (w + 1e-9)
        dist[node_idx[u]][node_idx[v]] = dist_val
        dist[node_idx[v]][node_idx[u]] = dist_val
    np.fill_diagonal(dist, 0)

    if len(set(labels)) < 2:
        return np.nan
    try:
        return round(silhouette_score(dist, labels, metric="precomputed"), 4)
    except Exception:
        return np.nan


def compare_clustering_methods(path: str, coocc_params: dict = None) -> None:
    """
    Esegue e confronta diversi algoritmi di clustering (Louvain, Leiden,
    Girvan-Newman, Infomap) calcolandone modularity e silhouette score.
    """
    if coocc_params is None:
        coocc_params = {"p_val": 0.05, "log2or": 1.0, "min_cooc": 3}

    cohort_name = os.path.basename(os.path.normpath(path))
    print(f"\n  🔬 CONFRONTO CLUSTERING — {cohort_name.upper()}")

    stats_file = os.path.join(
        path, "stats", f"Full_Cooccurrence_Stats_{cohort_name}.tsv"
    )
    if not os.path.exists(stats_file):
        return

    df_stats = pd.read_csv(stats_file, sep="\t", low_memory=False)

    valid_edges = df_stats[
        (df_stats["P_Value"] <= coocc_params["p_val"])
        & (df_stats["Log2OR"] >= coocc_params["log2or"])
        & (df_stats["Co_Occurrence_Count"] >= coocc_params["min_cooc"])
    ]

    G = nx.Graph()
    for _, row in valid_edges.iterrows():
        G.add_edge(row["Gene_A"], row["Gene_B"], weight=int(row["Co_Occurrence_Count"]))

    if G.number_of_nodes() == 0:
        return

    methods = {
        "Louvain": run_louvain,
        "Leiden": run_leiden,
        "Girvan-Newman": lambda g: run_girvan_newman(g, max_communities=25),
        "Infomap": run_infomap,
    }

    results = {}
    louvain_vec = None

    for method_name, method_fn in methods.items():
        try:
            t0 = time.time()
            comms = method_fn(G)
            elapsed = time.time() - t0
        except Exception:
            results[method_name] = None
            continue

        if comms is None:
            results[method_name] = None
            continue

        label_map = comms_to_labels(G, comms)
        vec = labels_vector(G, label_map)
        modularity = nx_comm.modularity(G, comms, weight="weight")
        sil = compute_silhouette(G, label_map)

        if louvain_vec is None and method_name == "Louvain":
            louvain_vec = vec

        results[method_name] = {
            "n_communities": len(comms),
            "modularity": round(modularity, 4),
            "silhouette": sil,
            "time_s": round(elapsed, 2),
        }
        print(
            f"✓ {method_name}: Q={modularity:.3f} k={len(comms)} sil={sil} ({elapsed:.1f}s)"
        )

    rows = []
    for m, r in results.items():
        if r is None:
            continue
        rows.append(
            {
                "Metodo": m,
                "K": r["n_communities"],
                "Modularity": r["modularity"],
                "Silhouette": r["silhouette"],
                "Tempo (s)": r["time_s"],
            }
        )

    if rows:
        df_res = pd.DataFrame(rows)
        out_file = os.path.join(
            path, "intracluster", f"Clustering_Comparison_{cohort_name}.tsv"
        )
        df_res.to_csv(out_file, sep="\t", index=False)
        print(f"✅ Tabella confronto salvata in: {out_file}")


# =============================================================================
# ANALISI PANCANCER (SUPERHUBS E CONSENSUS CLUSTERS)
# =============================================================================


def compute_pancancer_superhubs_robust(
    cohorts_paths: list, output_dir: str = "./outputs_mut"
) -> None:
    """
    Calcola i 'Superhub' Pancancer aggregando le centralità (Rank Aggregation)
    da diverse coorti e salvando un report.

    Args:
        cohorts_paths (list): Lista dei percorsi delle cartelle delle coorti.
        output_dir (str): Cartella base per il salvataggio dei risultati pancancer.
    """
    print(f"\n" + "=" * 80)
    print("🌍 CALCOLO PANCANCER SUPERHUBS ROBUSTO (Rank Aggregation)")
    print("=" * 80)

    pan_dir = os.path.join(output_dir, "PANCANCER", "superhubs")
    os.makedirs(pan_dir, exist_ok=True)

    dfs = []
    for path in cohorts_paths:
        cohort_name = os.path.basename(os.path.normpath(path))
        # Proviamo a usare la FULL network
        cent_file = os.path.join(
            path, "intracluster", f"Intracluster_Centrality_FULL_{cohort_name}.tsv"
        )
        if not os.path.exists(cent_file):
            cent_file = os.path.join(
                path,
                "intracluster",
                f"Intracluster_Centrality_FILTERED_{cohort_name}.tsv",
            )
            if not os.path.exists(cent_file):
                continue

        df = pd.read_csv(cent_file, sep="\t")
        agg_df = (
            df.groupby("Gene")
            .agg({"Degree_Centrality": "max", "Betweenness_Centrality": "max"})
            .reset_index()
        )

        agg_df["Cohort"] = cohort_name
        agg_df["Degree_Rank"] = agg_df["Degree_Centrality"].rank(ascending=False)
        agg_df["Betw_Rank"] = agg_df["Betweenness_Centrality"].rank(ascending=False)
        dfs.append(agg_df)

    if not dfs:
        print("[-] Dati insufficienti per il pancancer.")
        return

    df_all = pd.concat(dfs, ignore_index=True)
    df_pivot = df_all.pivot(
        index="Gene", columns="Cohort", values=["Degree_Rank", "Betw_Rank"]
    )

    df_pivot.fillna(9999, inplace=True)
    df_pivot.columns = [f"{col[0]}_{col[1]}" for col in df_pivot.columns]

    deg_cols = [c for c in df_pivot.columns if "Degree" in c]
    betw_cols = [c for c in df_pivot.columns if "Betw" in c]

    df_pivot["Median_Degree_Rank"] = df_pivot[deg_cols].median(axis=1)
    df_pivot["Median_Betw_Rank"] = df_pivot[betw_cols].median(axis=1)
    df_pivot["Cohorts_Present"] = (df_pivot[deg_cols] < 9999).sum(axis=1)

    df_final = df_pivot.sort_values(
        by=["Cohorts_Present", "Median_Degree_Rank"], ascending=[False, True]
    ).reset_index()

    out_tsv = os.path.join(pan_dir, "Pancancer_Superhubs_Robust.tsv")
    df_final.to_csv(out_tsv, sep="\t", index=False)

    top_20 = df_final[df_final["Cohorts_Present"] >= 2].head(20)
    print(" Top 10 Pancancer Superhubs (presenti in più coorti, best rank mediano):")
    for _, row in top_20.head(10).iterrows():
        print(
            f"  🔸 {row['Gene']:<8} | Coorti: {row['Cohorts_Present']} | Med. Rank Deg: {row['Median_Degree_Rank']:.1f}"
        )
    print(f"✅ Report salvato in: {out_tsv}")


def find_consensus_clusters(
    cohorts_paths: list, output_dir: str = "./outputs_mut", min_overlap: int = 3
) -> None:
    """
    Cerca le 'comunità' pancancer trovando i set di geni che co-occorrono
    nello stesso cluster in diverse coorti.
    """
    print(f"\n" + "=" * 80)
    print("🌐 RICERCA CONSENSUS CLUSTERS (Moduli Pancancer)")
    print("=" * 80)

    pan_dir = os.path.join(output_dir, "PANCANCER", "consensus_clusters")
    os.makedirs(pan_dir, exist_ok=True)

    cluster_lists = []

    for path in cohorts_paths:
        cohort_name = os.path.basename(os.path.normpath(path))
        c_file = os.path.join(path, "networks", f"Cluster_Genes_FULL_{cohort_name}.tsv")
        if not os.path.exists(c_file):
            continue

        df = pd.read_csv(c_file, sep="\t")
        for c_id in df["Cluster_ID"].unique():
            genes = df[df["Cluster_ID"] == c_id]["Gene"].tolist()
            if len(genes) >= min_overlap:
                cluster_lists.append(
                    {"Cohort": cohort_name, "Cluster_ID": c_id, "Genes": set(genes)}
                )

    if not cluster_lists:
        return

    G_consensus = nx.Graph()
    for c_dict in cluster_lists:
        genes = list(c_dict["Genes"])
        import itertools

        for u, v in itertools.combinations(genes, 2):
            if G_consensus.has_edge(u, v):
                G_consensus[u][v]["weight"] += 1
            else:
                G_consensus.add_edge(u, v, weight=1)

    edges_to_keep = [
        (u, v) for u, v, d in G_consensus.edges(data=True) if d["weight"] >= 2
    ]
    G_filtered = G_consensus.edge_subgraph(edges_to_keep)

    components = list(nx.connected_components(G_filtered))
    consensus_clusters = [comp for comp in components if len(comp) >= min_overlap]

    res = []
    for i, comp in enumerate(consensus_clusters):
        res.append(
            {
                "Consensus_ID": i,
                "Size": len(comp),
                "Genes": ", ".join(sorted(list(comp))),
            }
        )

    if res:
        df_res = pd.DataFrame(res).sort_values(by="Size", ascending=False)
        out_file = os.path.join(pan_dir, "Pancancer_Consensus_Clusters.tsv")
        df_res.to_csv(out_file, sep="\t", index=False)
        print(f"✅ Trovati {len(res)} cluster di consenso. Salvati in: {out_file}")
    else:
        print("[-] Nessun consensus cluster trovato.")

# ---------------------------------------------------------------------------
# WRAPPER — chiamata unificata per tutte le analisi su una coorte
# ---------------------------------------------------------------------------

def run_cohort_analysis(
    path: str,
    target_gene: str = "KRAS",
    coocc_params: dict = None,
    me_params: dict = None,
    clinical_file: str = None,
) -> None:
    """
    Esegue l'intera pipeline di analisi su una singola coorte.
    Gestisce internamente il caricamento dei dati per le funzioni
    che non accettano (path, target_gene) come parametri standard.

    Parameters
    ----------
    path : str
        Cartella della coorte (es. "./strat/kras_FGA_Group_pancreas/Matched").
    target_gene : str
        Gene target (default "KRAS").
    coocc_params : dict, optional
        Soglie co-occorrenza. Default: p_val=0.05, log2or=1.0, min_cooc=3.
    me_params : dict, optional
        Soglie mutua esclusività. Default: p_val=0.01, log2or=-1.0.
    clinical_file : str, optional
        Nome del file clinico dentro `path` (es. "KRAS_F_pancreas.csv").
        Se None, lo cerca automaticamente cercando la colonna
        'Overall Survival (Months)'.
    """
    if coocc_params is None:
        coocc_params = {"p_val": 0.05, "log2or": 1.0, "min_cooc": 3}
    if me_params is None:
        me_params = {"p_val": 0.01, "log2or": -1.0}

    cohort_name = _cohort_name(path)
    print(f"\n{'='*60}")
    print(f"AVVIO PIPELINE: {cohort_name.upper()}")
    print(f"{'='*60}")

    # --- 1. Funzioni con firma standard (path, target_gene) ---
    generate_matrices(path, target_gene)
    calculate_statistics(path, target_gene, coocc_params, me_params)
    plot_volcanos(path, target_gene, coocc_params, me_params)
    build_all_networks(path, target_gene, coocc_params)
    calculate_metrics(path, target_gene, coocc_params)
    find_intracluster_hubs(path, target_gene)
    enrich_network_hubs(path, target_gene)

    # --- 2. Caricamento dati per le funzioni con firma non standard ---

    # Matrice mutazionale
    bin_file = os.path.join(path, "matrices", f"M_binary_{cohort_name}.tsv")
    if not os.path.exists(bin_file):
        print(f"[!] Matrice binaria non trovata, salto analisi survival. Atteso: {bin_file}")
        return
    df_mut = pd.read_csv(bin_file, sep="\t", index_col=0)

    # File clinico — usa quello passato, oppure lo cerca automaticamente
    df_clin = None
    if clinical_file:
        clin_path = os.path.join(path, clinical_file)
        if os.path.exists(clin_path):
            df_clin = pd.read_csv(clin_path, sep="\t", low_memory=False)
    
    if df_clin is None:
        # Ricerca automatica: primo file con colonna di sopravvivenza
        for fname in os.listdir(path):
            if not fname.endswith((".csv", ".tsv", ".txt")):
                continue
            try:
                tmp = pd.read_csv(os.path.join(path, fname), sep="\t", low_memory=False)
                if "Overall Survival (Months)" in tmp.columns:
                    df_clin = tmp
                    print(f"  [clinical] Trovato automaticamente: {fname}")
                    break
            except Exception:
                continue

    if df_clin is None:
        print("[!] File clinico non trovato, salto analisi survival.")
        return

    # --- 3. survival_by_gene_hub — firma: (df_clinical, df_mut, gene_name) ---
    print(f"\n--- ⏳ SURVIVAL ANALYSIS: {cohort_name.upper()} ---")
    
    # Recupera gli hub dal file di centralità per analizzarli tutti
    centrality_file = os.path.join(
        path, "intracluster", f"Intracluster_Centrality_FULL_{cohort_name}.tsv"
    )
    if not os.path.exists(centrality_file):
        centrality_file = os.path.join(
            path, "intracluster", f"Intracluster_Centrality_FILTERED_{cohort_name}.tsv"
        )

    hub_genes = [target_gene]
    if os.path.exists(centrality_file):
        df_cent = pd.read_csv(centrality_file, sep="\t")
        for c_id in df_cent["Cluster_ID"].unique():
            top3 = df_cent[df_cent["Cluster_ID"] == c_id].head(3)["Gene"].tolist()
            hub_genes.extend(top3)
        hub_genes = list(dict.fromkeys(hub_genes))  # dedup, preserva ordine

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n = len(hub_genes)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(7 * cols, 5 * rows))
    gs = gridspec.GridSpec(rows, cols, figure=fig)

    for idx, gene in enumerate(hub_genes):
        ax = fig.add_subplot(gs[idx // cols, idx % cols])
        survival_by_gene_hub(df_clin, df_mut, gene, ax=ax)

    fig.suptitle(f"Survival by Hub Gene — {cohort_name.upper()}", fontsize=16)
    plt.tight_layout()

    out_dir = _out(path, "plots")
    out_png = os.path.join(out_dir, f"Survival_Hubs_{cohort_name}.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Plot sopravvivenza salvato in: {out_png}")

# ---------------------------------------------------------------------------
# CONFRONTO TRA COORTI — Kaplan-Meier comparativo Matched vs Unmatched
# ---------------------------------------------------------------------------

def run_comparison(
    path_matched: str,
    path_unmatched: str,
    target_gene: str = "KRAS",
    clinical_file: str = None,
    coocc_params: dict = None,
) -> None:
    """
    Confronta le due coorti (Matched vs Unmatched) con:
      1. plot_kaplan_meier — sopravvivenza per gruppo
      2. find_significant_features — feature cliniche prognostiche
      3. compare_networks — archi guadagnati/persi tra le reti

    Parameters
    ----------
    path_matched : str
        Cartella della coorte Matched.
    path_unmatched : str
        Cartella della coorte Unmatched.
    target_gene : str
        Gene target (default "KRAS").
    clinical_file : str, optional
        Nome del file clinico (es. "KRAS_F_pancreas.csv").
        Se None lo cerca automaticamente.
    coocc_params : dict, optional
        Soglie co-occorrenza per compare_networks.
    """
    if coocc_params is None:
        coocc_params = {"p_val": 0.05, "log2or": 1.0, "min_cooc": 3}

    print(f"\n{'='*60}")
    print(f"CONFRONTO: {_cohort_name(path_matched).upper()} vs {_cohort_name(path_unmatched).upper()}")
    print(f"{'='*60}")

    # --- Caricamento file clinici da entrambe le coorti ---
    def _load_clinical(path):
        if clinical_file:
            fpath = os.path.join(path, clinical_file)
            if os.path.exists(fpath):
                return pd.read_csv(fpath, sep="\t", low_memory=False)
        for fname in os.listdir(path):
            if not fname.endswith((".csv", ".tsv", ".txt")):
                continue
            try:
                tmp = pd.read_csv(os.path.join(path, fname), sep="\t", low_memory=False)
                if "Overall Survival (Months)" in tmp.columns:
                    print(f"  [clinical] Trovato: {fname} in {_cohort_name(path)}")
                    return tmp
            except Exception:
                continue
        print(f"[!] File clinico non trovato in {path}")
        return None

    df_matched   = _load_clinical(path_matched)
    df_unmatched = _load_clinical(path_unmatched)

    # --- 1. Kaplan-Meier comparativo Matched vs Unmatched ---
    if df_matched is not None and df_unmatched is not None:
        print(f"\n--- 📈 KAPLAN-MEIER COMPARATIVO ---")

        # Aggiunge colonna gruppo e unisce i due dataframe
        df_matched["Cohort_Group"]   = f"Matched (n={len(df_matched)})"
        df_unmatched["Cohort_Group"] = f"Unmatched (n={len(df_unmatched)})"
        df_combined = pd.concat([df_matched, df_unmatched], ignore_index=True)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Matched vs Unmatched sulla sopravvivenza globale
        plot_kaplan_meier(df_combined, group_col="Cohort_Group", ax=axes[0])
        axes[0].set_title(
            f"Matched vs Unmatched\nSopravvivenza Globale",
            fontweight="bold"
        )

        # Plot 2: sopravvivenza per stato mutazionale del target gene nel combined
        df_mut_m  = load_mutation_matrix(path_matched)
        df_mut_um = load_mutation_matrix(path_unmatched)

        if df_mut_m is not None and df_mut_um is not None:
            # Ricostruisce lo stato mutazionale sul dataset combinato
            gene_status = {}
            for df_mut in [df_mut_m, df_mut_um]:
                if target_gene in df_mut.columns:
                    gene_status.update(df_mut[target_gene].to_dict())

            df_combined["Gene_Status"] = df_combined["Sample_Id"].astype(str).map(gene_status)
            df_combined["Gene_Group"]  = df_combined["Gene_Status"].map(
                {1.0: f"{target_gene} Mut", 0.0: f"{target_gene} WT"}
            )
            df_combined_gene = df_combined.dropna(subset=["Gene_Group"])
            plot_kaplan_meier(df_combined_gene, group_col="Gene_Group", ax=axes[1])
            axes[1].set_title(
                f"{target_gene} Mut vs WT\nSopravvivenza Globale (entrambe le coorti)",
                fontweight="bold"
            )
        else:
            axes[1].axis("off")

        plt.suptitle(
            f"Analisi Comparativa — {_cohort_name(path_matched)} vs {_cohort_name(path_unmatched)}",
            fontsize=15, y=1.02
        )
        plt.tight_layout()

        # Salva nella cartella parent comune
        out_dir = os.path.dirname(path_matched)
        os.makedirs(out_dir, exist_ok=True)
        out_png = os.path.join(out_dir, f"KM_Comparison_{target_gene}.png")
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"✅ Plot KM comparativo salvato in: {out_png}")

    # --- 2. Feature cliniche prognostiche su entrambe le coorti ---
    clinical_cols = [
        "Age_Group", "TMB_Group", "FGA_Group",
        "Sex", "Stage", "Cohort_Group"
    ]

    for label, df_clin in [("MATCHED", df_matched), ("UNMATCHED", df_unmatched)]:
        if df_clin is None:
            continue
        cols_presenti = [c for c in clinical_cols if c in df_clin.columns]
        if cols_presenti:
            print(f"\n--- 🔍 FEATURE SIGNIFICATIVE: {label} ---")
            find_significant_features(df_clin, cols_presenti)

    # --- 3. Confronto reti tra le due coorti ---
    print(f"\n--- 🔀 CONFRONTO RETI ---")
    compare_networks(path_matched, path_unmatched, coocc_params, coocc_params)