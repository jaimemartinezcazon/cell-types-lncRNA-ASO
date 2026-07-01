'''
Author: Jaime Martínez Cazón
Adapted for direct gene names, unparquet edge lists, and large-scale networks.

Description:
Performs a bow-tie decomposition analysis on a Gene Regulatory Network.
Calculates the size of the main bow-tie components (IN, OUT, SCC, and OTHERS)
and interface metrics. Compares the real network against an ensemble of
surrogate networks (null models) using multiprocessing for speed.

Additionally reports all secondary SCCs with >= MIN_SCC_SIZE nodes and
produces a bar plot of their sizes.

Outputs:
  - bow_tie_comparison_results.json   : full metrics for real + null models
  - bow_tie_plot_percentages.json     : data for the sector percentage bar plot
  - secondary_sccs.json               : sizes of secondary SCCs (real network)

IN / OUT / SCC DISJOINTNESS NOTE:
  These three sets are always mutually exclusive by construction.  A node in
  both IN and OUT would require a cycle through itself into the SCC, making it
  part of the SCC — a contradiction.  Therefore IN + SCC + OUT + OTHERS = N
  exactly, with no overlap and no gap.
'''

import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import multiprocessing as mp
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm

# =============================================================================
# SETUP: DIRECTORIES AND FILE PATHS
# =============================================================================
script_dir = Path(__file__).parent

# Saved data directory
INPUT_DATA_DIR = Path(script_dir / "../../data")

# Output directories
FIG_DIR = Path(script_dir / "figures")
OUTPUT_DATA_DIR = Path(script_dir / "data_output")

# Create directories if they do not exist
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# GRN edge list path
EDGE_LIST_PATH = INPUT_DATA_DIR / "edge_list_to_analyze.parquet"

NULL_MODELS_DIR     = INPUT_DATA_DIR / "null_models"
RESULTS_FILE        = OUTPUT_DATA_DIR / "bow_tie_comparison_results.json"
PLOT_DATA_FILE      = OUTPUT_DATA_DIR / "bow_tie_plot_percentages.json"
SECONDARY_SCC_FILE  = OUTPUT_DATA_DIR / "secondary_sccs.json"

NUM_CORES = max(1, mp.cpu_count() - 2)

# Minimum size for a secondary SCC to be considered non-trivial.
# Set to 3 because a cycle requires at least 3 nodes (triangles and above).
MIN_SCC_SIZE = 3

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def multi_source_shortest_path_length(G, sources):
    """
    BFS from a set of source nodes. Returns {node: distance} for all nodes
    reachable from any source.

      - Called with G      → distances FROM the SCC outward (toward OUT).
      - Called with G_rev  → distances TO the SCC inward   (from IN).

    nx.DiGraph.neighbors() returns only successors (outgoing edges), so
    passing G_rev correctly traverses edges backwards relative to G.
    """
    dist = {}
    queue = deque()
    for s in sources:
        if s in G:
            dist[s] = 0
            queue.append(s)
    while queue:
        u = queue.popleft()
        for v in G.neighbors(u):
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def get_all_sccs(G):
    """
    Returns all strongly connected components sorted by size (descending).
    Each entry is a frozenset of node names.
    """
    return sorted(
        (frozenset(c) for c in nx.strongly_connected_components(G)),
        key=len, reverse=True
    )


def calculate_bow_tie_metrics(G):
    """
    Bow-tie decomposition of a directed graph (Broder et al. 2000;
    Yang et al. IEEE 2011).

    Components (mutually exclusive, sum to N):
      SCC    — largest strongly connected component.
      IN     — nodes that can reach the SCC but are not in it.
      OUT    — nodes reachable from the SCC but not in it.
      OTHERS — everything else (tendrils, tubes as edges not nodes,
               disconnected sub-graphs).

    Key algorithm note:
      Because all SCC nodes are mutually reachable, the set of descendants
      of ANY single SCC member in G equals the descendants of the entire SCC.
      One nx.descendants call is therefore sufficient for each direction.

    Returns None if the graph has no SCC with >= 2 nodes.
    """
    N = G.number_of_nodes()
    if N == 0:
        return None

    sccs = get_all_sccs(G)
    if not sccs or len(sccs[0]) < 2:
        return None

    scc_nodes_set = set(sccs[0])

    # Deterministic representative node (sorted for reproducibility).
    rep_node = sorted(scc_nodes_set)[0]

    # OUT: nodes reachable from the SCC (excluding SCC itself).
    reachable_from_scc = nx.descendants(G, rep_node)
    out_sector_set = reachable_from_scc - scc_nodes_set

    # IN: nodes that can reach the SCC (excluding SCC itself).
    # copy=True: avoid a mutable view of G that could be corrupted.
    G_rev = G.reverse(copy=True)
    reachable_to_scc = nx.descendants(G_rev, rep_node)
    in_sector_set = reachable_to_scc - scc_nodes_set

    # OTHERS: everything not in IN, SCC, or OUT.
    # IN/SCC/OUT are provably disjoint (see module docstring).
    others_set = set(G.nodes()) - scc_nodes_set - in_sector_set - out_sector_set

    in_size     = len(in_sector_set)
    scc_size    = len(scc_nodes_set)
    out_size    = len(out_sector_set)
    others_size = len(others_set)

    assert in_size + scc_size + out_size + others_size == N, (
        f"Partition error: {in_size}+{scc_size}+{out_size}+{others_size} != {N}"
    )

    # Interface edge densities — both normalised by SOURCE sector size.
    #   ratio_in_scc  = edges(IN→SCC)  / |IN|   (out-edges per IN node)
    #   ratio_scc_out = edges(SCC→OUT) / |SCC|  (out-edges per SCC node)
    in_to_scc_edges = sum(
        1 for u in in_sector_set
        for v in G.successors(u) if v in scc_nodes_set
    )
    ratio_in_scc = in_to_scc_edges / in_size if in_size > 0 else 0.0

    scc_to_out_edges = sum(
        1 for u in scc_nodes_set
        for v in G.successors(u) if v in out_sector_set
    )
    ratio_scc_out = scc_to_out_edges / scc_size if scc_size > 0 else 0.0

    # Average shortest path lengths (multi-source BFS from entire SCC).
    dist_to_scc = multi_source_shortest_path_length(G_rev, scc_nodes_set)
    distances_in = [dist_to_scc[n] for n in in_sector_set if n in dist_to_scc]
    avg_dist_in_scc = np.mean(distances_in) if distances_in else float('nan')

    dist_from_scc = multi_source_shortest_path_length(G, scc_nodes_set)
    distances_out = [dist_from_scc[n] for n in out_sector_set if n in dist_from_scc]
    avg_dist_scc_out = np.mean(distances_out) if distances_out else float('nan')

    return {
        'N_nodes':              N,
        'L_edges':              G.number_of_edges(),
        'in_pct':               (in_size     / N) * 100,
        'scc_pct':              (scc_size    / N) * 100,
        'out_pct':              (out_size    / N) * 100,
        'others_pct':           (others_size / N) * 100,
        'ratio_in_scc':         ratio_in_scc,
        'ratio_scc_out':        ratio_scc_out,
        'average_dist_in_scc':  avg_dist_in_scc,
        'average_dist_scc_out': avg_dist_scc_out,
    }


def get_secondary_scc_sizes(G, min_size=MIN_SCC_SIZE):
    """
    Returns a sorted list (descending) of sizes for all SCCs with >= min_size
    nodes, excluding the largest SCC (the bow-tie core).

    These secondary SCCs represent self-regulatory sub-circuits — groups of
    genes that mutually regulate each other without belonging to the main
    regulatory core.  Any SCC >= 3 nodes contains at least one directed cycle
    (triangle or larger), which is why min_size=3 is the default threshold.

    Parameters
    ----------
    G        : nx.DiGraph
    min_size : int  — minimum number of nodes to include (default 3)

    Returns
    -------
    list[int]  — sizes of secondary SCCs, sorted descending.
                 Empty list if none exist above the threshold.
    """
    all_sccs = get_all_sccs(G)
    if len(all_sccs) < 2:
        return []
    # Skip index 0 (the main SCC / bow-tie core).
    secondary = [len(c) for c in all_sccs[1:] if len(c) >= min_size]
    return sorted(secondary, reverse=True)


def process_single_null_model(filepath):
    """
    Worker: loads one graphml file and returns bow-tie metrics.
    Returns None on failure so the caller can count skipped files.
    """
    try:
        G_null = nx.read_graphml(filepath, node_type=str)
        return calculate_bow_tie_metrics(G_null)
    except Exception:
        return None

# =============================================================================
# MAIN ANALYSIS SCRIPT
# =============================================================================

def main():
    start_time = time.time()

    # -------------------------------------------------------------------------
    # 1. Load pre-computed results or run full analysis
    # -------------------------------------------------------------------------
    if os.path.exists(RESULTS_FILE):
        print(f"Loading previously computed results from {RESULTS_FILE}")
        print("  (Delete this file to force recomputation.)")
        with open(RESULTS_FILE, 'r') as f:
            results_data = json.load(f)
        real_metrics      = results_data['real_metrics']
        null_mean_metrics = results_data['null_mean_metrics']
        null_std_metrics  = results_data['null_std_metrics']
        # Secondary SCCs are always re-derived from the real network if needed.
        G_real = None
    else:
        print("Loading real network...")
        edge_list = pd.read_parquet(EDGE_LIST_PATH)
        G_real = nx.from_pandas_edgelist(
            edge_list, source='source', target='target',
            create_using=nx.DiGraph()
        )
        G_real.remove_edges_from(nx.selfloop_edges(G_real))
        print(f"  Nodes: {G_real.number_of_nodes():,}   "
              f"Edges: {G_real.number_of_edges():,}")

        print("\nComputing bow-tie metrics for the real network...")
        real_metrics = calculate_bow_tie_metrics(G_real)
        if real_metrics is None:
            raise ValueError("Real network has no valid SCC — cannot proceed.")

        print(f"\nComputing metrics for null models using {NUM_CORES} cores...")
        null_files = sorted(
            os.path.join(NULL_MODELS_DIR, f)
            for f in os.listdir(NULL_MODELS_DIR)
            if f.endswith(".graphml")
        )
        null_metrics_list = []
        failed_count = 0

        with mp.Pool(processes=NUM_CORES) as pool:
            for res in tqdm(
                pool.imap_unordered(process_single_null_model, null_files),
                total=len(null_files)
            ):
                if res is not None:
                    null_metrics_list.append(res)
                else:
                    failed_count += 1

        if failed_count:
            print(f"  WARNING: {failed_count} null model file(s) failed and were skipped.")
        if not null_metrics_list:
            raise ValueError("No valid null model metrics were collected.")

        df_null           = pd.DataFrame(null_metrics_list)
        null_mean_metrics = df_null.mean().to_dict()
        null_std_metrics  = df_null.std().to_dict()

        print(f"\nSaving results to {RESULTS_FILE}...")
        with open(RESULTS_FILE, 'w') as f:
            json.dump({
                'real_metrics':      real_metrics,
                'null_mean_metrics': null_mean_metrics,
                'null_std_metrics':  null_std_metrics,
            }, f, indent=4)

    # -------------------------------------------------------------------------
    # 2. Secondary SCC analysis (real network only)
    # -------------------------------------------------------------------------
    # Load the real network if we came from cache and don't have it in memory.
    if G_real is None:
        print("\nLoading real network for secondary SCC analysis...")
        edge_list = pd.read_parquet(EDGE_LIST_PATH)
        G_real = nx.from_pandas_edgelist(
            edge_list, source='source', target='target',
            create_using=nx.DiGraph()
        )
        G_real.remove_edges_from(nx.selfloop_edges(G_real))

    print(f"\nSearching for secondary SCCs with >= {MIN_SCC_SIZE} nodes...")
    secondary_sizes = get_secondary_scc_sizes(G_real, min_size=MIN_SCC_SIZE)

    if secondary_sizes:
        print(f"  Found {len(secondary_sizes)} secondary SCC(s) above threshold.")
        print(f"  Sizes: {secondary_sizes[:10]}"
              + (" ..." if len(secondary_sizes) > 10 else ""))
    else:
        print(f"  No secondary SCCs with >= {MIN_SCC_SIZE} nodes found.")

    with open(SECONDARY_SCC_FILE, 'w') as f:
        json.dump({'secondary_scc_sizes': secondary_sizes,
                   'min_size_threshold': MIN_SCC_SIZE}, f, indent=4)
    print(f"  Secondary SCC data saved to {SECONDARY_SCC_FILE}")

    # -------------------------------------------------------------------------
    # 3. Print summary comparison table
    # -------------------------------------------------------------------------
    print("\n--- Bow-Tie Metrics Comparison ---")
    metrics_to_print = {
        'in_pct':               '% IN Nodes',
        'scc_pct':              '% SCC Nodes',
        'out_pct':              '% OUT Nodes',
        'others_pct':           '% OTHERS Nodes',
        'ratio_in_scc':         'Edges IN→SCC  / IN Node',
        'ratio_scc_out':        'Edges SCC→OUT / SCC Node',
        'average_dist_in_scc':  'Avg. Dist. IN → SCC',
        'average_dist_scc_out': 'Avg. Dist. SCC → OUT',
    }
    print(f"{'Metric':<35} | {'Real Network':<15} | {'Null Models (Mean ± Std)'}")
    print("-" * 85)
    for key, label in metrics_to_print.items():
        real_val  = real_metrics.get(key, float('nan'))
        null_mean = null_mean_metrics.get(key, float('nan'))
        null_std  = null_std_metrics.get(key, float('nan'))
        print(f"{label:<35} | {real_val:<15.4f} | {null_mean:.4f} ± {null_std:.4f}")
    print("-" * 85)

    # -------------------------------------------------------------------------
    # 4. Prepare and save plot data (format unchanged for sector % plot)
    # -------------------------------------------------------------------------
    plot_sectors = ['in_pct', 'scc_pct', 'out_pct', 'others_pct']
    plot_labels  = ['IN', 'SCC', 'OUT', 'Others']

    plot_data = {
        'labels':                plot_labels,
        'real_percentages':      [real_metrics.get(s, 0.0)      for s in plot_sectors],
        'null_mean_percentages': [null_mean_metrics.get(s, 0.0) for s in plot_sectors],
        'null_std_percentages':  [null_std_metrics.get(s, 0.0)  for s in plot_sectors],
        'secondary_scc_sizes':   secondary_sizes,
    }
    with open(PLOT_DATA_FILE, 'w') as f:
        json.dump(plot_data, f, indent=4)
    print(f"Plotting data saved to {PLOT_DATA_FILE}")

    print(f"\nAnalysis complete in {time.time() - start_time:.2f} seconds.")
    return plot_data

# =============================================================================
# PLOTTING
# =============================================================================

def create_sector_plot(plot_data):
    """
    Bar plot comparing bow-tie sector percentages (real vs. null models).
    Format and style unchanged from original.
    """
    if not plot_data:
        print("No data available for plotting.")
        return

    labels                = plot_data['labels']
    real_percentages      = plot_data['real_percentages']
    null_mean_percentages = plot_data['null_mean_percentages']
    null_std_percentages  = plot_data['null_std_percentages']

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(x - width/2, real_percentages, width,
           label='Real Network', color='#bcbd22')
    ax.bar(x + width/2, null_mean_percentages, width,
           yerr=null_std_percentages, capsize=5,
           label='Surrogate Data (Mean ± Std)', color='#cccccc',
           hatch='//', edgecolor='black', alpha=0.8)

    ax.set_ylabel('% of Nodes', fontsize=20)
    ax.set_title('Bow-Tie Sector Node Percentage Comparison',
                 fontsize=24, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=18)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=16)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    plt.savefig(FIG_DIR / "bow_tie_sizes_plot")


def create_secondary_scc_plot(secondary_sizes, min_size=MIN_SCC_SIZE):
    """
    Bar plot showing the size of each secondary SCC (>= min_size nodes),
    sorted from largest to smallest.

    Each bar represents one secondary SCC; the x-axis is a rank index
    (1 = second largest SCC overall, 2 = third largest, etc.).

    If no secondary SCCs were found above the threshold, the function prints
    a message and returns without showing a plot.
    """
    if not secondary_sizes:
        print(f"No secondary SCCs with >= {min_size} nodes found. "
              "Skipping secondary SCC plot.")
        return

    ranks  = np.arange(1, len(secondary_sizes) + 1)
    fig, ax = plt.subplots(figsize=(max(10, len(secondary_sizes) * 0.5 + 2), 6))

    bars = ax.bar(ranks, secondary_sizes, color='#1f77b4', edgecolor='black', alpha=0.85)

    # Annotate each bar with its exact size.
    for bar, size in zip(bars, secondary_sizes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(secondary_sizes) * 0.01,
            str(size),
            ha='center', va='bottom', fontsize=9
        )

    ax.set_xlabel('Secondary SCC rank (1 = largest secondary)', fontsize=14)
    ax.set_ylabel('Number of nodes', fontsize=14)
    ax.set_title(
        f'Secondary SCCs with ≥ {min_size} nodes (real network)\n'
        f'Total found: {len(secondary_sizes)}',
        fontsize=16, fontweight='bold'
    )
    ax.set_xticks(ranks)
    ax.set_xticklabels([str(r) for r in ranks], fontsize=10)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    plt.savefig(FIG_DIR / "secondary_SCC_plot")


# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    plotting_data   = main()
    secondary_sizes = plotting_data.get('secondary_scc_sizes', [])

    create_sector_plot(plotting_data)
    create_secondary_scc_plot(secondary_sizes)