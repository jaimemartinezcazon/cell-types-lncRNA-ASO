'''
Bow-tie decomposition with memory diagnostics and safe null model processing.
Key changes from original:
  - psutil memory logging at each step
  - G.reverse(copy=False) to avoid doubling RAM
  - null models processed sequentially (or with NUM_CORES=1 by default)
  - all logs written to a file for post-mortem analysis
  - upfront graph size diagnostics before any heavy computation
'''

import os
import gc
import json
import time
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import multiprocessing as mp
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("WARNING: psutil not available. Install with: pip install psutil")

# =============================================================================
# SETUP
# =============================================================================

script_dir     = Path(__file__).parent
INPUT_DATA_DIR = Path(script_dir / "../../data")
FIG_DIR        = Path(script_dir / "figures")
OUTPUT_DATA_DIR= Path(script_dir / "data_output")
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

EDGE_LIST_PATH  = INPUT_DATA_DIR / "edge_list_to_analyze.parquet"
NULL_MODELS_DIR = INPUT_DATA_DIR / "null_models"
RESULTS_FILE    = OUTPUT_DATA_DIR / "bow_tie_comparison_results.json"
PLOT_DATA_FILE  = OUTPUT_DATA_DIR / "bow_tie_plot_percentages.json"
SECONDARY_SCC_FILE = OUTPUT_DATA_DIR / "secondary_sccs.json"
LOG_FILE        = OUTPUT_DATA_DIR / "bowtie_run.log"

# CRITICAL FIX: use 1 core by default to avoid OOM from parallel graph loading.
# Increase only if you have confirmed that each null model graph fits comfortably
# in RAM × NUM_CORES (check null model file sizes first).
NUM_CORES   = 1
MIN_SCC_SIZE = 3

# =============================================================================
# LOGGING — writes to both console and file
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def log_memory(label: str):
    """Log current RAM usage at a checkpoint."""
    if HAS_PSUTIL:
        proc = psutil.Process(os.getpid())
        rss_gb = proc.memory_info().rss / 1e9
        avail_gb = psutil.virtual_memory().available / 1e9
        log.info(f"[MEM] {label} — process RSS: {rss_gb:.2f} GB | "
                 f"system available: {avail_gb:.2f} GB")
    else:
        log.info(f"[MEM] {label} — psutil not available, cannot report memory")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def multi_source_shortest_path_length(G, sources):
    """BFS from a set of sources. Returns {node: distance}."""
    dist  = {}
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
    return sorted(
        (frozenset(c) for c in nx.strongly_connected_components(G)),
        key=len, reverse=True
    )


def calculate_bow_tie_metrics(G):
    """
    Bow-tie decomposition (Broder et al. 2000).
    FIXED: uses G.reverse(copy=False) to avoid doubling RAM.
    """
    N = G.number_of_nodes()
    if N == 0:
        return None

    sccs = get_all_sccs(G)
    if not sccs or len(sccs[0]) < 2:
        return None

    scc_nodes_set = set(sccs[0])
    rep_node      = sorted(scc_nodes_set)[0]

    # OUT: descendants of SCC
    reachable_from_scc = nx.descendants(G, rep_node)
    out_sector_set     = reachable_from_scc - scc_nodes_set

    # CRITICAL FIX: copy=False avoids duplicating the full graph in RAM
    G_rev             = G.reverse(copy=False)
    reachable_to_scc  = nx.descendants(G_rev, rep_node)
    in_sector_set     = reachable_to_scc - scc_nodes_set

    others_set = set(G.nodes()) - scc_nodes_set - in_sector_set - out_sector_set

    in_size     = len(in_sector_set)
    scc_size    = len(scc_nodes_set)
    out_size    = len(out_sector_set)
    others_size = len(others_set)

    assert in_size + scc_size + out_size + others_size == N

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

    dist_to_scc    = multi_source_shortest_path_length(G_rev, scc_nodes_set)
    distances_in   = [dist_to_scc[n] for n in in_sector_set if n in dist_to_scc]
    avg_dist_in_scc = np.mean(distances_in) if distances_in else float('nan')

    dist_from_scc    = multi_source_shortest_path_length(G, scc_nodes_set)
    distances_out    = [dist_from_scc[n] for n in out_sector_set if n in dist_from_scc]
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
    all_sccs = get_all_sccs(G)
    if len(all_sccs) < 2:
        return []
    return sorted([len(c) for c in all_sccs[1:] if len(c) >= min_size], reverse=True)


def process_single_null_model(filepath):
    """Worker: load one graphml and return bow-tie metrics."""
    try:
        G_null = nx.read_graphml(filepath, node_type=str)
        metrics = calculate_bow_tie_metrics(G_null)
        del G_null
        gc.collect()
        return metrics
    except Exception as e:
        log.warning(f"Failed to process {filepath}: {e}")
        return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    start_time = time.time()
    log.info("="*60)
    log.info("BOW-TIE ANALYSIS — START")
    log.info("="*60)
    log_memory("startup")

    # -------------------------------------------------------------------------
    # 1. Load and inspect the real network
    # -------------------------------------------------------------------------
    if os.path.exists(RESULTS_FILE):
        log.info(f"Loading cached results from {RESULTS_FILE}")
        with open(RESULTS_FILE, 'r') as f:
            results_data = json.load(f)
        real_metrics      = results_data['real_metrics']
        null_mean_metrics = results_data['null_mean_metrics']
        null_std_metrics  = results_data['null_std_metrics']
        G_real = None
    else:
        log.info(f"Loading edge list from {EDGE_LIST_PATH}")
        edge_list = pd.read_parquet(EDGE_LIST_PATH)
        log.info(f"Edge list loaded: {len(edge_list):,} rows")
        log_memory("after parquet load")

        # DIAGNOSTIC: report graph statistics before building the NetworkX object
        n_unique_sources  = edge_list['source'].nunique()
        n_unique_targets  = edge_list['target'].nunique()
        all_nodes         = set(edge_list['source']) | set(edge_list['target'])
        log.info(f"Unique sources (TFs): {n_unique_sources:,}")
        log.info(f"Unique targets:       {n_unique_targets:,}")
        log.info(f"Total unique nodes:   {len(all_nodes):,}")
        log.info(f"Total edges:          {len(edge_list):,}")
        log.info("NOTE: if total edges >> 7M, you may be using a dense-orphan GRN version.")
        log.info("      Consider using a strict GRN version for topological analysis.")

        log.info("Building NetworkX DiGraph...")
        G_real = nx.from_pandas_edgelist(
            edge_list, source='source', target='target',
            create_using=nx.DiGraph()
        )
        G_real.remove_edges_from(nx.selfloop_edges(G_real))
        del edge_list
        gc.collect()
        log_memory("after graph construction")

        log.info(f"Graph: {G_real.number_of_nodes():,} nodes, "
                 f"{G_real.number_of_edges():,} edges")

        log.info("Computing bow-tie metrics for real network...")
        real_metrics = calculate_bow_tie_metrics(G_real)
        log_memory("after real network bow-tie")

        if real_metrics is None:
            raise ValueError("Real network has no valid SCC.")

        log.info(f"Real network metrics: {real_metrics}")

        # -------------------------------------------------------------------------
        # 2. Null models — SEQUENTIAL by default to avoid OOM
        # -------------------------------------------------------------------------
        null_files = sorted(
            os.path.join(NULL_MODELS_DIR, f)
            for f in os.listdir(NULL_MODELS_DIR)
            if f.endswith(".graphml")
        )
        log.info(f"Found {len(null_files)} null model files")

        # Report null model file sizes before loading
        if null_files:
            sample_size_mb = os.path.getsize(null_files[0]) / 1e6
            log.info(f"Sample null model file size: {sample_size_mb:.1f} MB")
            log.info(f"Estimated memory per null model graph: "
                     f"~{sample_size_mb * 5:.0f}–{sample_size_mb * 10:.0f} MB "
                     f"(NetworkX overhead ~5–10× file size)")
            log.info(f"Processing with NUM_CORES={NUM_CORES}")

        null_metrics_list = []
        failed_count      = 0

        if NUM_CORES == 1:
            # Sequential — safe for large graphs
            for i, fp in enumerate(tqdm(null_files, desc="Null models")):
                if i % 10 == 0:
                    log_memory(f"null model {i}/{len(null_files)}")
                res = process_single_null_model(fp)
                if res is not None:
                    null_metrics_list.append(res)
                else:
                    failed_count += 1
        else:
            # Parallel — only use if you have confirmed sufficient RAM
            log.warning(f"Using {NUM_CORES} parallel workers. "
                        "Monitor memory carefully — reduce NUM_CORES if OOM occurs.")
            with mp.Pool(processes=NUM_CORES) as pool:
                for res in tqdm(
                    pool.imap_unordered(process_single_null_model, null_files),
                    total=len(null_files)
                ):
                    if res is not None:
                        null_metrics_list.append(res)
                    else:
                        failed_count += 1

        log.info(f"Null models processed: {len(null_metrics_list)} OK, "
                 f"{failed_count} failed")
        log_memory("after null models")

        if not null_metrics_list:
            raise ValueError("No valid null model metrics collected.")

        df_null           = pd.DataFrame(null_metrics_list)
        null_mean_metrics = df_null.mean().to_dict()
        null_std_metrics  = df_null.std().to_dict()

        with open(RESULTS_FILE, 'w') as f:
            json.dump({
                'real_metrics':      real_metrics,
                'null_mean_metrics': null_mean_metrics,
                'null_std_metrics':  null_std_metrics,
            }, f, indent=4)
        log.info(f"Results saved to {RESULTS_FILE}")

    # -------------------------------------------------------------------------
    # 3. Secondary SCCs
    # -------------------------------------------------------------------------
    if G_real is None:
        log.info("Reloading real network for secondary SCC analysis...")
        edge_list = pd.read_parquet(EDGE_LIST_PATH)
        G_real    = nx.from_pandas_edgelist(
            edge_list, source='source', target='target',
            create_using=nx.DiGraph()
        )
        G_real.remove_edges_from(nx.selfloop_edges(G_real))
        del edge_list
        gc.collect()

    log.info(f"Computing secondary SCCs (min_size={MIN_SCC_SIZE})...")
    secondary_sizes = get_secondary_scc_sizes(G_real, min_size=MIN_SCC_SIZE)
    log.info(f"Secondary SCCs found: {len(secondary_sizes)}")
    if secondary_sizes:
        log.info(f"Top 10 sizes: {secondary_sizes[:10]}")

    with open(SECONDARY_SCC_FILE, 'w') as f:
        json.dump({'secondary_scc_sizes': secondary_sizes,
                   'min_size_threshold': MIN_SCC_SIZE}, f, indent=4)

    # -------------------------------------------------------------------------
    # 4. Summary table
    # -------------------------------------------------------------------------
    metrics_to_print = {
        'in_pct':               '% IN Nodes',
        'scc_pct':              '% SCC Nodes',
        'out_pct':              '% OUT Nodes',
        'others_pct':           '% OTHERS Nodes',
        'ratio_in_scc':         'Edges IN→SCC / IN Node',
        'ratio_scc_out':        'Edges SCC→OUT / SCC Node',
        'average_dist_in_scc':  'Avg Dist IN → SCC',
        'average_dist_scc_out': 'Avg Dist SCC → OUT',
    }
    log.info("\n" + "="*85)
    log.info(f"{'Metric':<35} | {'Real Network':<15} | Null (Mean ± Std)")
    log.info("-"*85)
    for key, label in metrics_to_print.items():
        rv   = real_metrics.get(key, float('nan'))
        nm   = null_mean_metrics.get(key, float('nan'))
        ns   = null_std_metrics.get(key, float('nan'))
        log.info(f"{label:<35} | {rv:<15.4f} | {nm:.4f} ± {ns:.4f}")
    log.info("="*85)

    # -------------------------------------------------------------------------
    # 5. Plot data
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

    log.info(f"Done in {time.time() - start_time:.1f}s")
    log_memory("end of main")
    return plot_data


# =============================================================================
# PLOTTING — unchanged
# =============================================================================

def create_sector_plot(plot_data):
    if not plot_data:
        return
    labels                = plot_data['labels']
    real_percentages      = plot_data['real_percentages']
    null_mean_percentages = plot_data['null_mean_percentages']
    null_std_percentages  = plot_data['null_std_percentages']

    x, width = np.arange(len(labels)), 0.35
    fig, ax  = plt.subplots(figsize=(12, 8))
    ax.bar(x - width/2, real_percentages, width, label='Real Network', color='#bcbd22')
    ax.bar(x + width/2, null_mean_percentages, width,
           yerr=null_std_percentages, capsize=5,
           label='Surrogate Data (Mean ± Std)', color='#cccccc',
           hatch='//', edgecolor='black', alpha=0.8)
    ax.set_ylabel('% of Nodes', fontsize=20)
    ax.set_title('Bow-Tie Sector Node Percentage Comparison', fontsize=24, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=18)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=16)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    plt.savefig(FIG_DIR / "bow_tie_sizes_plot.pdf")
    plt.close()


def create_secondary_scc_plot(secondary_sizes, min_size=MIN_SCC_SIZE):
    if not secondary_sizes:
        log.info(f"No secondary SCCs >= {min_size} nodes. Skipping plot.")
        return
    ranks  = np.arange(1, len(secondary_sizes) + 1)
    fig, ax = plt.subplots(figsize=(max(10, len(secondary_sizes) * 0.5 + 2), 6))
    bars = ax.bar(ranks, secondary_sizes, color='#1f77b4', edgecolor='black', alpha=0.85)
    for bar, size in zip(bars, secondary_sizes):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(secondary_sizes) * 0.01,
                str(size), ha='center', va='bottom', fontsize=9)
    ax.set_xlabel('Secondary SCC rank', fontsize=14)
    ax.set_ylabel('Number of nodes', fontsize=14)
    ax.set_title(f'Secondary SCCs ≥ {min_size} nodes | Total: {len(secondary_sizes)}',
                 fontsize=16, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    plt.savefig(FIG_DIR / "secondary_SCC_plot.pdf")
    plt.close()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    plotting_data   = main()
    secondary_sizes = plotting_data.get('secondary_scc_sizes', [])
    create_sector_plot(plotting_data)
    create_secondary_scc_plot(secondary_sizes)