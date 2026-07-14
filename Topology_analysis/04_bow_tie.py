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
import gc
import json
import time
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# =============================================================================
# CONFIG
# =============================================================================

script_dir      = Path(__file__).parent
INPUT_DATA_DIR  = Path(script_dir / "../../data")
FIG_DIR         = Path(script_dir / "figures")
OUTPUT_DATA_DIR = Path(script_dir / "data_output")
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

EDGE_LIST_PATH     = INPUT_DATA_DIR / "edge_list_to_analyze.parquet"
NULL_MODELS_DIR    = INPUT_DATA_DIR / "null_models"
CHECKPOINT_FILE    = OUTPUT_DATA_DIR / "null_model_checkpoint.jsonl"  # one result per line
RESULTS_FILE       = OUTPUT_DATA_DIR / "bow_tie_comparison_results.json"
PLOT_DATA_FILE     = OUTPUT_DATA_DIR / "bow_tie_plot_percentages.json"
SECONDARY_SCC_FILE = OUTPUT_DATA_DIR / "secondary_sccs.json"

MIN_SCC_SIZE = 3

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)


def log_memory(label: str):
    if not HAS_PSUTIL:
        return
    proc     = psutil.Process(os.getpid())
    rss_gb   = proc.memory_info().rss / 1e9
    avail_gb = psutil.virtual_memory().available / 1e9
    log.info(f"[MEM] {label}: RSS={rss_gb:.2f}GB | available={avail_gb:.2f}GB")


# =============================================================================
# CORE ALGORITHMS
# =============================================================================

def multi_source_bfs(G, sources):
    """BFS from a set of sources. Returns set of all reachable nodes."""
    visited = set(sources)
    queue   = deque(s for s in sources if s in G)
    while queue:
        u = queue.popleft()
        for v in G.neighbors(u):
            if v not in visited:
                visited.add(v)
                queue.append(v)
    return visited


def multi_source_bfs_with_dist(G, sources):
    """BFS with distance tracking. Used for average path length metrics."""
    dist  = {s: 0 for s in sources if s in G}
    queue = deque(dist.keys())
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


def calculate_bow_tie_metrics(G, return_sets=False):
    """
    Bow-tie decomposition (Broder et al. 2000).

    Key optimizations vs original:
      - multi_source_bfs from ALL SCC nodes (not nx.descendants from one node)
      - G.reverse(copy=False): no RAM duplication
      - We eliminate self-loops before analysis (they don't affect SCC membership)
      - BFS results reused for distance metrics (single traversal per direction)
    """
    N = G.number_of_nodes()
    if N == 0:
        return None

    sccs = get_all_sccs(G)
    if not sccs or len(sccs[0]) < 2:
        return None

    scc_nodes_set = set(sccs[0])

    # Single reverse graph view — no copy, no extra RAM
    G_rev = G.reverse(copy=False)

    # BFS forward from entire SCC simultaneously → finds OUT + distances
    dist_from_scc  = multi_source_bfs_with_dist(G, scc_nodes_set)
    out_sector_set = set(dist_from_scc.keys()) - scc_nodes_set

    # BFS backward from entire SCC simultaneously → finds IN + distances
    dist_to_scc   = multi_source_bfs_with_dist(G_rev, scc_nodes_set)
    in_sector_set = set(dist_to_scc.keys()) - scc_nodes_set

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
    scc_to_out_edges = sum(
        1 for u in scc_nodes_set
        for v in G.successors(u) if v in out_sector_set
    )

    ratio_in_scc  = in_to_scc_edges  / in_size  if in_size  > 0 else 0.0
    ratio_scc_out = scc_to_out_edges / scc_size if scc_size > 0 else 0.0

    distances_in  = [dist_to_scc[n]   for n in in_sector_set  if n in dist_to_scc]
    distances_out = [dist_from_scc[n] for n in out_sector_set if n in dist_from_scc]

    metrics = {
        'N_nodes':              N,
        'L_edges':              G.number_of_edges(),
        'in_pct':               (in_size     / N) * 100,
        'scc_pct':              (scc_size    / N) * 100,
        'out_pct':              (out_size    / N) * 100,
        'others_pct':           (others_size / N) * 100,
        'ratio_in_scc':         ratio_in_scc,
        'ratio_scc_out':        ratio_scc_out,
        'average_dist_in_scc':  float(np.mean(distances_in))  if distances_in  else float('nan'),
        'average_dist_scc_out': float(np.mean(distances_out)) if distances_out else float('nan'),
    }

    if return_sets:
        components = {
            'IN': in_sector_set,
            'SCC': scc_nodes_set,
            'OUT': out_sector_set,
            'OTHERS': others_set
        }
        return metrics, components

    return metrics

def get_secondary_scc_sizes(G, min_size=MIN_SCC_SIZE):
    all_sccs = get_all_sccs(G)
    if len(all_sccs) < 2:
        return []
    return sorted([len(c) for c in all_sccs[1:] if len(c) >= min_size], reverse=True)


# =============================================================================
# CHECKPOINTING
# =============================================================================

def load_checkpoint():
    """
    Reads completed null model results from the checkpoint file.
    Each line is a JSON object (one null model result).
    Returns list of result dicts and set of already-processed filenames.
    """
    if not CHECKPOINT_FILE.exists():
        return [], set()
    results  = []
    done     = set()
    with open(CHECKPOINT_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done.add(rec['filename'])
            results.append(rec['metrics'])
    return results, done


def append_checkpoint(filename: str, metrics: dict):
    """Appends one null model result to the checkpoint file (atomic per line)."""
    with open(CHECKPOINT_FILE, 'a') as f:
        f.write(json.dumps({'filename': filename, 'metrics': metrics}) + '\n')


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info("BOW-TIE ANALYSIS — START")
    log.info("=" * 60)
    log_memory("startup")

    # -------------------------------------------------------------------------
    # 1. Real network
    # -------------------------------------------------------------------------
    log.info(f"Loading real network from {EDGE_LIST_PATH}")
    edge_list = pd.read_parquet(EDGE_LIST_PATH)
    log.info(f"Edge list: {len(edge_list):,} rows | "
             f"sources: {edge_list['source'].nunique():,} | "
             f"targets: {edge_list['target'].nunique():,}")
    log_memory("after parquet load")

    G_full = nx.from_pandas_edgelist(
        edge_list, source='source', target='target', create_using=nx.DiGraph()
    )

    ## Only work with main WCC
    giant_component_nodes = max(nx.weakly_connected_components(G_full), key=len)
    G_real = G_full.subgraph(giant_component_nodes).copy()

    ## Eliminate self-loops
    G_real.remove_edges_from(nx.selfloop_edges(G_real))
    
    del edge_list
    gc.collect()
    log_memory("after graph construction")
    log.info(f"Graph: {G_real.number_of_nodes():,} nodes | {G_real.number_of_edges():,} edges")

    log.info("Computing bow-tie for real network...")
    t_real = time.time()
    bow_tie = calculate_bow_tie_metrics(G_real, return_sets=True)

    if bow_tie is None:
        raise ValueError("Real network has no valid SCC.")
    
    real_metrics, real_components = bow_tie 

    log.info(f"Real network bow-tie done in {time.time()-t_real:.1f}s")
    log_memory("after real bow-tie")

    if real_metrics is None:
        raise ValueError("Real network has no valid SCC.")
    log.info(f"Real metrics: {real_metrics}")

    # Save bow-tie components to parquet files for later use
    for comp_name, nodes in real_components.items():
        comp_df = pd.DataFrame({'Gene': list(nodes)})
        comp_df['Gene'] = comp_df['Gene'].astype(str)
        comp_path = OUTPUT_DATA_DIR / f"{comp_name.lower()}_sector_nodes.parquet"
        comp_df.to_parquet(comp_path, index=False)

    # -------------------------------------------------------------------------
    # 2. Null models — sequential with checkpointing
    # -------------------------------------------------------------------------
    null_files = sorted(NULL_MODELS_DIR.glob("*.parquet"))
    log.info(f"Null model files found: {len(null_files)}")

    if null_files:
        sample_mb = null_files[0].stat().st_size / 1e6
        log.info(f"Sample null model size: {sample_mb:.1f} MB")

    # Load previously completed results
    null_metrics_list, already_done = load_checkpoint()
    log.info(f"Checkpoint: {len(already_done)} null models already processed")

    remaining = [f for f in null_files if f.name not in already_done]
    log.info(f"Remaining: {len(remaining)} null models to process")

    failed_count = 0
    for i, fp in enumerate(tqdm(remaining, desc="Null models")):
        t_iter = time.time()
        try:
            edges  = pd.read_parquet(fp)
            G_null = nx.from_pandas_edgelist(
                edges, source='source', target='target', create_using=nx.DiGraph()
            )
            G_null.remove_edges_from(nx.selfloop_edges(G_null))
            del edges

            metrics = calculate_bow_tie_metrics(G_null)
            del G_null
            gc.collect()

            if metrics is not None:
                null_metrics_list.append(metrics)
                append_checkpoint(fp.name, metrics)
            else:
                failed_count += 1
                log.warning(f"No valid SCC in {fp.name}")

        except Exception as e:
            failed_count += 1
            log.error(f"Failed {fp.name}: {e}")
            gc.collect()

        # Memory + timing report every 50 models
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta     = (len(remaining) - i - 1) / rate / 3600 if rate > 0 else float('nan')
            log.info(f"Progress: {i+1}/{len(remaining)} | "
                     f"elapsed: {elapsed/3600:.2f}h | ETA: {eta:.2f}h | "
                     f"iter time: {time.time()-t_iter:.1f}s")
            log_memory(f"iter {i+1}")

    log.info(f"Null models: {len(null_metrics_list)} OK | {failed_count} failed")

    if not null_metrics_list:
        raise ValueError("No null model metrics collected.")

    df_null           = pd.DataFrame(null_metrics_list)
    null_mean_metrics = df_null.mean().to_dict()
    null_std_metrics  = df_null.std().to_dict()

    # -------------------------------------------------------------------------
    # 3. Secondary SCCs
    # -------------------------------------------------------------------------
    log.info(f"Computing secondary SCCs (min_size={MIN_SCC_SIZE})...")
    secondary_sizes = get_secondary_scc_sizes(G_real, min_size=MIN_SCC_SIZE)
    log.info(f"Secondary SCCs: {len(secondary_sizes)} | top 5: {secondary_sizes[:5]}")

    # -------------------------------------------------------------------------
    # 4. Save all results
    # -------------------------------------------------------------------------
    with open(RESULTS_FILE, 'w') as f:
        json.dump({
            'real_metrics':      real_metrics,
            'null_mean_metrics': null_mean_metrics,
            'null_std_metrics':  null_std_metrics,
        }, f, indent=4)

    with open(SECONDARY_SCC_FILE, 'w') as f:
        json.dump({'secondary_scc_sizes': secondary_sizes,
                   'min_size_threshold':  MIN_SCC_SIZE}, f, indent=4)

    # -------------------------------------------------------------------------
    # 5. Summary table
    # -------------------------------------------------------------------------
    metrics_labels = {
        'in_pct':               '% IN',
        'scc_pct':              '% SCC',
        'out_pct':              '% OUT',
        'others_pct':           '% OTHERS',
        'ratio_in_scc':         'Edges IN→SCC / IN node',
        'ratio_scc_out':        'Edges SCC→OUT / SCC node',
        'average_dist_in_scc':  'Avg dist IN→SCC',
        'average_dist_scc_out': 'Avg dist SCC→OUT',
    }
    log.info("\n" + "="*80)
    log.info(f"{'Metric':<30} | {'Real':<12} | Null mean ± std")
    log.info("-"*80)
    for key, label in metrics_labels.items():
        rv = real_metrics.get(key, float('nan'))
        nm = null_mean_metrics.get(key, float('nan'))
        ns = null_std_metrics.get(key, float('nan'))
        log.info(f"{label:<30} | {rv:<12.4f} | {nm:.4f} ± {ns:.4f}")
    log.info("="*80)

    plot_sectors = ['in_pct', 'scc_pct', 'out_pct', 'others_pct']
    plot_data = {
        'labels':                ['IN', 'SCC', 'OUT', 'Others'],
        'real_percentages':      [real_metrics.get(s, 0.0)      for s in plot_sectors],
        'null_mean_percentages': [null_mean_metrics.get(s, 0.0) for s in plot_sectors],
        'null_std_percentages':  [null_std_metrics.get(s, 0.0)  for s in plot_sectors],
        'secondary_scc_sizes':   secondary_sizes,
    }
    with open(PLOT_DATA_FILE, 'w') as f:
        json.dump(plot_data, f, indent=4)

    log.info(f"Total runtime: {(time.time()-t0)/3600:.2f}h")
    log_memory("end")
    return plot_data


# =============================================================================
# PLOTTING
# =============================================================================

def create_sector_plot(plot_data):
    labels = plot_data['labels']
    x, width = np.arange(len(labels)), 0.35
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(x - width/2, plot_data['real_percentages'], width,
           label='Real Network', color='#bcbd22')
    ax.bar(x + width/2, plot_data['null_mean_percentages'], width,
           yerr=plot_data['null_std_percentages'], capsize=5,
           label='Surrogate (Mean ± Std)', color='#cccccc',
           hatch='//', edgecolor='black', alpha=0.8)
    ax.set_ylabel('% of Nodes', fontsize=20)
    ax.set_title('Bow-Tie Sector Comparison', fontsize=24, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=18)
    ax.legend(fontsize=16)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    plt.savefig(FIG_DIR / "bow_tie_sizes_plot.pdf")
    plt.close()


def create_secondary_scc_plot(secondary_sizes, min_size=MIN_SCC_SIZE):
    if not secondary_sizes:
        log.info("No secondary SCCs to plot.")
        return
    ranks = np.arange(1, len(secondary_sizes) + 1)
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
    ax.grid(axis='y', linestyle='--', alpha=0.4)
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