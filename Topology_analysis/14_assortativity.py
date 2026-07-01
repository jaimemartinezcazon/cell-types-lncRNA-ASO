'''
Author: Jaime Martínez Cazón
Adapted for direct gene names, unparquet edge lists, bug fixes, and multiprocessing.

Description:
Analyzes the assortativity of a Gene Regulatory Network by calculating the 
Average Nearest-Neighbor degree (ANN) as a function of node degree (k). 
Compares the real network against an ensemble of surrogate networks using 
multiprocessing for speed. Analyzes undirected and 4 directed configurations.
'''

import os
import glob
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# SETUP: FILE PATHS AND PARAMETERS
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

NUM_CORES = max(1, mp.cpu_count() - 2)

# =============================================================================
# UTILITY AND ANALYSIS FUNCTIONS
# =============================================================================

def exponential_binning(data_points, base=1.5):
    """Applies exponential binning to (degree, value) data points."""
    if not data_points:
        return np.array([]), np.array([]), np.array([])
    
    k_data = np.array([k for k, _ in data_points])
    val_data = np.array([v for _, v in data_points])
    
    min_k, max_k = np.min(k_data), np.max(k_data)
    
    current_edge = float(min_k)
    bin_edges = [current_edge]
    while current_edge <= max_k:
        current_edge *= base
        bin_edges.append(current_edge)
    
    binned_k, binned_val, binned_std = [], [], []
    for i in range(len(bin_edges) - 1):
        low, high = bin_edges[i], bin_edges[i+1]
        indices = np.where((k_data >= low) & (k_data < high))[0]
        if len(indices) > 0:
            binned_k.append(np.mean(k_data[indices]))
            binned_val.append(np.mean(val_data[indices]))
            binned_std.append(np.std(val_data[indices]))
            
    return np.array(binned_k), np.array(binned_val), np.array(binned_std)


def load_real_network():
    """Loads the real network and extracts its giant component."""
    print("Loading real network...")
    edge_list = pd.read_parquet(EDGE_LIST_PATH)
    G_full = nx.from_pandas_edgelist(
        edge_list, source='source', target='target', create_using=nx.DiGraph()
    )
    
    giant_component_nodes = max(nx.weakly_connected_components(G_full), key=len)
    G_real = G_full.subgraph(giant_component_nodes).copy()
    
    print(f"Real network (giant component) loaded: {G_real.number_of_nodes()} nodes.")
    return G_real


def process_single_null_model(filepath):
    """Worker function to process ANN metrics for a single null model."""
    try:
        G_null = nx.read_graphml(filepath, node_type=str)
        if G_null.number_of_nodes() == 0: 
            return None

        res = {'undirected': [], 'in-in': [], 'in-out': [], 'out-in': [], 'out-out': []}
        
        # Undirected case
        G_undir = G_null.to_undirected()
        ann_undir = nx.average_neighbor_degree(G_undir)
        res['undirected'] = [(G_undir.degree(n), ann) for n, ann in ann_undir.items() if G_undir.degree(n) > 0]
        
        # Directed cases (IN)
        ann_inin = nx.average_neighbor_degree(G_null, source='in', target='in')
        ann_inout = nx.average_neighbor_degree(G_null, source='in', target='out')
        for n, k_in in G_null.in_degree():
            if k_in > 0:
                res['in-in'].append((k_in, ann_inin[n]))
                res['in-out'].append((k_in, ann_inout[n]))
                
        # Directed cases (OUT)
        ann_outin = nx.average_neighbor_degree(G_null, source='out', target='in')
        ann_outout = nx.average_neighbor_degree(G_null, source='out', target='out')
        for n, k_out in G_null.out_degree():
            if k_out > 0:
                res['out-in'].append((k_out, ann_outin[n]))
                res['out-out'].append((k_out, ann_outout[n]))

        return res
    except Exception:
        return None


def plot_ann_distribution(ax, real_data, null_data, title, x_label, y_label, y_scale='linear'):
    """Creates one ANN(k) vs. k plot comparing real and null model data."""
    # --- Plot real network data ---
    if real_data:
        k_real, ann_real = zip(*real_data)
        k_real_binned, ann_real_binned, _ = exponential_binning(real_data)
        
        ax.scatter(k_real, ann_real, color='#8F9058', s=20, alpha=0.1, label='Real Network (Raw)', zorder=1)
        ax.plot(k_real_binned, ann_real_binned, 's-', color='#bcbd22', markersize=8, label='Real Network (Binned)', zorder=10)
        
    # --- Plot null model data ---
    if null_data:
        k_null_binned, ann_null_binned, ann_null_std = exponential_binning(null_data)
        ax.errorbar(k_null_binned, ann_null_binned, yerr=ann_null_std, fmt='o',
                    color='black', markersize=5, alpha=0.8, label='Null Model (Binned)', zorder=5)

    # --- Formatting ---
    ax.set_xscale('log')
    ax.set_yscale(y_scale)
    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # --- 1. Load and analyze the real network ---
    G_real = load_real_network()
    
    print("\nCalculating ANN for real network...")
    G_real_undir = G_real.to_undirected()
    
    # Fixed the dictionary comprehension logic from the original code
    real_ann_data = {
        'undirected': [(G_real_undir.degree(n), ann) for n, ann in nx.average_neighbor_degree(G_real_undir).items() if G_real_undir.degree(n) > 0],
        'in-in':      [(G_real.in_degree(n), ann) for n, ann in nx.average_neighbor_degree(G_real, source='in', target='in').items() if G_real.in_degree(n) > 0],
        'in-out':     [(G_real.in_degree(n), ann) for n, ann in nx.average_neighbor_degree(G_real, source='in', target='out').items() if G_real.in_degree(n) > 0],
        'out-in':     [(G_real.out_degree(n), ann) for n, ann in nx.average_neighbor_degree(G_real, source='out', target='in').items() if G_real.out_degree(n) > 0],
        'out-out':    [(G_real.out_degree(n), ann) for n, ann in nx.average_neighbor_degree(G_real, source='out', target='out').items() if G_real.out_degree(n) > 0]
    }
    
    # --- 2. Load and analyze null models ---
    null_files = glob.glob(os.path.join(NULL_MODELS_DIR, "*.graphml"))
    null_ann_points = {'undirected': [], 'in-in': [], 'in-out': [], 'out-in': [], 'out-out': []}
    
    if null_files:
        print(f"Calculating ANN for {len(null_files)} null models using {NUM_CORES} cores...")
        with mp.Pool(processes=NUM_CORES) as pool:
            for res in tqdm(pool.imap_unordered(process_single_null_model, null_files), total=len(null_files)):
                if res is not None:
                    for key in null_ann_points:
                        null_ann_points[key].extend(res[key])
    else:
        print(f"Warning: No null models found in {NULL_MODELS_DIR}")

    # --- 3. Generate plots ---
    print("\nGenerating Assortativity (ANN) plots...")
    
    plot_configs = [
        {"case": "undirected", "title": "Undirected Assortativity", "xlabel": "Degree k", "ylabel": "Avg. Neighbor Degree", "yscale": "log"},
        {"case": "in-in", "title": "Directed Assortativity (In-In)", "xlabel": "In-Degree k_in", "ylabel": "Avg. Neighbor In-Degree", "yscale": "linear"},
        {"case": "in-out", "title": "Directed Assortativity (In-Out)", "xlabel": "In-Degree k_in", "ylabel": "Avg. Neighbor Out-Degree", "yscale": "linear"},
        {"case": "out-in", "title": "Directed Assortativity (Out-In)", "xlabel": "Out-Degree k_out", "ylabel": "Avg. Neighbor In-Degree", "yscale": "linear"},
        {"case": "out-out", "title": "Directed Assortativity (Out-Out)", "xlabel": "Out-Degree k_out", "ylabel": "Avg. Neighbor Out-Degree", "yscale": "log"}
    ]

    for config in plot_configs:
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_ann_distribution(
            ax,
            real_ann_data[config["case"]],
            null_ann_points[config["case"]],
            config["title"],
            config["xlabel"],
            config["ylabel"],
            config["yscale"]
        )
        plt.tight_layout()
        plt.savefig(FIG_DIR / "assortativity_plot.png")

    print("\nAnalysis complete.")