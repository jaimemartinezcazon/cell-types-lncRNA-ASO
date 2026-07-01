'''
Author: Jaime Martínez Cazón
Adapted for direct gene names and unparquet edge lists.

Description:
Analyzes the relationship between the clustering coefficient C(k) and node 
degree (k) for a Gene Regulatory Network. Compares the real network against 
an ensemble of surrogate networks (null models). Includes exponential binning 
and power-law fitting for C(k) vs. k.
'''

import os
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import linregress
import matplotlib.pyplot as plt

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

# Bow-tie data path
NULL_MODELS_DIR = INPUT_DATA_DIR / "null_models"

N_NULL_MODELS = 1000  

# fit power law in data?
FIT_POWER_LAW = False

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_empirical_p_value(real_value, null_distribution, direction='greater'):
    """Calculates the empirical p-value based on a null distribution."""
    n_simulations = len(null_distribution)
    if n_simulations == 0:
        return np.nan
        
    null_array = np.array(null_distribution)
    
    if direction == 'greater':
        count = np.sum(null_array >= real_value)
    else:
        count = np.sum(null_array <= real_value)
        
    p_value = (count + 1) / (n_simulations + 1)
    return p_value


def exponential_binning(data_points, base=1.5):
    """Applies exponential binning to a list of (degree, value) data points."""
    if not data_points:
        return np.array([]), np.array([]), np.array([])

    degrees = np.array([k for k, _ in data_points])
    c_values = np.array([c for _, c in data_points])

    max_degree = np.max(degrees)
    min_degree = np.min(degrees)
    
    current_edge = float(min_degree)
    bin_edges = [current_edge]
    while current_edge <= max_degree:
        current_edge *= base
        bin_edges.append(current_edge)

    binned_k_means, binned_c_means, binned_c_stds = [], [], []
    for i in range(len(bin_edges) - 1):
        low_bound, high_bound = bin_edges[i], bin_edges[i+1]
        
        indices = np.where((degrees >= low_bound) & (degrees < high_bound))[0]
        if len(indices) > 0:
            binned_k_means.append(np.mean(degrees[indices]))
            binned_c_means.append(np.mean(c_values[indices]))
            binned_c_stds.append(np.std(c_values[indices]))

    return np.array(binned_k_means), np.array(binned_c_means), np.array(binned_c_stds)

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_real_network():
    """Loads the real network and extracts its giant component."""
    print("Loading real network...")
    edge_list = pd.read_parquet(EDGE_LIST_PATH)
    G_full = nx.from_pandas_edgelist(edge_list, source='source', target='target', create_using=nx.DiGraph())
    
    undirected_components = (G_full.subgraph(c) for c in nx.connected_components(G_full.to_undirected()))
    G_real = max(undirected_components, key=len)
    G_real = G_full.subgraph(G_real.nodes()).copy() 
    
    print(f"Real network loaded: {G_real.number_of_nodes()} nodes, {G_real.number_of_edges()} edges.")
    return G_real

def analyze_null_models():
    """Loads and analyzes the ensemble of null models."""
    print(f"\nLoading and analyzing up to {N_NULL_MODELS} null models...")
    null_global_clustering = []
    null_ck_raw_data = {"all": defaultdict(list), "in": defaultdict(list), "out": defaultdict(list)}
    
    for i in tqdm(range(N_NULL_MODELS)):
        file_path = os.path.join(NULL_MODELS_DIR, f"null_model_{str(i).zfill(4)}.graphml")
        if not os.path.exists(file_path):
            continue
            
        try:
            # node_type=str because gene names are now strings
            G_null = nx.read_graphml(file_path, node_type=str)
            
            local_c_null = nx.clustering(G_null.to_undirected())
            null_global_clustering.append(np.mean(list(local_c_null.values())))

            for node, c_val in local_c_null.items():
                if c_val > 0:
                    null_ck_raw_data["all"][G_null.degree(node)].append(c_val)
                    null_ck_raw_data["in"][G_null.in_degree(node)].append(c_val)
                    null_ck_raw_data["out"][G_null.out_degree(node)].append(c_val)
        except Exception as e:
            tqdm.write(f"Warning: Could not process {os.path.basename(file_path)}. Error: {e}")

    return null_global_clustering, null_ck_raw_data


# =============================================================================
# PLOTTING AND ANALYSIS
# =============================================================================

def plot_ck_distribution(real_data, null_data, degree_type):
    """Creates a C(k) vs. k plot with exponential binning and power-law fit."""
    plt.figure(figsize=(9, 7))
    ax = plt.gca()

    # Real network data
    real_raw_points = [(k, c) for k, c in zip(real_data[f'degree_{degree_type}'], real_data['local_clustering']) if k > 0 and c > 0]
    if not real_raw_points:
        print(f"Warning: No valid data points for real network ({degree_type} degree).")
        return

    k_real_binned, c_real_binned, c_real_std = exponential_binning(real_raw_points)
    
    ax.scatter([k for k,c in real_raw_points], [c for k,c in real_raw_points],
               color='#8F9058', marker='o', s=20, alpha=0.1, label='Real Network (Raw)', zorder=1)
    ax.errorbar(k_real_binned, c_real_binned, yerr=c_real_std, fmt='s', color='#bcbd22',
                markersize=8, label='Real Network (Binned)', zorder=10)

    # Null model data
    null_raw_points = [(k, c) for k, c_list in null_data[degree_type].items() for c in c_list if k > 0]
    if null_raw_points:
        k_null_binned, c_null_binned, c_null_std = exponential_binning(null_raw_points)
        ax.errorbar(k_null_binned, c_null_binned, yerr=c_null_std, fmt='o', color='black',
                    markersize=5, alpha=0.8, label='Null Model (Binned)', zorder=5)

    # Power-law fit
    if FIT_POWER_LAW and len(k_real_binned) > 1:
        valid_indices = c_real_binned > 0
        if np.sum(valid_indices) > 1:
            log_k = np.log(k_real_binned[valid_indices])
            log_c = np.log(c_real_binned[valid_indices])
            slope, intercept, _, _, _ = linregress(log_k, log_c)
            
            fit_k = np.linspace(min(k_real_binned), max(k_real_binned), 100)
            fit_c = np.exp(intercept) * (fit_k ** slope)
            ax.plot(fit_k, fit_c, color='#d62728', linewidth=3,
                    label=fr'Fit ($\gamma={slope:.2f}$)', zorder=9)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel("Degree (k)", fontsize=18)
    ax.set_ylabel("Clustering C(k)", fontsize=18)
    ax.set_title(f"C(k) vs. {degree_type.capitalize()} Degree", fontsize=22, fontweight='bold')
    ax.tick_params(labelsize=16)
    ax.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "clustering_undirected_plot.png")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    G_real = load_real_network()
    
    real_network_data = {
        'local_clustering': list(nx.clustering(G_real.to_undirected()).values()),
        'global_clustering': np.mean(list(nx.clustering(G_real.to_undirected()).values())),
        'degree_all': [d for _, d in G_real.degree()],
        'degree_in': [d for _, d in G_real.in_degree()],
        'degree_out': [d for _, d in G_real.out_degree()]
    }
    
    null_global_clustering, null_ck_raw_data = analyze_null_models()
    
    print("\n--- Global Clustering Significance Analysis ---")
    if null_global_clustering:
        p_value = calculate_empirical_p_value(
            real_network_data['global_clustering'], 
            null_global_clustering
        )
        print(f"Real Network Global Clustering: {real_network_data['global_clustering']:.4f}")
        print(f"Null Model Mean Clustering:     {np.mean(null_global_clustering):.4f} ± {np.std(null_global_clustering):.4f}")
        print(f"Empirical P-value:              {p_value:.4f}")
        if p_value < 0.05:
            print("Result is statistically significant (p < 0.05).")
        else:
            print("Result is not statistically significant (p >= 0.05).")
    else:
        print("No null model data available for significance testing.")
    print("-" * 50)
    
    print("\nGenerating C(k) vs. Degree plots...")
    for deg_type in ["all", "in", "out"]:
        plot_ck_distribution(real_network_data, null_ck_raw_data, deg_type)