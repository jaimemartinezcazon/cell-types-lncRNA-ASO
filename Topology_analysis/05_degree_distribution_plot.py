'''
Author: Jaime Martínez Cazón
Adapted for direct gene names, unparquet edge lists, and toggleable power-law fit.

Description:
Analyzes and visualizes the degree distributions of a Gene Regulatory Network.
Generates separate figures for CCDFs, Histograms, and a density scatter plot.
'''

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib import Path

# =============================================================================
# SETUP: FILE PATHS
# =============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
EDGE_LIST_PATH = os.path.join(script_dir, "../data/celloracle_data/base_GRN_edge_list.parquet")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_ccdf(degrees):
    if not isinstance(degrees, np.ndarray):
        degrees = np.array(degrees)
    
    unique_degrees, counts = np.unique(degrees, return_counts=True)
    cumulative_counts = np.cumsum(counts[::-1])[::-1]
    ccdf = cumulative_counts / len(degrees)
    
    return unique_degrees, ccdf

def exponential_binning(k_data, val_data, base=1.5):
    if len(k_data) == 0:
        return np.array([]), np.array([]), np.array([])
        
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

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_ccdf_distribution(ax, degrees, title, fit_power_law=False):
    k_unique, ccdf_values = calculate_ccdf(degrees)
    valid_indices = (k_unique > 0) & (ccdf_values > 0)
    k_plot, ccdf_plot = k_unique[valid_indices], ccdf_values[valid_indices]
    
    if len(k_plot) == 0:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        return

    ax.scatter(k_plot, ccdf_plot, marker='o', s=30, alpha=0.2, color='#8F9058', label='Raw CCDF', zorder=1)
    
    k_binned, ccdf_binned, ccdf_std = exponential_binning(k_plot, ccdf_plot)
    ax.errorbar(k_binned, ccdf_binned, yerr=ccdf_std, fmt='s', color='#bcbd22',
                markersize=8, capsize=5, label='Binned CCDF', zorder=10)
                
    if fit_power_law and len(k_plot) > 1:
        log_k = np.log(k_plot)
        log_ccdf = np.log(ccdf_plot)
        
        try:
            slope, intercept, r_value, _, _ = linregress(log_k, log_ccdf)
            gamma = -slope + 1 
            
            fit_k_range = np.logspace(np.log10(min(k_plot)), np.log10(max(k_plot)), 100)
            fit_ccdf = np.exp(intercept) * (fit_k_range ** slope)
            
            ax.plot(fit_k_range, fit_ccdf, color='#d62728', linewidth=3,
                    label=f'Fit (γ ≈ {gamma:.2f})', zorder=9)
        except ValueError:
            pass

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Degree (k)", fontsize=16)
    ax.set_ylabel("CCDF P(K ≥ k)", fontsize=16)
    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.legend(fontsize=12)

def plot_degree_histogram(ax, degrees, title, xlabel):
    ax.hist(degrees, bins=50, color='#1f77b4', edgecolor='black', log=True, alpha=0.8)
    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("Frequency (log scale)", fontsize=16)

def plot_density_scatter(ax, fig, G):
    nodes = list(G.nodes())
    in_deg = [G.in_degree(n) for n in nodes]
    out_deg = [G.out_degree(n) for n in nodes]
    
    # Using hexbin for density visualization. 'bins='log'' helps visualize highly skewed distributions.
    hb = ax.hexbin(in_deg, out_deg, gridsize=50, cmap='viridis', bins='log', mincnt=1)
    
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('log10(N nodes)', fontsize=14)
    
    ax.set_title("In-Degree vs Out-Degree Density", fontsize=20, fontweight='bold')
    ax.set_xlabel("In-Degree", fontsize=16)
    ax.set_ylabel("Out-Degree", fontsize=16)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    FIT_POWER_LAW = False 

    print("Loading network data...")
    edge_list = pd.read_parquet(EDGE_LIST_PATH)
    
    G_full = nx.from_pandas_edgelist(
        edge_list, 
        source='source', 
        target='target', 
        create_using=nx.DiGraph()
    )
    
    print("Extracting giant component and calculating degrees...")
    giant_component_nodes = max(nx.weakly_connected_components(G_full), key=len)
    G = G_full.subgraph(giant_component_nodes).copy()
    
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]
    total_degrees = [d for _, d in G.to_undirected().degree()]

    print("Generating plots...")
    
    # --- FIGURE 1: CCDFs ---
    fig1, axes1 = plt.subplots(1, 3, figsize=(24, 7))
    plot_ccdf_distribution(axes1[0], in_degrees, "In-Degree CCDF", fit_power_law=FIT_POWER_LAW)
    plot_ccdf_distribution(axes1[1], out_degrees, "Out-Degree CCDF", fit_power_law=FIT_POWER_LAW)
    plot_ccdf_distribution(axes1[2], total_degrees, "Total Degree CCDF", fit_power_law=FIT_POWER_LAW)
    fig1.tight_layout()
    
    # --- FIGURE 2: Histograms ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(24, 7))
    plot_degree_histogram(axes2[0], in_degrees, "In-Degree Histogram", "In-Degree (k)")
    plot_degree_histogram(axes2[1], out_degrees, "Out-Degree Histogram", "Out-Degree (k)")
    plot_degree_histogram(axes2[2], total_degrees, "Total Degree Histogram", "Total Degree (k)")
    fig2.tight_layout()
    
    # --- FIGURE 3: Density Scatter ---
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    plot_density_scatter(ax3, fig3, G)
    fig3.tight_layout()
    
    # Display all figures
    plt.show()

    print("\nAnalysis complete.")