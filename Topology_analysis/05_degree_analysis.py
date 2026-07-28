'''
Author: Jaime Martínez Cazón
Adapted for direct gene names as strings and unweighted edges.

Description:
This script performs a comprehensive topological analysis of a 
Gene Regulatory Network. It covers several key aspects of the network's 
structure:
1.  Global degree distributions (in-degree and out-degree) and power-law fits.
2.  Analysis of the Strongly Connected Component (SCC), including the degree
    properties of its internal and external connections.
3.  Degree distribution analysis for each of the main bow-tie components 
    (IN, OUT, SCC, OTHERS).
'''

import os
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import powerlaw
from pathlib import Path

# =============================================================================
# SETUP: FILE PATHS
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
BOWTIE_COMPONENTS_DIR = INPUT_DATA_DIR / "bow_tie"

# =============================================================================
# DATA LOADING
# =============================================================================

def load_network_data():
    """
    Loads the network edge list.

    Returns:
        nx.DiGraph: The directed network graph.
    """
    print("Loading network data...")
    edge_list = pd.read_parquet(EDGE_LIST_PATH)
    
    G_full = nx.from_pandas_edgelist(
        edge_list,
        source='source',
        target='target',
        create_using=nx.DiGraph()
    )

    ## Only work with main WCC
    giant_component_nodes = max(nx.weakly_connected_components(G_full), key=len)
    G = G_full.subgraph(giant_component_nodes).copy()

    ## Eliminate self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    print(f"Network data loaded successfully. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_global_properties(G):
    """
    Calculates and prints basic global properties of the network.
    """
    print("\n--- GLOBAL NETWORK PROPERTIES ---")
    avg_in_degree = np.mean([d for _, d in G.in_degree()])
    avg_out_degree = np.mean([d for _, d in G.out_degree()])
    print(f"Average in-degree: {avg_in_degree:.2f}")
    print(f"Average out-degree: {avg_out_degree:.2f}")
    print("-" * 35)

def plot_global_degree_distributions(G, fit_powerlaw=False):
    """
    Plots the global in-degree and out-degree distributions.
    Optional power-law fit.
    """
    print("\nPlotting global degree distributions...")
    degree_in = dict(G.in_degree())
    degree_out = dict(G.out_degree())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    def _plot_dist(ax, degree_dict, degree_type):
        data = np.array([d for d in degree_dict.values() if d > 0])
        
        if len(data) == 0:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            return

        # Calculate empirical probability
        unique, counts = np.unique(data, return_counts=True)
        probabilities = counts / counts.sum()
        
        # Plot points with transparency (alpha)
        ax.scatter(unique, probabilities, color='black', alpha=0.4, marker='o', label="Data")
        
        if fit_powerlaw:
            fit = powerlaw.Fit(data, discrete=True, xmin=1, verbose=False)
            fit.power_law.plot_pdf(ax=ax, linestyle='-', color='r', label=f'Fit (γ={fit.alpha:.2f})')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f"{degree_type}-Degree Distribution", fontsize=18, fontweight='bold')
        ax.set_xlabel("Degree (k)", fontsize=14)
        ax.set_ylabel("P(k)", fontsize=14)
        ax.legend(fontsize=12)

    _plot_dist(axes[0], degree_in, "In")
    _plot_dist(axes[1], degree_out, "Out")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "degree_distribution_plot_global.pdf")


def analyze_scc_properties(G):
    """
    Analyzes the properties of the largest Strongly Connected Component (SCC).
    """
    print("\nAnalyzing Strongly Connected Component (SCC)...")
    largest_scc_nodes = max(nx.strongly_connected_components(G), key=len)
    G_scc = G.subgraph(largest_scc_nodes).copy()
    
    nodes_sorted = sorted(G_scc.nodes(), key=lambda n: G.out_degree(n), reverse=True)
    
    # Select only the top 50 nodes to avoid matplotlib freezing
    top_n = 50
    if len(nodes_sorted) > top_n:
        print(f"SCC has {len(nodes_sorted)} nodes. Plotting only the top {top_n} by out-degree.")
        nodes_sorted = nodes_sorted[:top_n]
        
    gene_names_sorted = list(nodes_sorted) 
    
    data = {
        'Global In': [G.in_degree(n) for n in nodes_sorted],
        'Global Out': [G.out_degree(n) for n in nodes_sorted],
        'Internal In': [G_scc.in_degree(n) for n in nodes_sorted],
        'Internal Out': [G_scc.out_degree(n) for n in nodes_sorted]
    }
    
    def _plot_scc_bars(ax, title, in_data, out_data):
        x = np.arange(len(nodes_sorted))
        width = 0.4
        ax.bar(x - width/2, in_data, width, label='In-Degree')
        ax.bar(x + width/2, out_data, width, label='Out-Degree')
        ax.set_title(title, fontsize=20, fontweight='bold')
        ax.set_ylabel("Degree", fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(gene_names_sorted, rotation=90, fontsize=10)
        ax.legend(fontsize=14)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    _plot_scc_bars(axes[0], f"SCC Global Connections (Top {top_n})", data['Global In'], data['Global Out'])
    _plot_scc_bars(axes[1], f"SCC Internal Connections (Top {top_n})", data['Internal In'], data['Internal Out'])
    plt.tight_layout()
    plt.savefig(FIG_DIR / "SCC_connection_plot.pdf")

def analyze_bowtie_components(G, fit_powerlaw=False):
    """
    Analyzes the degree distributions for each bow-tie component.
    """
    print("\nAnalyzing degree distributions per bow-tie component...")
    
    def _plot_dist(ax, data, title):
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(title, fontsize=18, fontweight='bold')
        ax.set_xlabel("Degree (k)", fontsize=14)
        ax.set_ylabel("P(k)", fontsize=14)
        
        if len(data) > 0:
            unique, counts = np.unique(data, return_counts=True)
            probabilities = counts / counts.sum()
            ax.scatter(unique, probabilities, color='black', alpha=0.4, marker='o', label='Data')
            
            if fit_powerlaw and len(data) > 1:
                fit = powerlaw.Fit(data, discrete=True, xmin=1, verbose=False)
                fit.power_law.plot_pdf(ax=ax, linestyle='-', color='r', label=f'Fit (γ={fit.alpha:.2f})')
                
            ax.legend(fontsize=12)
        else:
            ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center', transform=ax.transAxes)

    components = {'IN': 'in_sector_nodes.parquet', 'SCC': 'scc_nodes.parquet',
                  'OUT': 'out_sector_nodes.parquet', 'OTHERS': 'others_nodes.parquet'}

    for comp_name, filename in components.items():
        filepath = os.path.join(BOWTIE_COMPONENTS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Warning: Component file not found, skipping: {filepath}")
            continue
            
        ids = pd.read_parquet(filepath).iloc[:, 0].tolist()
        
        deg_in = np.array([G.in_degree(n) for n in ids if n in G and G.in_degree(n) > 0])
        deg_out = np.array([G.out_degree(n) for n in ids if n in G and G.out_degree(n) > 0])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Degree Distribution for {comp_name} Component", fontsize=22, fontweight='bold')
        _plot_dist(axes[0], deg_in, "In-Degree Distribution")
        _plot_dist(axes[1], deg_out, "Out-Degree Distribution")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(FIG_DIR / f"degree_distribution_plot_{comp_name}.pdf")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    G_main = load_network_data()
    
    analyze_global_properties(G_main)
    plot_global_degree_distributions(G_main)
    analyze_scc_properties(G_main)
    analyze_bowtie_components(G_main)
    
    print("\nTopological analysis complete.")