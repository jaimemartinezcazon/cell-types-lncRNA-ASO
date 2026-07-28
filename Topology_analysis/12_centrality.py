'''
Author: Jaime Martínez Cazón
Adapted for direct gene names and unparquet edge lists.

Description:
Performs a comprehensive centrality analysis on a Gene Regulatory Network.
Calculates In/Out Degree, Betweenness, In/Out Closeness, and PageRank.
Identifies the top N most central nodes and cross-references them with 
functional annotations (TFs, candidates, bow-tie structure).
'''

##
## NOTE: We are not using candidates here (set of genes that we want to specifically analyze), necessary to modify and include a list if necessary.
##

import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from collections import defaultdict
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

BOWTIE_DIR = INPUT_DATA_DIR / "bow_tie"

# Number of nodes with centrality information
N_TOP = 10  

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_network_and_annotations():
    """Loads the main network graph and all annotation files."""
    print("Loading all required data...")
    
    edge_list = pd.read_parquet(EDGE_LIST_PATH)
    print("Loading real network...")
    edge_list = pd.read_parquet(EDGE_LIST_PATH)

    G_full = nx.from_pandas_edgelist(
        edge_list, source='source', target='target', create_using=nx.DiGraph()
    )
    ## Only work with main WCC
    giant_component_nodes = max(nx.weakly_connected_components(G_full), key=len)
    G = G_full.subgraph(giant_component_nodes).copy()
    
    ## Eliminate self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    try:
        candidates_path = INPUT_DATA_DIR / "candidates_list.parquet"
        candidates_genes = set(pd.read_parquet(candidates_path).iloc[:, 0].dropna().astype(str))
    except FileNotFoundError:
        print("Warning: Candidate list file not found. Skipping candidate analysis.")
        candidates_genes = set()

    try:
        tfs_path = INPUT_DATA_DIR / "input_tfs.parquet"
        tfs_genes = set(pd.read_parquet(tfs_path).iloc[:, 0].dropna().astype(str))
    except FileNotFoundError:
        print("Warning: TF list file not found. Skipping TF analysis.")
        tfs_genes = set()

    node_sector_map = {}
    for sector in ['IN', 'SCC', 'OUT', 'OTHERS']:
        try:
            sector_path = BOWTIE_DIR / f"{sector.lower()}_sector_nodes.parquet"
            sector_nodes = pd.read_parquet(sector_path).iloc[:, 0].dropna().astype(str)
            for node in sector_nodes:
                node_sector_map[node] = sector
        except FileNotFoundError:
            print(f"Warning: Bow-tie file for '{sector}' not found.")

    print(f"Data loading complete. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G, candidates_genes, tfs_genes, node_sector_map

# =============================================================================
# CENTRALITY CALCULATION
# =============================================================================

def calculate_centrality_measures(G):
    """Calculates all centrality measures and returns them in a DataFrame."""
    print("\nCalculating centrality measures...")
    
    nodes = list(G.nodes())
    centrality_df = pd.DataFrame({"Gene_Name": nodes})

    print("  - Degree (In/Out)...")
    centrality_df["In_Degree"] = centrality_df["Gene_Name"].map(dict(G.in_degree())).fillna(0).astype(int)
    centrality_df["Out_Degree"] = centrality_df["Gene_Name"].map(dict(G.out_degree())).fillna(0).astype(int)

    print("  - Betweenness Centrality (This may take a while for large networks)...")
    betweenness = nx.betweenness_centrality(G, weight=None)
    centrality_df["Betweenness"] = centrality_df["Gene_Name"].map(betweenness).fillna(0)

    print("  - Closeness Centrality (In/Out)...")
    closeness_out = nx.closeness_centrality(G)
    closeness_in = nx.closeness_centrality(G.reverse()) 
    centrality_df["Closeness_Out"] = centrality_df["Gene_Name"].map(closeness_out).fillna(0)
    centrality_df["Closeness_In"] = centrality_df["Gene_Name"].map(closeness_in).fillna(0)

    print("  - PageRank Centrality...")
    pagerank = nx.pagerank(G, weight=None)
    centrality_df["PageRank"] = centrality_df["Gene_Name"].map(pagerank).fillna(0)
    
    print("Centrality calculations complete.")
    return centrality_df

# =============================================================================
# TOP NODE ANALYSIS
# =============================================================================

def analyze_top_nodes(centrality_df, measure, annotations):
    """Analyzes and prints a report for the top N nodes for a given measure."""
    candidates_genes, tfs_genes, node_sector_map = annotations
    
    print(f"\n--- Analysis for Top {N_TOP} nodes by {measure} ---")
    
    top_nodes = centrality_df.sort_values(by=measure, ascending=False).head(N_TOP)
    
    print(f"{'Gene Name':<15} | {measure:<15} | {'Sector':<8} | {'Candidate':<9} | {'TF':<3}")
    print("-" * 65)
    
    sector_counts = defaultdict(int)
    
    for _, row in top_nodes.iterrows():
        gene_name = row["Gene_Name"]
        score = row[measure]
        
        sector = node_sector_map.get(gene_name, "Unknown")
        is_candidate = "Yes" if gene_name in candidates_genes else "No"
        is_tf = "Yes" if gene_name in tfs_genes else "No"
        
        sector_counts[sector] += 1
        
        score_str = f"{score:.6f}" if isinstance(score, float) else str(score)
        print(f"{gene_name:<15} | {score_str:<15} | {sector:<8} | {is_candidate:<9} | {is_tf:<3}")
        
    print("-" * 65)
    print("Summary for this group:")
    for sector, count in sorted(sector_counts.items()):
        print(f"  - Sector '{sector}': {count}/{N_TOP} nodes")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    try:
        G_main, candidates, tfs, sector_map = load_network_and_annotations()
        annotations_tuple = (candidates, tfs, sector_map)
    except Exception as e:
        print(f"Fatal Error during data loading: {e}")
        exit()

    centrality_results_df = calculate_centrality_measures(G_main)
    
    output_path = OUTPUT_DATA_DIR / "full_centrality_analysis.parquet"
    centrality_results_df.to_parquet(output_path, index=False)
    print(f"\nFull centrality data saved to: {output_path}")

    centrality_measures = [
        "In_Degree", "Out_Degree", "Betweenness", 
        "Closeness_In", "Closeness_Out", "PageRank"
    ]
    
    for metric in centrality_measures:
        analyze_top_nodes(centrality_results_df, metric, annotations_tuple)

    print("\nCentrality analysis complete.")