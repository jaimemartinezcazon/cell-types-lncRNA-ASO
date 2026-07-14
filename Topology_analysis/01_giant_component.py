'''
Author: Jaime Martínez Cazón
Adapted for direct gene names as strings and unweighted edges.

Description:
This script performs a pre-processing step on the raw S. cerevisiae Gene
Regulatory Network. Its main purpose is to identify and extract the largest
weakly connected component (the "giant component") of the network.

The workflow includes:
1.  Loading the full network from the original edge list.
2.  Analyzing the initial connectivity of the network (number of weakly and
    strongly connected components).
3.  Extracting the subgraph corresponding to the giant component.
4.  Saving new edge list and node list files that contain only the data for
    this giant component, ensuring subsequent analyses are performed on a
    single, connected network structure.
'''

import pandas as pd
import networkx as nx
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

# Output paths for the filtered giant component data
GC_EDGE_LIST_PATH = OUTPUT_DATA_DIR / "giantC_edge_list.parquet"

# =============================================================================
# ANALYSIS AND PROCESSING FUNCTIONS
# =============================================================================

def load_and_analyze_components(edge_path):
    """
    Loads the full directed graph and analyzes its initial component structure.

    Args:
        edge_path (str): The path to the raw edge list file.

    Returns:
        nx.DiGraph: The loaded full directed graph.
    """
    print("--- Initial Network Analysis ---")
    try:
        edge_list = pd.read_parquet(edge_path)
    except FileNotFoundError:
        print(f"Error: Raw edge list file not found at '{edge_path}'")
        return None

    G = nx.from_pandas_edgelist(
        edge_list, 'source', 'target', create_using=nx.DiGraph()
    )
    
    ## Eliminate self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    print(f"Full network loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    
    # Analyze connectivity
    num_wcc = nx.number_weakly_connected_components(G)
    num_scc = nx.number_strongly_connected_components(G)
    print(f"Number of weakly connected components: {num_wcc}")
    print(f"Number of strongly connected components: {num_scc}")
    
    # Print sizes of the largest weakly connected components
    wcc_sorted = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    print("\nSizes of the largest weakly connected components:")
    for i, component in enumerate(wcc_sorted[:5]): # Show top 5
        print(f"  Component {i+1}: {len(component)} nodes")
        
    return G

def extract_and_save_giant_component(G, raw_edge_list, gc_edge_path):
    """
    Extracts the giant component and saves the filtered data to new files.

    Args:
        G (nx.DiGraph): The full network graph.
        raw_edge_list (pd.DataFrame): The original DataFrame of edges.
        gc_edge_path (str): The output path for the giant component edge list.
    """
    print("\n--- Extracting Giant Component ---")
    
    # Find the largest weakly connected component (the giant component)
    giant_component_nodes = max(nx.weakly_connected_components(G), key=len)
    G_giant = G.subgraph(giant_component_nodes).copy()
    
    print(f"Giant component extracted: {G_giant.number_of_nodes()} nodes, {G_giant.number_of_edges()} edges.")
    
    # Filter the original DataFrames to keep only nodes in the giant component
    filtered_edges = raw_edge_list[
        raw_edge_list["source"].isin(giant_component_nodes) & 
        raw_edge_list["target"].isin(giant_component_nodes)
    ]
        
    # Save the filtered data to new Parquet files
    try:
        filtered_edges.to_parquet(gc_edge_path, index=False)
        print(f"Giant component edge list saved to: '{gc_edge_path}'")
        
    except Exception as e:
        print(f"Error saving filtered files: {e}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # 1. Load the full network and perform an initial analysis
    full_graph = load_and_analyze_components(EDGE_LIST_PATH)
    
    if full_graph:
        # 2. Load the raw data files as DataFrames
        try:
            edge_list_df = pd.read_parquet(EDGE_LIST_PATH)
        
            # 3. Extract the giant component and save the new, filtered files
            extract_and_save_giant_component(
                full_graph, 
                edge_list_df, 
                GC_EDGE_LIST_PATH
            )
        except FileNotFoundError:
            print(f"Error: Could not find raw data files to create giant component subset.")
            
    print("\nPreprocessing complete.")

