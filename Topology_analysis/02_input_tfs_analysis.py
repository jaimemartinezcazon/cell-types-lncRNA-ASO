'''
Author: Jaime Martínez Cazón
Adapted for direct gene names, unparquet edge lists, and unweighted edges.

Description:
This script extracts all input Transcription Factors (TFs), defined as nodes 
with k_in=0, and saves them for downstream analysis. 
It also analyzes the regulatory scope (Total Out-Degree) of two specific 
hardcoded sets of TFs: those associated with logarithmic growth and starvation.
'''

from pathlib import Path
import pandas as pd
import networkx as nx

# =============================================================================
# SETUP: FILE PATHS AND GENE LISTS
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
OUTPUT_TFS_PATH = OUTPUT_DATA_DIR / "input_tfs.parquet"


# =============================================================================
# DATA LOADING AND TF EXTRACTION
# =============================================================================

def load_data_and_generate_tfs(edge_path, tf_output_path):
    """
    Loads the edge list, generates a list of all network TFs (k_in=0), 
    and saves it to a PARQUET file for downstream scripts.
    """
    print("Loading network data...")
    
    edges_df = pd.read_parquet(edge_path)
    
    # Build graph to find all k_in=0 nodes
    G_full = nx.from_pandas_edgelist(edges_df, source='source', target='target', create_using=nx.DiGraph())
    
    ## Only work with main WCC
    giant_component_nodes = max(nx.weakly_connected_components(G_full), key=len)
    G = G_full.subgraph(giant_component_nodes).copy()

    ## Eliminate self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # Identify network TFs (in-degree 0, out-degree > 0)
    network_tfs = [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) > 0]
    
    # Save the TF list for other scripts (like Centrality.py)
    pd.DataFrame(network_tfs, columns=["Gene"]).to_parquet(tf_output_path, index=False)
    print(f"Extracted {len(network_tfs)} general TFs (k_in=0) and saved to {tf_output_path}")

    return edges_df, set(G.nodes())

# NOT USED NOW: give a list of important TFs ( [..., ..., ] )to use:
def analyze_tf_set_out_degree(tf_gene_list, edges_df, all_nodes):
    """Calculates the total out-degree for a specific set of TFs."""
    missing_genes = [g for g in tf_gene_list if g not in all_nodes]
    if missing_genes:
        print(f"    -> Warning: These genes were not found in the network and will be ignored: {missing_genes}")

    # Filter the edge list to find all outgoing interactions from the TF set
    outgoing_edges_df = edges_df[edges_df['source'].isin(tf_gene_list)]
    
    # Metric: Total out-degree (interaction count)
    total_out_degree = len(outgoing_edges_df)
    
    return total_out_degree

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    try:
        edges_dataframe, network_nodes = load_data_and_generate_tfs(EDGE_LIST_PATH, OUTPUT_TFS_PATH)
        
        print("\n" + "="*60)
        print("Analysis complete.")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
