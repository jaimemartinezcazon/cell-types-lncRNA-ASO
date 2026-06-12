'''
Author: Jaime Martínez Cazón
Adapted for direct gene names, unparquet edge lists, and unweighted edges.

Description:
This script extracts all input Transcription Factors (TFs), defined as nodes 
with k_in=0, and saves them for downstream analysis. 
It also analyzes the regulatory scope (Total Out-Degree) of two specific 
hardcoded sets of TFs: those associated with logarithmic growth and starvation.
'''

import os
import pandas as pd
import networkx as nx

# =============================================================================
# SETUP: FILE PATHS AND GENE LISTS
# =============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))

EDGE_LIST_PATH = os.path.join(script_dir, "../data/celloracle_data/base_GRN_edge_list.parquet")
OUTPUT_TFS_PATH = os.path.join(script_dir, "../data/TFs_network.csv")

# Specific sets of Transcription Factors to be analyzed
LOG_PHASE_TFS = [
    "YDL056W", "YER111C", "YGR044C", "YIL131C", "YDR146C", "YLR131C", "YNL068C", 
    "YNL199C", "YPL075W", "YLR403W", "YPR104C", "YGL035C", "YMR172W", "YMR016C", 
    "YDR310C", "YCR084C"
]

STARVATION_TFS = [
    "YMR037C", "YGL073W", "YOL116W", "YNL027W", "YOR028C", "YDR310C", 
    "YCR084C", "YDR207C", "YFL031W"
]

# =============================================================================
# DATA LOADING AND TF EXTRACTION
# =============================================================================

def load_data_and_generate_tfs(edge_path, tf_output_path):
    """
    Loads the edge list, generates a list of all network TFs (k_in=0), 
    and saves it to a CSV file for downstream scripts.
    """
    print("Loading network data...")
    if not os.path.exists(edge_path):
        raise FileNotFoundError(f"Edge list not found at: {edge_path}")
    
    edges_df = pd.read_parquet(edge_path)
    
    # Build graph to find all k_in=0 nodes
    G = nx.from_pandas_edgelist(edges_df, source='source', target='target', create_using=nx.DiGraph())
    
    # Identify network TFs (in-degree 0, out-degree > 0)
    network_tfs = [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) > 0]
    
    # Save the TF list for other scripts (like Centrality.py)
    os.makedirs(os.path.dirname(tf_output_path), exist_ok=True)
    pd.DataFrame(network_tfs, columns=["Gene"]).to_csv(tf_output_path, index=False)
    print(f"Extracted {len(network_tfs)} general TFs (k_in=0) and saved to {tf_output_path}")

    return edges_df, set(G.nodes())

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
        print("Analyzing Outgoing Influence of TF Groups...")
        print("="*60)

        print("\nProcessing Set: LogPhase_TFs")
        log_phase_degree = analyze_tf_set_out_degree(LOG_PHASE_TFS, edges_dataframe, network_nodes)
        print(f"  -> Total Out-Degree (Interaction Count): {log_phase_degree}")

        print("\nProcessing Set: Starvation_TFs")
        starvation_degree = analyze_tf_set_out_degree(STARVATION_TFS, edges_dataframe, network_nodes)
        print(f"  -> Total Out-Degree (Interaction Count): {starvation_degree}")
        
        print("\n" + "="*60)
        print("Analysis complete.")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
