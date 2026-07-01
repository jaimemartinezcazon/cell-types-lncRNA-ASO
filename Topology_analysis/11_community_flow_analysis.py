'''
Author: Jaime Martínez Cazón
Adapted for direct gene names, unparquet edge lists, and NetworkX native Louvain.

Description:
Analyzes the directional nature of inter-community connections in a Gene 
Regulatory Network. Detects communities using the undirected Louvain algorithm, 
then calculates the "outward flow percentage" for each community (proportion 
of external connections that are outgoing) to identify source/sink dynamics.
'''

import pandas as pd
import networkx as nx
from networkx.algorithms import community as nx_comm
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


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def load_and_partition_network(edge_path):
    """
    Loads the network, detects communities on its undirected version, 
    and returns the directed graph along with the community partition.
    """
    print("Step 1: Loading data and detecting communities...")
    edge_list = pd.read_parquet(edge_path)
    G_directed = nx.from_pandas_edgelist(
        edge_list,
        source='source',
        target='target',
        create_using=nx.DiGraph()
    )
    
    # Community detection on the undirected version
    G_undirected = G_directed.to_undirected()
    
    # nx_comm.louvain_communities natively returns a list of sets of nodes
    communities = nx_comm.louvain_communities(G_undirected, seed=42)
        
    print(f"Found {len(communities)} communities.")
    return G_directed, communities

def analyze_outward_flow(G_directed, communities):
    """Calculates the outward flow percentage for each detected community."""
    print("\nStep 2: Calculating outward flow percentage for each community...")
    
    # Using a quotient graph to count inter-community edges.
    G_meta = nx.quotient_graph(G_directed, communities, create_using=nx.MultiDiGraph)
    
    analysis_results = []
    for cid, node_set in enumerate(communities):
        # The node in the meta-graph corresponds to the frozenset of its nodes.
        meta_node = frozenset(node_set)
        
        total_out_links = sum(d.get('weight', 1) for _, _, d in G_meta.out_edges(meta_node, data=True))
        total_in_links = sum(d.get('weight', 1) for _, _, d in G_meta.in_edges(meta_node, data=True))
        
        total_external_links = total_out_links + total_in_links
        
        if total_external_links > 0:
            perc_outward_flow = (total_out_links / total_external_links) * 100
        else:
            perc_outward_flow = 0.0
            
        analysis_results.append({
            "Community_ID": cid,
            "Total_Nodes": len(node_set),
            "Outward_Flow_Percentage": perc_outward_flow,
            "External_Outgoing_Links": total_out_links,
            "External_Incoming_Links": total_in_links,
        })
        
    return pd.DataFrame(analysis_results)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    directed_graph, community_list = load_and_partition_network(EDGE_LIST_PATH)
    
    if directed_graph:
        results_df = analyze_outward_flow(directed_graph, community_list)
        
        print("\n--- OUTWARD FLOW PERCENTAGE BY COMMUNITY ---")
        print("(Percentage of external connections that are outgoing)")
        
        results_df_sorted = results_df.sort_values(by="Outward_Flow_Percentage", ascending=False).reset_index(drop=True)
        results_df_sorted['Outward_Flow_Percentage'] = results_df_sorted['Outward_Flow_Percentage'].map('{:,.2f}%'.format)
        
        print(results_df_sorted.to_string(index=False))