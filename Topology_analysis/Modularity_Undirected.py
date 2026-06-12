'''
Author: Jaime Martínez Cazón
Adapted for direct gene names, unparquet edge lists, and NetworkX native Louvain.

Description:
Performs a comprehensive community structure analysis of a Gene Regulatory Network.
1. Detects communities using the native NetworkX Louvain algorithm.
2. Evaluates the statistical significance of global modularity and individual 
   community properties (clustering, conductance) against null models using Multiprocessing.
3. Locates a predefined list of candidate genes within the detected communities.
4. Maps the distribution of each community across the network's bow-tie components.
'''

import os
import glob
import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import community as nx_comm
from tqdm import tqdm
import multiprocessing as mp

# =============================================================================
# SETUP: FILE PATHS AND PARAMETERS
# =============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))

EDGE_LIST_PATH = os.path.join(script_dir, "../data/celloracle_data/base_GRN_edge_list.parquet")
NULL_MODELS_DIR = os.path.join(script_dir, "../data/GRN_data/Null_Models")
BOWTIE_DIR = os.path.join(script_dir, "../data/GRN_data/bow_tie_components")
CANDIDATES_PATH = os.path.join(script_dir, "../data/Candidates_list.csv")
OUTPUT_CSV_PATH = os.path.join(script_dir, "../data/GRN_data/community_significance_analysis.csv")

NUM_CORES = max(1, mp.cpu_count() - 2)

# Default candidates if CSV is not found
DEFAULT_CANDIDATES = [
    "YAL049C", "YBR208C", "YDL182W", "YFL014W", "YGR088W", "YGR180C", 
    "YHL034C", "YJR096W", "YJR137C", "YKL001C", "YLR178C", "YML128C", 
    "YMR105C", "YPL223C", "YPL226W"
]

# =============================================================================
# UTILITY AND ANALYSIS FUNCTIONS
# =============================================================================

def calculate_empirical_p_value(real_value, null_distribution, direction='greater'):
    """Calculates the empirical p-value from a null distribution."""
    n_simulations = len(null_distribution)
    if n_simulations == 0: return np.nan
    
    null_array = np.array(null_distribution)
    count = np.sum(null_array >= real_value) if direction == 'greater' else np.sum(null_array <= real_value)
    return (count + 1) / (n_simulations + 1)


def load_network_and_detect_communities(edge_path):
    """Loads the network and detects communities using the Louvain algorithm."""
    print("Loading network and detecting communities...")
    edge_list = pd.read_parquet(edge_path)
    G_original = nx.from_pandas_edgelist(edge_list, source='source', target='target', create_using=nx.DiGraph())
    G_simple = G_original.to_undirected()
    
    # nx_comm natively returns a list of sets of nodes
    communities_list = nx_comm.louvain_communities(G_simple, seed=42)
    
    # Map to format {node: community_id}
    partition = {node: cid for cid, subset in enumerate(communities_list) for node in subset}
    nodes_by_community = {cid: set(subset) for cid, subset in enumerate(communities_list)}
    
    print(f"Detection complete. Found {len(communities_list)} communities.")
    return G_original, G_simple, partition, nodes_by_community, communities_list


def process_single_null_model(args):
    """
    Worker function to calculate modularity and community properties for a single null model.
    Packed to be used with multiprocessing.Pool.
    """
    file_path, real_nodes_by_comm = args
    try:
        G_null = nx.read_graphml(file_path, node_type=str).to_undirected()
        
        # 1. Global Modularity
        null_comms = nx_comm.louvain_communities(G_null, seed=42)
        modularity = nx_comm.modularity(G_null, null_comms)
        
        # 2. Individual Community Properties (evaluated on the real community structure)
        comm_metrics = {}
        for cid, nodes in real_nodes_by_comm.items():
            subgraph_null = G_null.subgraph(nodes)
            try:
                clust = nx.average_clustering(subgraph_null)
                cond = nx.conductance(G_null, nodes)
            except (nx.NetworkXError, ZeroDivisionError):
                clust, cond = 0.0, 0.0
            
            comm_metrics[cid] = {'clustering': clust, 'conductance': cond}
            
        return {'modularity': modularity, 'comm_metrics': comm_metrics}
    except Exception:
        return None


def locate_candidate_genes(partition, candidates_path):
    """Locates a list of candidate genes within the detected communities."""
    print("\n--- LOCATING CANDIDATE GENES ---")
    try:
        candidates = pd.read_csv(candidates_path).iloc[:, 0].dropna().astype(str).tolist()
    except FileNotFoundError:
        print("Candidate list CSV not found. Using default internal list.")
        candidates = DEFAULT_CANDIDATES

    for gene in candidates:
        community_id = partition.get(gene)
        if community_id is not None:
            print(f"- {gene}: Found in Community {community_id}.")
        else:
            print(f"- {gene}: Not found in the main network component.")


def map_communities_to_bowtie(partition, bowtie_dir):
    """Analyzes the distribution of communities across bow-tie components."""
    print("\n--- MAPPING COMMUNITIES TO BOW-TIE COMPONENTS ---")
    bow_tie_sets = {}
    for component in ['IN', 'SCC', 'OUT', 'OTHERS']:
        filepath = os.path.join(bowtie_dir, f"{component.lower()}_sector_nodes.csv")
        if os.path.exists(filepath):
            bow_tie_sets[component] = set(pd.read_csv(filepath).iloc[:, 0].astype(str))
        else:
            bow_tie_sets[component] = set()

    nodes_by_community = {cid: {n for n, c in partition.items() if c == cid} for cid in set(partition.values())}
    
    analysis_results = []
    for cid, nodes in sorted(nodes_by_community.items()):
        total_nodes = len(nodes)
        if total_nodes == 0: continue
        
        counts = {comp: len(nodes.intersection(s)) for comp, s in bow_tie_sets.items()}
        
        result_row = {'Community_ID': cid, 'Total_Nodes': total_nodes}
        result_row.update({f'% {comp}': (count / total_nodes) * 100 for comp, count in counts.items()})
        analysis_results.append(result_row)
        
    results_df = pd.DataFrame(analysis_results).set_index('Community_ID')
    print("\nDistribution of Communities across Bow-Tie Components (%):")
    print(results_df.to_string(float_format="%.2f"))


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # 1. Load network and detect communities
    G_main, G_simple_main, partition_main, nodes_by_community_main, comms_list_main = load_network_and_detect_communities(EDGE_LIST_PATH)
    
    # Calculate real network metrics
    real_modularity = nx_comm.modularity(G_simple_main, comms_list_main)
    
    # 2. Analyze Null Models in Parallel
    null_model_files = glob.glob(os.path.join(NULL_MODELS_DIR, "*.graphml"))
    
    if not null_model_files:
        print(f"\nWarning: No null models found in {NULL_MODELS_DIR}. Skipping statistical significance tests.")
    else:
        print(f"\nAnalyzing Global Modularity and Community Properties on {len(null_model_files)} null models using {NUM_CORES} cores...")
        
        pool_args = [(f, nodes_by_community_main) for f in null_model_files]
        null_modularity_list = []
        null_metrics_agg = {cid: {'clustering': [], 'conductance': []} for cid in nodes_by_community_main}

        with mp.Pool(processes=NUM_CORES) as pool:
            for result in tqdm(pool.imap_unordered(process_single_null_model, pool_args), total=len(pool_args)):
                if result is not None:
                    null_modularity_list.append(result['modularity'])
                    for cid, metrics in result['comm_metrics'].items():
                        null_metrics_agg[cid]['clustering'].append(metrics['clustering'])
                        null_metrics_agg[cid]['conductance'].append(metrics['conductance'])

        # --- GLOBAL MODULARITY SIGNIFICANCE ---
        print("\n--- GLOBAL MODULARITY SIGNIFICANCE ANALYSIS ---")
        mean_modularity = np.mean(null_modularity_list)
        std_modularity = np.std(null_modularity_list)
        p_value_mod = calculate_empirical_p_value(real_modularity, null_modularity_list, 'greater')
        
        print(f"Real Network Modularity:         {real_modularity:.4f}")
        print(f"Null Model Modularity (Mean±SD): {mean_modularity:.4f} ± {std_modularity:.4f}")
        print(f"Empirical P-value:               {p_value_mod:.4f}")

        # --- INDIVIDUAL COMMUNITY SIGNIFICANCE ---
        print("\n--- INDIVIDUAL COMMUNITY SIGNIFICANCE ANALYSIS ---")
        community_stats = []
        for cid, nodes in sorted(nodes_by_community_main.items()):
            subgraph_real = G_simple_main.subgraph(nodes)
            
            real_clustering = nx.average_clustering(subgraph_real)
            try:
                real_conductance = nx.conductance(G_simple_main, nodes)
            except (nx.NetworkXError, ZeroDivisionError):
                real_conductance = 0.0
            
            p_val_clust = calculate_empirical_p_value(real_clustering, null_metrics_agg[cid]['clustering'], 'greater')
            p_val_cond = calculate_empirical_p_value(real_conductance, null_metrics_agg[cid]['conductance'], 'less')

            stats = {
                'Community_ID': cid, 
                'Num_Nodes': len(nodes), 
                'Density': nx.density(subgraph_real),
                'Clustering_Real': real_clustering, 
                'Clustering_PValue': p_val_clust,
                'Conductance_Real': real_conductance, 
                'Conductance_PValue': p_val_cond,
            }
            community_stats.append(stats)

        summary_df = pd.DataFrame(community_stats).set_index('Community_ID')
        print("\nSummary of Community Properties and Significance:")
        print(summary_df.to_string(float_format="%.4f"))
        
        os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
        summary_df.to_csv(OUTPUT_CSV_PATH)
        print(f"\nDetailed summary saved to '{OUTPUT_CSV_PATH}'")

    # 3. Locate specific genes of interest
    locate_candidate_genes(partition_main, CANDIDATES_PATH)
    
    # 4. Map communities to the bow-tie structure
    map_communities_to_bowtie(partition_main, BOWTIE_DIR)
    
    print("\n\nCommunity analysis complete.")