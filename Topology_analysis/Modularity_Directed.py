'''
Author: Jaime Martínez Cazón
Adapted for direct gene names, unparquet edge lists, and NetworkX directed Louvain.

Description:
Performs a directed community structure analysis on a Gene Regulatory Network.
1. Detects communities using the directed Louvain algorithm.
2. Evaluates the statistical significance of directed modularity and community 
   properties against a null model ensemble using Multiprocessing.
3. Locates candidate genes within the detected directed communities.
4. Maps the community distribution across the network's bow-tie structure.
'''

import os
import glob
import pandas as pd
import numpy as np
import networkx as nx
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

OUTPUT_DIR = os.path.join(script_dir, "../data/GRN_data")
OUTPUT_SIG_PATH = os.path.join(OUTPUT_DIR, "community_significance_directed.csv")
OUTPUT_BOWTIE_PATH = os.path.join(OUTPUT_DIR, "community_bowtie_distribution_directed.csv")

NUM_CORES = max(1, mp.cpu_count() - 2)

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


def load_network_and_detect_directed_communities(edge_path):
    """Loads the directed network and detects communities."""
    print("Loading directed network and detecting communities...")
    edge_list = pd.read_parquet(edge_path)
    G = nx.from_pandas_edgelist(edge_list, source='source', target='target', create_using=nx.DiGraph())
    
    communities_list = nx.community.louvain_communities(G, seed=42)
    
    partition = {node: i for i, comm in enumerate(communities_list) for node in comm}
    nodes_by_community = {cid: set(comm) for cid, comm in enumerate(communities_list)}
    
    print(f"Detection complete. Found {len(communities_list)} directed communities.")
    return G, communities_list, partition, nodes_by_community


def process_single_null_model(args):
    """
    Worker function to calculate directed modularity and community properties for a single null model.
    """
    file_path, real_nodes_by_comm = args
    try:
        G_null = nx.read_graphml(file_path, node_type=str)
        
        # 1. Directed Modularity
        null_comms = nx.community.louvain_communities(G_null, seed=42)
        modularity = nx.community.modularity(G_null, null_comms)
        
        # 2. Individual Community Properties
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
        community_id = partition.get(gene, 'N/A (Isolated or not in network)')
        print(f"- {gene}: Found in Community {community_id}.")


def map_communities_to_bowtie(partition, bowtie_dir):
    """Analyzes the distribution of communities across bow-tie components."""
    print("\n--- MAPPING COMMUNITIES TO BOW-TIE COMPONENTS ---")
    bow_tie_sets = {}
    for comp in ['IN', 'SCC', 'OUT', 'OTHERS']:
        filepath = os.path.join(bowtie_dir, f"{comp.lower()}_sector_nodes.csv")
        if os.path.exists(filepath):
            bow_tie_sets[comp] = set(pd.read_csv(filepath).iloc[:, 0].astype(str))
        else:
            bow_tie_sets[comp] = set()

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
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_df.to_csv(OUTPUT_BOWTIE_PATH)
    print(f"\nBow-tie distribution report saved to '{OUTPUT_BOWTIE_PATH}'")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # 1. Load directed network and detect directed communities
    G_main, communities_main, partition_main, nodes_by_community_main = load_network_and_detect_directed_communities(EDGE_LIST_PATH)
    real_modularity = nx.community.modularity(G_main, communities_main)
    
    # 2. Analyze Null Models in Parallel
    null_model_files = glob.glob(os.path.join(NULL_MODELS_DIR, "*.graphml"))
    
    if not null_model_files:
        print(f"\nWarning: No null models found in {NULL_MODELS_DIR}. Skipping statistical significance tests.")
    else:
        print(f"\nAnalyzing Directed Modularity and Community Properties on {len(null_model_files)} null models using {NUM_CORES} cores...")
        
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

        # --- DIRECTED MODULARITY SIGNIFICANCE ---
        print("\n--- DIRECTED MODULARITY SIGNIFICANCE ANALYSIS ---")
        mean_modularity = np.mean(null_modularity_list)
        std_modularity = np.std(null_modularity_list)
        p_value_mod = calculate_empirical_p_value(real_modularity, null_modularity_list, 'greater')
        
        print(f"Real Network Directed Modularity:  {real_modularity:.4f}")
        print(f"Null Model Modularity (Mean±SD):  {mean_modularity:.4f} ± {std_modularity:.4f}")
        print(f"Empirical P-value:                 {p_value_mod:.4f}")

        # --- INDIVIDUAL COMMUNITY SIGNIFICANCE ---
        print("\n--- INDIVIDUAL DIRECTED COMMUNITY ANALYSIS ---")
        community_stats = []
        for cid, nodes in sorted(nodes_by_community_main.items()):
            subgraph_real = G_main.subgraph(nodes)
            
            real_clustering = nx.average_clustering(subgraph_real)
            try:
                real_conductance = nx.conductance(G_main, nodes)
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
        print("\nSummary of Directed Community Properties and Significance:")
        print(summary_df.to_string(float_format="%.4f"))
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        summary_df.to_csv(OUTPUT_SIG_PATH)
        print(f"\nDetailed summary saved to '{OUTPUT_SIG_PATH}'")

    # 3. Locate specific genes of interest
    locate_candidate_genes(partition_main, CANDIDATES_PATH)
    
    # 4. Map communities to the bow-tie structure
    map_communities_to_bowtie(partition_main, BOWTIE_DIR)
    
    print("\n\nDirected community analysis complete.")