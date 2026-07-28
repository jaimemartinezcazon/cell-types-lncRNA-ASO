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

# Null models data path (updated for parquet)
NULL_MODELS_DIR     = INPUT_DATA_DIR / "null_models"
BOWTIE_DIR = INPUT_DATA_DIR / "bow_tie"
OUTPUT_SIG_PATH = OUTPUT_DATA_DIR / "community_significance_undirected.csv"

# Try with CPUs assigned by SLURM. If not use maximum 8 CPUs. 
slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
if slurm_cpus is not None:
    NUM_CORES = int(slurm_cpus)
else:
    NUM_CORES = min(8, mp.cpu_count())
    
# Default candidates if CSV is not found
GEN_SET = [
    "SOX2"
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
    G_full = nx.from_pandas_edgelist(edge_list, source='source', target='target', create_using=nx.DiGraph())
 
    ## Only work with main WCC
    giant_component_nodes = max(nx.weakly_connected_components(G_full), key=len)
    G_original= G_full.subgraph(giant_component_nodes).copy()

    ## Eliminate self-loops
    G_original.remove_edges_from(nx.selfloop_edges(G_original))

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
        # Load edge list from parquet and convert to undirected networkx graph
        edges = pd.read_parquet(file_path)
        G_null = nx.from_pandas_edgelist(
            edges, source='source', target='target', create_using=nx.DiGraph()
        ).to_undirected()

        ## Eliminate self-loops
        G_null.remove_edges_from(nx.selfloop_edges(G_null))

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


#def locate_candidate_genes(partition, candidates):
#    """Locates a list of candidate genes within the detected communities."""
#    print("\n--- LOCATING CANDIDATE GENES ---")
#
#    for gene in candidates:
#        community_id = partition.get(gene)
#        if community_id is not None:
#            print(f"- {gene}: Found in Community {community_id}.")
#        else:
#            print(f"- {gene}: Not found in the main network component.")


def map_communities_to_bowtie(partition, bowtie_dir):
    """Analyzes the distribution of communities across bow-tie components."""
    print("\n--- MAPPING COMMUNITIES TO BOW-TIE COMPONENTS ---")
    bow_tie_sets = {}
    for comp in ['IN', 'SCC', 'OUT', 'OTHERS']:
        # 1. Cambiamos la extensión buscando un .parquet
        filepath = os.path.join(bowtie_dir, f"{comp.lower()}_sector_nodes.parquet")
        
        if os.path.exists(filepath):
            # 2. Usamos pd.read_parquet en lugar de read_csv
            # iloc[:, 0] asegura que tomamos la primera columna sin importar su nombre ('Node')
            bow_tie_sets[comp] = set(pd.read_parquet(filepath).iloc[:, 0].astype(str))
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
        
    return analysis_results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # 1. Load network and detect communities
    G_main, G_simple_main, partition_main, nodes_by_community_main, comms_list_main = load_network_and_detect_communities(EDGE_LIST_PATH)
    
    # Calculate real network metrics
    real_modularity = nx_comm.modularity(G_simple_main, comms_list_main)
    
    # 2. Analyze Null Models in Parallel
    # Updated to search for .parquet files
    null_model_files = glob.glob(os.path.join(NULL_MODELS_DIR, "*.parquet"))
    
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
        
        summary_df.to_csv(OUTPUT_SIG_PATH)
        print(f"\nDetailed summary saved to '{OUTPUT_SIG_PATH}'")

    # 3. Locate specific genes of interest
    #locate_candidate_genes(partition_main, GEN_SET)
    
    # 4. Map communities to the bow-tie structure
    map_communities_to_bowtie(partition_main, BOWTIE_DIR)
    
    print("\n\nCommunity analysis complete.")