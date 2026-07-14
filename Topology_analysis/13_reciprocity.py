'''
Author: Jaime Martínez Cazón
Adapted for direct gene names and unparquet edge lists.

Description:
Assesses the global reciprocity of a Gene Regulatory Network. 
Calculates the reciprocity of the real network and compares it against 
an ensemble of surrogate networks (null models) to determine statistical 
significance using a two-tailed empirical p-value.
'''

import os
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
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

# The directory name is kept as requested
NULL_MODELS_DIR     = INPUT_DATA_DIR / "null_models"

N_NULL_MODELS = 1000  

# =============================================================================
# UTILITY AND ANALYSIS FUNCTIONS
# =============================================================================

def calculate_two_tailed_p_value(real_value, null_distribution):
    """Calculates the two-tailed empirical p-value from a null distribution."""
    n_simulations = len(null_distribution)
    if n_simulations == 0:
        return np.nan

    null_array = np.array(null_distribution)
    null_mean = np.mean(null_array)
    
    observed_deviation = abs(real_value - null_mean)
    null_deviations = abs(null_array - null_mean)
    
    count_as_extreme = np.sum(null_deviations >= observed_deviation)
    p_value = (count_as_extreme + 1) / (n_simulations + 1)
    return p_value


def load_real_network(filepath):
    """Loads the real network and extracts its giant component."""
    print(f"Loading real network from '{filepath}'...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Real network file not found at: {filepath}")
        
    edge_list = pd.read_parquet(filepath)
    G_full = nx.from_pandas_edgelist(
        edge_list, 'source', 'target', create_using=nx.DiGraph()
    )
    
    ## Only work with main WCC
    giant_component_nodes = max(nx.weakly_connected_components(G_full), key=len)
    G_real = G_full.subgraph(giant_component_nodes).copy()
    
    ## Eliminate self-loops
    G_real.remove_edges_from(nx.selfloop_edges(G_real))

    print(f"Real network (giant component) loaded: {G_real.number_of_nodes()} nodes.")
    return G_real


def analyze_null_models_reciprocity(null_dir, num_models):
    """Loads and analyzes the ensemble of null models to calculate reciprocity."""
    print(f"\nAnalyzing up to {num_models} null models for reciprocity...")
    null_reciprocity_list = []
    
    for i in tqdm(range(num_models), desc="Analyzing Null Models"):
        # Updated to search for .parquet files
        file_path = os.path.join(null_dir, f"null_model_{str(i).zfill(4)}.parquet")
        if not os.path.exists(file_path):
            continue
            
        try:
            # Load edge list from parquet and convert to directed networkx graph
            edges = pd.read_parquet(file_path)
            G_null = nx.from_pandas_edgelist(
                edges, source='source', target='target', create_using=nx.DiGraph()
            )
            
            if G_null.number_of_nodes() > 0:
                G_null.remove_edges_from(nx.selfloop_edges(G_null))
                null_reciprocity_list.append(nx.reciprocity(G_null))
        except Exception as e:
            tqdm.write(f"Warning: Could not process {os.path.basename(file_path)}. Error: {e}")
            
    return null_reciprocity_list

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    try:
        G_real = load_real_network(EDGE_LIST_PATH)
        real_reciprocity = nx.reciprocity(G_real)
        
        null_reciprocity_values = analyze_null_models_reciprocity(NULL_MODELS_DIR, N_NULL_MODELS)
        
        print("\n" + "="*70)
        print("--- GLOBAL RECIPROCITY SIGNIFICANCE ANALYSIS ---")
        
        if null_reciprocity_values:
            mean_null_reciprocity = np.mean(null_reciprocity_values)
            std_null_reciprocity = np.std(null_reciprocity_values)
            
            p_value = calculate_two_tailed_p_value(real_reciprocity, null_reciprocity_values)
            
            print(f"Real Network Reciprocity:               {real_reciprocity:.6f}")
            print(f"Null Model Reciprocity (Mean ± SD):     {mean_null_reciprocity:.6f} ± {std_null_reciprocity:.6f}")
            print(f"Two-Tailed Empirical P-value:           {p_value:.6f}")
            
            alpha = 0.05
            print(f"\nSignificance Level (alpha): {alpha}")
            if p_value < alpha:
                print(f"Conclusion: The observed reciprocity ({real_reciprocity:.6f}) is statistically significant (p < {alpha}).")
                print("This indicates the network's tendency for mutual connections is not random.")
            else:
                print(f"Conclusion: The observed reciprocity is NOT statistically significant (p >= {alpha}).")
                print("This suggests the level of reciprocity could be explained by the degree sequence alone.")
        else:
            print("No null model data available to perform significance analysis.")
            
        print("="*70)

    except FileNotFoundError as e:
        print(f"\nFatal Error: {e}")
        print("Please ensure all required data files exist.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")