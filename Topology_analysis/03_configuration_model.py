'''
Author: Jaime Martínez Cazón
Adapted for direct gene names and unparquet edge lists (unweighted).

Description:
Generates an ensemble of surrogate networks (null models) based on a Gene 
Regulatory Network using the directed configuration model. This preserves the 
in-degree and out-degree sequence of every node while randomizing connections. 
The generated networks maintain original gene names (strings) as nodes and are 
saved in GraphML format.
'''

import os
from pathlib import Path
import pandas as pd
import networkx as nx
from tqdm import tqdm

# =============================================================================
# SETUP: CONFIGURATION PARAMETERS
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

NUM_NULL_MODELS = 1000  

# =============================================================================
# DATA LOADING FUNCTION
# =============================================================================

def load_real_network(filepath):
    """Loads the real network from an edge list file."""
    
    print(f"Loading real network from: {filepath}")
    edge_list = pd.read_parquet(filepath)
    
    G_real = nx.from_pandas_edgelist(
        edge_list,
        source='source',
        target='target',
        create_using=nx.DiGraph()
    )
    print("Real network loaded successfully.")
    return G_real

# =============================================================================
# NULL MODEL GENERATION FUNCTION
# =============================================================================

def generate_null_models(G_real, num_models, output_dir):
    """
    Generates and saves null models using the directed configuration model,
    preserving the original string node names.
    """
    print(f"Generating {num_models} null models in directory: '{output_dir}'")
    
    # Fix the order of nodes to properly map degrees and relabel later
    nodes_list = list(G_real.nodes())
    
    # Extract degree sequences matching the exact order of nodes_list
    in_degree_sequence = [G_real.in_degree(n) for n in nodes_list]
    out_degree_sequence = [G_real.out_degree(n) for n in nodes_list]
    
    # Mapping dictionary to restore string gene names after generation
    node_mapping = {i: nodes_list[i] for i in range(len(nodes_list))}

    for i in tqdm(range(num_models), desc="Generating Null Models"):
        # Generate null model (creates integer nodes 0 to N-1)
        G_null_multi = nx.directed_configuration_model(
            in_degree_sequence,
            out_degree_sequence,
            create_using=nx.MultiDiGraph
        )

        # Convert to simple DiGraph to remove parallel edges
        G_null = nx.DiGraph(G_null_multi)
        
        # Remove self-loops
        G_null.remove_edges_from(nx.selfloop_edges(G_null))

        # Relabel integer nodes back to original gene name strings
        nx.relabel_nodes(G_null, node_mapping, copy=False)

        # Save as GraphML
        file_path = os.path.join(output_dir, f"null_model_{str(i).zfill(4)}.graphml")
        nx.write_graphml(G_null, file_path)

    print(f"\nSuccessfully generated and saved {num_models} null models.")

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    try:
        real_network = load_real_network(EDGE_LIST_PATH)
        generate_null_models(real_network, NUM_NULL_MODELS, OUTPUT_DATA_DIR)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the data file is in the correct location and try again.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")