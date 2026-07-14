'''
Author: Jaime Martínez Cazón

Generates an ensemble of surrogate (null model) networks from a GRN edge list
using the directed configuration model. Preserves in/out-degree sequence of
every node while randomizing connections. Saves each null model as a parquet
edge list (source, target) instead of GraphML for faster I/O downstream.
'''

import os
from pathlib import Path
import pandas as pd
import networkx as nx
from tqdm import tqdm

# =============================================================================
# PATHS
# =============================================================================

script_dir     = Path(__file__).parent
INPUT_DATA_DIR = Path(script_dir / "../../data")
FIG_DIR        = Path(script_dir / "figures")
OUTPUT_DATA_DIR= Path(script_dir / "data_output")

FIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

EDGE_LIST_PATH  = INPUT_DATA_DIR / "edge_list_to_analyze.parquet"
NUM_NULL_MODELS = 1000

# =============================================================================
# FUNCTIONS
# =============================================================================

def load_real_network(filepath):
    """Loads the real network from a parquet edge list."""
    print(f"Loading real network from: {filepath}")
    edge_list = pd.read_parquet(filepath)
    G_full = nx.from_pandas_edgelist(
        edge_list, source='source', target='target',
        create_using=nx.DiGraph()
    )

    ## Only work with main WCC
    giant_component_nodes = max(nx.weakly_connected_components(G_full), key=len)
    G = G_full.subgraph(giant_component_nodes).copy()

    ## Eliminate self-loops
    G.remove_edges_from(nx.selfloop_edges(G))
    print(f"Loaded: {G.number_of_nodes():,} nodes | {G.number_of_edges():,} edges")
    return G


def generate_null_models(G_real, num_models, output_dir):
    """
    Generates null models via the directed configuration model and saves
    each as a parquet edge list. Node integer indices are relabeled back
    to original gene name strings before saving.

    Note: converting MultiDiGraph -> DiGraph removes parallel edges, which
    slightly reduces edge count relative to the original. Self-loops are
    also removed. Both are standard post-processing steps for this null model.
    """
    nodes_list         = list(G_real.nodes())
    in_degree_sequence = [G_real.in_degree(n)  for n in nodes_list]
    out_degree_sequence= [G_real.out_degree(n) for n in nodes_list]
    node_mapping       = {i: nodes_list[i] for i in range(len(nodes_list))}

    print(f"Generating {num_models} null models → {output_dir}")

    for i in tqdm(range(num_models), desc="Null models"):
        # Configuration model returns a MultiDiGraph with integer nodes
        G_multi = nx.directed_configuration_model(
            in_degree_sequence,
            out_degree_sequence,
            create_using=nx.MultiDiGraph
        )
        # Remove parallel edges and self-loops
        G_null = nx.DiGraph(G_multi)
        G_null.remove_edges_from(nx.selfloop_edges(G_null))

        # Restore original gene name strings
        nx.relabel_nodes(G_null, node_mapping, copy=False)

        # Save as parquet edge list — much faster to load than GraphML
        edges_df = nx.to_pandas_edgelist(G_null)[['source', 'target']]
        out_path = OUTPUT_DATA_DIR / f"null_model_{str(i).zfill(3)}.parquet"
        edges_df.to_parquet(out_path, index=False)

    print(f"Done. {num_models} null models saved to {OUTPUT_DATA_DIR}")

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        G_real = load_real_network(EDGE_LIST_PATH)
        generate_null_models(G_real, NUM_NULL_MODELS, OUTPUT_DATA_DIR)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise