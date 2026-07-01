'''
Author: Jaime Martínez Cazón
Description:
Calculates the abundance of 7 directed triangle motifs in a Gene Regulatory
Network and its corresponding null models. Uses multiprocessing to speed up
the triadic census calculation on large graph ensembles.
Outputs a consolidated CSV file for plotting.
'''

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import igraph as ig
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

NULL_MODELS_DIR     = INPUT_DATA_DIR / "null_models"

# Limit the number of null models to process (useful for testing, set to None for all)
MAX_NULL_MODELS = 1000
NUM_CORES = max(1, mp.cpu_count() - 2)  # Leave a couple of cores free

# Map user-defined motif names to igraph / Davis-Leinhardt triad census codes.
# igraph's TriadCensus object supports the same string keys as NetworkX
# (e.g. tc["030C"]), so the mapping table is identical to the original.
#
# The 16 possible triad types in a directed graph (Davis & Leinhardt 1972):
#   003, 012, 102, 021D, 021U, 021C, 111D, 111U,
#   030T, 030C, 201, 120D, 120U, 120C, 210, 300
#
# Only the 7 "closed" triangle types (≥3 directed edges on 3 nodes) are kept.
MOTIF_MAPPING = {
    # name          : igraph/NX code  — description
    "3-cycle"       : "030C",   # A→B→C→A              (directed 3-cycle / feedback loop)
    "3-nocycle"     : "030T",   # A→B, A→C, B→C        (feedforward / transitive triple)
    "4-1biout"      : "120D",   # one bidirectional + 2 out-edges from the mutual pair
    "4-1biin"       : "120U",   # one bidirectional + 2 in-edges  to  the mutual pair
    "4-1biflow"     : "120C",   # one bidirectional + feedforward arrangement
    "5-2bi"         : "210",    # two bidirectional edges + one unidirectional
    "6-3bi"         : "300",    # all three edges bidirectional (fully mutual triangle)
}

# =============================================================================
# CALCULATION FUNCTIONS
# =============================================================================

def count_target_motifs_igraph(G_ig):
    """
    Runs igraph's C-based triad census and returns counts for the 7 target motifs.

    igraph.Graph.triad_census() implements the same Batagelj-Mrvar algorithm as
    NetworkX but in compiled C code.  On large sparse graphs this is typically
    5-50x faster than the pure-Python NetworkX implementation.

    The TriadCensus object returned by igraph supports the same Davis-Leinhardt
    string keys as NetworkX (e.g. tc["030C"]), so MOTIF_MAPPING is unchanged.

    Parameters
    ----------
    G_ig : igraph.Graph
        A directed igraph Graph (loops already removed).

    Returns
    -------
    dict  {motif_name: count}
    """
    tc = G_ig.triad_census()
    return {name: tc[ig_code] for name, ig_code in MOTIF_MAPPING.items()}


def process_single_null_model(filepath):
    """
    Worker function: loads one graphml file with igraph and returns motif counts.

    igraph's GraphML reader is also faster than NetworkX's for large files
    because parsing is done in C.

    Returns None on failure; the main process counts how many None's appear
    so silent data loss is visible.
    """
    try:
        # directed=True is explicit to avoid ambiguity with undirected GraphML files
        G_null = ig.Graph.Read_GraphML(filepath)
        if not G_null.is_directed():
            G_null = G_null.as_directed()
        G_null.simplify(loops=True, multiple=False)   # remove self-loops only
        return count_target_motifs_igraph(G_null)
    except Exception as e:
        return None


def load_real_network():
    """
    Loads the real network from parquet and converts it to an igraph DiGraph.

    Steps:
      1. Read edge list from parquet with pandas (fast columnar read).
      2. Build an igraph.Graph from the edge list using TupleList (avoids
         creating a NetworkX object entirely).
      3. Remove self-loops.
    """
    print("Loading real network...")
    edge_list = pd.read_parquet(EDGE_LIST_PATH)

    # Build igraph graph directly from (source, target) pairs.
    # TupleList is the recommended fast constructor for edge lists.
    edges = list(zip(edge_list["source"], edge_list["target"]))
    G = ig.Graph.TupleList(edges, directed=True)

    # Remove self-loops (multiple=False keeps multi-edges if any; adjust if needed)
    G.simplify(loops=True, multiple=False)

    print(f"  Nodes: {G.vcount():,}   Edges: {G.ecount():,}")
    return G

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # 1. Analyze Real Network
    #    igraph runs on a single core like NetworkX, but the C implementation
    #    makes it 5-50x faster for large graphs.
    # -------------------------------------------------------------------------
    G_real = load_real_network()
    print("Calculating motifs for the real network (this may take a while)...")
    real_counts = count_target_motifs_igraph(G_real)
    print("Real network motifs calculated:", real_counts)

    # -------------------------------------------------------------------------
    # 2. Analyze Null Models
    #    Parallelisation strategy is unchanged: one process per graphml file.
    #    Each worker loads and analyses its file independently, so there is no
    #    shared memory and no GIL contention.
    # -------------------------------------------------------------------------
    null_files = sorted(
        os.path.join(NULL_MODELS_DIR, f)
        for f in os.listdir(NULL_MODELS_DIR)
        if f.endswith(".graphml")
    )

    if MAX_NULL_MODELS:
        null_files = null_files[:MAX_NULL_MODELS]

    print(f"\nCalculating motifs for {len(null_files)} null models using {NUM_CORES} CPU cores...")

    null_results = []
    failed_count = 0

    with mp.Pool(processes=NUM_CORES) as pool:
        for res in tqdm(
            pool.imap_unordered(process_single_null_model, null_files),
            total=len(null_files)
        ):
            if res is not None:
                null_results.append(res)
            else:
                failed_count += 1

    if failed_count:
        print(f"  WARNING: {failed_count} null model file(s) failed to process and were skipped.")

    # -------------------------------------------------------------------------
    # 3. Aggregate Results
    # -------------------------------------------------------------------------
    if not null_results:
        print("Error: No null models were successfully processed.")
        exit()

    null_df = pd.DataFrame(null_results)

    # -------------------------------------------------------------------------
    # 4. Save to CSV
    #    Output format is identical to the original so downstream plotting code
    #    requires no changes.
    # -------------------------------------------------------------------------
    final_data = []
    for motif in MOTIF_MAPPING.keys():
        final_data.append({
            "Motif":      motif,
            "Real_Count": real_counts[motif],
            "Null_Mean":  null_df[motif].mean(),
            "Null_Std":   null_df[motif].std()
        })

    results_df = pd.DataFrame(final_data)
    os.makedirs(os.path.dirname(OUTPUT_DATA_DIR), exist_ok=True)
    results_df.to_csv(OUTPUT_DATA_DIR, index=False)

    print(f"\nMotif analysis complete. Data saved to: {OUTPUT_DATA_DIR}")
    print(results_df.to_string(index=False))
