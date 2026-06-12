'''
Author: Jaime Martínez Cazón
Description:
Visualizes the abundance of different directed triangle motifs in a Gene 
Regulatory Network. Reads a consolidated CSV containing counts for the real 
network and mean/std stats for the surrogate model ensemble. 
Outputs a log-scale scatter plot.
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# SETUP: FILE PATHS
# =============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
MOTIF_COUNTS_FILE = os.path.join(script_dir, "../data/GRN_data/motif_counts.csv")

# =============================================================================
# PLOTTING FUNCTION
# =============================================================================

def create_motif_plot(df):
    """
    Generates and displays the scatter plot for triangle motif comparison.
    """
    plt.figure(figsize=(12, 8))

    motifs = df['Motif'].values
    real_counts = df['Real_Count'].values
    null_means = df['Null_Mean'].values
    null_stds = df['Null_Std'].values

    x_positions = np.arange(len(motifs))

    # --- Plot Real Network Data ---
    real_indices = np.where(real_counts > 0)[0]
    if len(real_indices) > 0:
        plt.scatter(x_positions[real_indices], real_counts[real_indices],
                    marker='s', color='#bcbd22', s=150, zorder=10,
                    label='Real Network')

    # --- Plot Surrogate Model Data with Error Bars ---
    null_indices = np.where(null_means > 0)[0]
    if len(null_indices) > 0:
        plt.errorbar(x_positions[null_indices], null_means[null_indices],
                     yerr=null_stds[null_indices],
                     fmt='o', color='black', markersize=10, capsize=5,
                     alpha=0.8, linestyle='None', label='Configuration Model')

    # --- Plot Formatting ---
    plt.yscale('log')
    plt.title("Triangle Motif Abundance", fontsize=28, fontweight='bold')
    plt.xlabel("Triangle Motif Category", fontsize=22)
    plt.ylabel("Number of Triangles (#)", fontsize=22)

    # Use motif names as x-axis labels instead of numbers for better readability
    plt.xticks(x_positions, motifs, fontsize=16, rotation=45, ha="right")
    plt.yticks(fontsize=18)

    plt.legend(fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    if not os.path.exists(MOTIF_COUNTS_FILE):
        print(f"Error: Data file not found at {MOTIF_COUNTS_FILE}")
        print("Please run the motif data generation script first.")
    else:
        print("Loading data...")
        results_df = pd.read_csv(MOTIF_COUNTS_FILE)
        
        print("Generating plot...")
        create_motif_plot(results_df)