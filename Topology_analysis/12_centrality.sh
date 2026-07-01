#!/bin/bash
#SBATCH --job-name=GRN_Centrality           # Sets the job's name shown in the queue
#SBATCH --output=logs/%x_%j.out	            # Writes the output to a given file
#SBATCH --error=logs/%x_%j.err  	        # Writes the errors to a given file
#SBATCH --time=24:00:00         			# If the job takes more time than specified, is remode from the queue
#SBATCH --ntasks=1              			# Number of tasks to run (= environment creation + [python scripts to run] + environment exportation )
#SBATCH --mem=48G                			# Real Memory per node. In MB by default. Other optional unita are [K|M|G|T]
#SBATCH --nodes=1               			# Nodes to be allocated to this job
#SBATCH -c 16       					    # Number of CPUs that will be used for each allocated node
#SBATCH --partition="all"   			    # Partition from wich the nodes will be selected # #SBATCH --nodelist=cpu13

## Create output directories (logs,figures,output_data) for the job inside the current working directory,
## using the job name and job ID in each output file to avoid overwriting files from different jobs. 
cd "$SLURM_SUBMIT_DIR"
mkdir -p figures data_output logs

# Load miniconda to run the python script with the celloracle environment.
module purge
module load tools/miniconda/python3.10/23.3.1

# Execute
srun -n1 conda run --prefix=/exports/ana-scarlab/jmartinezcazon/envs/celloracle_env python 12_centrality.py