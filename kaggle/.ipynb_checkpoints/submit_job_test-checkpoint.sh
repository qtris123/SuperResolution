#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=3:00:00
#SBATCH --job-name MYtest
#SBATCH --output MYtest.out
#SBATCH --error MYtest.err

# Run python file.

# Load our conda environment
module load conda/2024.09
source activate CS373

# Run the test code
python3 /scratch/scholar/$USER/SuperResolutionProject/test_.py
