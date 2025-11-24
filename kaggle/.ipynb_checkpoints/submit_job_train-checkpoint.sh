#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=30:00
#SBATCH --job-name MYtrain
#SBATCH --output MYtrain.out
#SBATCH --error MYtrain.err

# Run python file.

# Load our conda environment
module load conda/2024.09
source activate CS373S

# Run your train code
python3 /scratch/scholar/$USER/SuperResolutionProject/train_.py
