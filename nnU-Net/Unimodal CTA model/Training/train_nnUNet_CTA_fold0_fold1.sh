#!/bin/bash
#SBATCH --job-name=nnUNet_train_059
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=44:00:00
#SBATCH --mem=128GB
#SBATCH --partition=earth-4
#SBATCH --constraint=rhel8

export CONDA_PREFIX="/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/miniconda3"
export PATH="$CONDA_PREFIX/bin:$PATH"
source /cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/miniconda3/etc/profile.d/conda.sh
conda activate /cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/miniconda3/envs/nnUNet


# Confirm Environment Setup
echo "Using Conda from: $(which conda)"
echo "Using Python from: $(which python)"
echo "Using nnUNet from: $(which nnUNetv2_train)"

# Set nnUNet Data Paths
export nnUNet_raw="/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/nnUNet_data/nnUNet_results"

echo "nnUNet_raw is set to: $nnUNet_raw"
echo "nnUNet_preprocessed is set to: $nnUNet_preprocessed"
echo "nnUNet_results is set to: $nnUNet_results"

# Run Fold 0 on GPU 0
echo "Starting training for fold 0 on GPU 0..."
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 059 3d_fullres 0 --npz
echo "Training for fold 0 done."

# Run Fold 1 on GPU 1 AFTER Fold 0 is done
echo "Starting training for fold 1 on GPU 1..."
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 059 3d_fullres 1 --npz
echo "Training for fold 1 done."

echo "All nnUNet trainings completed!"
