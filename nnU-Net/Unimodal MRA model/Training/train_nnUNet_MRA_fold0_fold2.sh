#!/bin/bash
#SBATCH --job-name=nnUNet_train_fold0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --time=44:00:00
#SBATCH --mem=128GB
#SBATCH --partition=earth-5
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

for i in {0..2}
do
    echo "Starting training for fold $i..."
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 060 3d_fullres $i --npz
    echo "Training for fold $i done."
done

