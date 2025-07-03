#!/bin/bash
#SBATCH --job-name=predict_evaluate_062
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=64:00:00
#SBATCH --mem=64GB
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
export nnUNet_raw="/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/multimodal_data/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/multimodal_data/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/multimodal_data/nnUNet_data/nnUNet_results"

echo "nnUNet_raw is set to: $nnUNet_raw"
echo "nnUNet_preprocessed is set to: $nnUNet_preprocessed"
echo "nnUNet_results is set to: $nnUNet_results"

bash script_062_0.sh
bash script_062_1.sh
bash script_062_2.sh
bash script_062_3.sh
bash script_062_4.sh
