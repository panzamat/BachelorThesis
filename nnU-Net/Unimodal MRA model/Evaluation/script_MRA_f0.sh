#!/bin/bash
set -e
echo "Finding best configuration for fold 0"
nnUNetv2_find_best_configuration 060 -c 3d_fullres -f 0
echo "Predicting test set fold 0"
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -d Dataset060_IA -i /cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/nnUNet_data/nnUNet_raw/Dataset060_IA/imagesTs -o /cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/nnUNet_data/nnUNet_results/Dataset060_IA/predicted/predicted_f0 -f  0 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans
echo "Predicting completed successfully"
echo "Postprocessing test set fold 0"
nnUNetv2_apply_postprocessing -i /cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/nnUNet_data/nnUNet_results/Dataset060_IA/predicted/predicted_f0 -o /cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/nnUNet_data/nnUNet_results/Dataset060_IA/postprocessed/postprocessed_f0 -pp_pkl_file /cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/nnUNet_data/nnUNet_results/Dataset060_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0/postprocessing.pkl -np 8 -plans_json /cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/nnUNet_data/nnUNet_results/Dataset060_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0/plans.json
echo "Postprocessing completed successfully"
echo "Evaluation test set fold 0"
nnUNetv2_evaluate_folder -djfile /cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/nnUNet_data/nnUNet_raw/Dataset060_IA/dataset.json -pfile /cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/nnUNet_data/nnUNet_results/Dataset060_IA/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json --chill /cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/nnUNet_data/nnUNet_raw/Dataset060_IA/labelsTs /cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/nnUNet_data/nnUNet_results/Dataset060_IA/postprocessed/postprocessed_f0
echo "Evaluation MR data 0 completed successfully"

