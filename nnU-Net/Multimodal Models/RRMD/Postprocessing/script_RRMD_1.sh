#!/bin/bash
set -e
echo "Finding best configuration for fold 1"
nnUNetv2_find_best_configuration 064 -c 3d_fullres -f 1 -tr nnUNetTrainerTverskyFocal -p nnUNetPlans
echo "Predicting test set fold 1"
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -d Dataset064_IA -i /cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/multimodal_data/nnUNet_data/nnUNet_raw/Dataset064_IA/imagesTs -o /cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/multimodal_data/nnUNet_data/nnUNet_results/Dataset064_IA/predicted/predicted_f1 -f  1 -tr nnUNetTrainerTverskyFocal -c 3d_fullres -p nnUNetPlans
echo "Predicting completed successfully"
echo "Postprocessing test set fold 1"
nnUNetv2_apply_postprocessing -i /cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/multimodal_data/nnUNet_data/nnUNet_results/Dataset064_IA/predicted/predicted_f1 -o /cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/multimodal_data/nnUNet_data/nnUNet_results/Dataset064_IA/postprocessed/postprocessed_f1 -pp_pkl_file /cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/multimodal_data/nnUNet_data/nnUNet_results/Dataset064_IA/nnUNetTrainerTverskyFocal__nnUNetPlans__3d_fullres/crossval_results_folds_1/postprocessing.pkl -np 8 -plans_json /cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/multimodal_data/nnUNet_data/nnUNet_results/Dataset064_IA/nnUNetTrainerTverskyFocal__nnUNetPlans__3d_fullres/crossval_results_folds_1/plans.json
echo "Postprocessing completed successfully"
echo "Evaluation test set fold 1"
nnUNetv2_evaluate_folder -djfile /cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/multimodal_data/nnUNet_data/nnUNet_raw/Dataset064_IA/dataset.json -pfile /cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/multimodal_data/nnUNet_data/nnUNet_results/Dataset064_IA/nnUNetTrainerTverskyFocal__nnUNetPlans__3d_fullres/plans.json --chill /cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/multimodal_data/nnUNet_data/nnUNet_raw/Dataset064_IA/labelsTs /cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/multimodal_data/nnUNet_data/nnUNet_results/Dataset064_IA/postprocessed/postprocessed_f1
echo "Evaluation CT data 1 completed successfully"
