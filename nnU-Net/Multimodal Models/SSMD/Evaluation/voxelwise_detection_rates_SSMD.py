import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import pandas as pd

def load_nifti(file_path):
    img = nib.load(file_path)
    return img.get_fdata()

def compute_voxelwise_rates(pred_files, gt_files):
    total_tp, total_fp, total_fn = 0, 0, 0

    for pred_path, gt_path in tqdm(zip(pred_files, gt_files), total=len(pred_files), desc="Computing Global Rates"):
        pred = (load_nifti(pred_path) > 0.5).astype(int)
        gt = (load_nifti(gt_path) > 0.5).astype(int)

        total_tp += np.sum((pred == 1) & (gt == 1))
        total_fp += np.sum((pred == 1) & (gt == 0))
        total_fn += np.sum((pred == 0) & (gt == 1))

    total = total_tp + total_fp + total_fn
    if total == 0:
        return {"TP Rate": 0, "FP Rate": 0, "FN Rate": 0}

    return {
        "TP Rate": total_tp / total,
        "FP Rate": total_fp / total,
        "FN Rate": total_fn / total
    }

def evaluate_global_voxel_rates():
    dataset = "CTMR"
    folds = range(5)
    all_tp, all_fp, all_fn = 0, 0, 0

    for i in folds:
        pred_dir = f"/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/multimodal_data/nnUNet_data/nnUNet_results/Dataset062_IA/postprocessed/postprocessed_f{i}"
        gt_dir = f"/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/multimodal_data/nnUNet_data/nnUNet_raw/Dataset062_IA/labelsTs"

        gt_files_dict = {
            f: os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith(".nii.gz")
        }
        pred_filenames = [f for f in os.listdir(pred_dir) if f.endswith(".nii.gz") and f in gt_files_dict]

        pred_files = [os.path.join(pred_dir, f) for f in pred_filenames]
        gt_files = [gt_files_dict[f] for f in pred_filenames]

        rates = compute_voxelwise_rates(pred_files, gt_files)
        all_tp += rates["TP Rate"] * len(pred_files)
        all_fp += rates["FP Rate"] * len(pred_files)
        all_fn += rates["FN Rate"] * len(pred_files)

    total_cases = len(folds) * len(pred_files)
    final_rates = {
        "TP Rate": all_tp / total_cases,
        "FP Rate": all_fp / total_cases,
        "FN Rate": all_fn / total_cases
    }

    df = pd.DataFrame([final_rates])
    df.to_csv(f"voxelwise_rates_model_level_{dataset}_062.csv", index=False)
    print("\nModel-Level Voxel-wise Rates:")
    print(df)

if __name__ == "__main__":
    evaluate_global_voxel_rates()

