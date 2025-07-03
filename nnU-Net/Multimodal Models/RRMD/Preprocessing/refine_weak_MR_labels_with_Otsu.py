from pathlib import Path
import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import label as connected_components
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes, remove_small_objects

# Paths
MR_IMAGES_DIR = Path("/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/multimodal_data/nnUNet_data/nnUNet_raw/imagesTs_correct")
LABELS_DIRS = [
    Path("/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/nnUNet_data/nnUNet_raw/Dataset060_IA/labelsTs")
]
OUTPUT_LABELS_DIR = Path("/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/multimodal_data/nnUNet_data/nnUNet_raw/refined_MR_labels_Ts")
OUTPUT_LABELS_DIR.mkdir(parents=True, exist_ok=True)

def load_nifti(path):
    img = nib.load(str(path))
    return img.get_fdata(), img.affine, img.header

def save_nifti(data, affine, header, path):
    nib.save(nib.Nifti1Image(data.astype(np.uint8), affine, header), str(path))

def find_bbox(mask):
    coords = np.argwhere(mask)
    minc, maxc = coords.min(0), coords.max(0) + 1
    return tuple(slice(i, j) for i, j in zip(minc, maxc))

def refine_label_with_otsu(mr_image, weak_label):
    refined = np.zeros_like(weak_label)
    if np.any(weak_label):
        labeled_mask, num = connected_components(weak_label)
        for i in range(1, num + 1):
            comp = labeled_mask == i
            bbox = find_bbox(comp)
            mr_crop = mr_image[bbox]
            mask_crop = comp[bbox]
            values = mr_crop[mask_crop]
            if values.size == 0:
                continue
            try:
                threshold = threshold_otsu(values)
            except ValueError:
                continue
            refined_crop = (mr_crop >= threshold) & mask_crop
            refined_crop = remove_small_objects(refined_crop, min_size=10)
            refined_crop = remove_small_holes(refined_crop, area_threshold=10)
            refined[bbox][refined_crop] = 1
    return refined

if __name__ == "__main__":
    for mr_file in tqdm(sorted(MR_IMAGES_DIR.glob("Tr_*_0000.nii.gz")), desc="Refining labels"):
        # Extract XXX from Tr_XXX_0000.nii.gz
        subject_num = mr_file.name.split("_")[1]
        label_file = None
        for label_dir in LABELS_DIRS:
            candidate = label_dir / f"Tr_{subject_num}.nii.gz"
            if candidate.exists():
                label_file = candidate
                break
        if label_file is None:
            print(f"Label not found for Tr_{subject_num}")
            continue

        mr_data, affine, header = load_nifti(mr_file)
        label_data, _, _ = load_nifti(label_file)
        refined = refine_label_with_otsu(mr_data, label_data)
        save_nifti(refined, affine, header, OUTPUT_LABELS_DIR / f"Tr_{subject_num}.nii.gz")

    print("Refinement done.")

