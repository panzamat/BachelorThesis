import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from monai.data import load_decathlon_datalist
from monai.transforms import (
    Orientationd, Spacingd, SpatialPadd, EnsureChannelFirstd, LoadImaged,
    Compose, RandCropByPosNegLabeld, NormalizeIntensityd
)

# --- Parameters
data_dir = "/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/UNETR/Dataset_UNETR"
output_dir = os.path.join(data_dir, "preprocessed_patches_zscore_monaistd_finalfinal2")
os.makedirs(output_dir, exist_ok=True)
img_size = (96, 96, 96)
debug_visualize = False  # Optionally visualize patches

# --- MONAI transforms (spatial + normalization)
pre_transforms = [
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(0.455, 0.455, 0.4), mode=("bilinear", "nearest")),
    SpatialPadd(keys=["image", "label"], spatial_size=img_size),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=False),  # Standard z-score normalization!
]

def print_case_stats(img, label, when, caseid):
    fg_frac = float((label > 0).sum()) / label.size if label.size > 0 else 0.0
    print(
        f"[{when}] {caseid} | shape: {img.shape}, "
        f"min/max/mean/std: {img.min():.3f}/{img.max():.3f}/{img.mean():.3f}/{img.std():.3f} | "
        f"Label unique: {np.unique(label)} sum: {label.sum()} | fg%: {fg_frac:.5f}"
    )

def save_patch_debug(img, lbl, caseid, crop_idx, outdir):
    import matplotlib.pyplot as plt
    midz = img.shape[-1] // 2
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img[0, :, :, midz], cmap="gray")
    plt.title("Image (norm)")
    plt.subplot(1, 2, 2)
    plt.imshow(lbl[0, :, :, midz], cmap="gray")
    plt.title("Label")
    plt.suptitle(f"{caseid} patch {crop_idx}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{caseid}_patch{crop_idx}_debug.png"))
    plt.close()

def process_case(img_path, lbl_path, caseid, outdir, n_patches=20, pos=10, neg=10):
    # --- 1. Load and preprocess full image + label (orientation, spacing, pad, normalization)
    loader = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
    ] + pre_transforms)
    data = loader({"image": img_path, "label": lbl_path})
    img = data["image"].astype(np.float32)
    lbl = data["label"].astype(np.uint8)

    print_case_stats(img, lbl, "AFTER_NORM", caseid)

    # --- 2. Patch sampling (with anatomical constraint for background)
    sampler = RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=img_size,
        pos=pos, neg=neg,
        num_samples=n_patches,
        allow_smaller=True,
        image_key="image",           # <--- This makes it use image content for background
        image_threshold=0.01         # <--- Set this to a suitable value for your data
    )
    patches = sampler({"image": img, "label": lbl})

    for crop_idx, crop in enumerate(patches):
        patch_img = crop["image"].astype(np.float32)
        patch_lbl = crop["label"].astype(np.uint8)
        print_case_stats(patch_img, patch_lbl, f"PATCH_{crop_idx:02d}", caseid)
        patch_path = os.path.join(outdir, f"{caseid}_patch{crop_idx:02d}.npz")
        np.savez_compressed(patch_path, image=patch_img, label=patch_lbl)
        if debug_visualize and crop_idx < 1:
            save_patch_debug(patch_img, patch_lbl, caseid, crop_idx, outdir)

if __name__ == "__main__":
    dataset_json = os.path.join(data_dir, "dataset.json")
    train_files = load_decathlon_datalist(dataset_json, True, "training")

    print(f"Preprocessing and saving patches to {output_dir} ...")
    patch_counter = 0

    for idx, f in enumerate(tqdm(train_files)):
        img_path = f['image'] if os.path.isabs(f['image']) else os.path.join(data_dir, f['image'])
        lbl_path = f['label'] if os.path.isabs(f['label']) else os.path.join(data_dir, f['label'])
        caseid = os.path.splitext(os.path.basename(img_path))[0].replace(".nii", "")
        process_case(img_path, lbl_path, f"{idx:05d}_{caseid}", output_dir, n_patches=20, pos=10, neg=10)
        patch_counter += 1

    print(f"Done. Saved patches for {patch_counter} cases to {output_dir}")
