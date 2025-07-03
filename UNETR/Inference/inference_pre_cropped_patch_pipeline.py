import os
import glob
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, SpatialPadd, NormalizeIntensityd
)
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNETR

# --- Settings ---
MODEL_CKPTS = [
    "/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/UNETR/Dataset_UNETR/model_output/fold0_96/best_model.pth",
    "/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/UNETR/Dataset_UNETR/model_output/fold1_96/best_model.pth",
    "/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/UNETR/Dataset_UNETR/model_output/fold2_96/best_model.pth",
    "/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/UNETR/Dataset_UNETR/model_output/fold3_96/best_model.pth",
    "/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/UNETR/Dataset_UNETR/model_output/fold4_96/best_model.pth",
]

SPACING = (0.455, 0.455, 0.4)
IMG_SIZE = (96, 96, 96)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_FOLDER = "/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/UNETR/Dataset_UNETR/imagesTs"
OUTPUT_FOLDER = "/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/UNETR/Dataset_UNETR/model_output/unetr_inference_output3"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Preprocessing (must exactly match training!) ---
pre_trans = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=SPACING, mode="bilinear"),
    SpatialPadd(keys=["image"], spatial_size=IMG_SIZE),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=False),
])

def match_shape(arr, target_shape):
    """Pad or crop arr to match target_shape."""
    out = np.zeros(target_shape, dtype=arr.dtype)
    min_shape = tuple(min(s1, s2) for s1, s2 in zip(arr.shape, target_shape))
    slices_in = tuple(slice(0, s) for s in min_shape)
    slices_out = tuple(slice(0, s) for s in min_shape)
    out[slices_out] = arr[slices_in]
    return out

# --- Inference loop ---
nii_files = sorted(glob.glob(os.path.join(TEST_FOLDER, "*.nii.gz")))
print(f"Found {len(nii_files)} images for inference.")

for img_path in tqdm(nii_files):
    # --- Preprocess input image ---
    data = pre_trans({"image": img_path})
    img = data["image"]  # shape: (1, D, H, W) (MetaTensor or Tensor)

    # Convert MetaTensor to Tensor if needed (it's already Tensor-like)
    if not isinstance(img, torch.Tensor):
        img = torch.as_tensor(np.asarray(img), dtype=torch.float32)

    # Add batch dim: (1, 1, D, H, W)
    if img.ndim == 4:
        img_tensor = img.unsqueeze(0).to(DEVICE)
    elif img.ndim == 5:
        img_tensor = img.to(DEVICE)
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")

    # --- Save original image meta
    orig_nii = nib.load(img_path)
    orig_affine = orig_nii.affine
    orig_shape = orig_nii.shape

    # --- Ensemble prediction (class 1 probability) ---
    pred_probs = []
    with torch.no_grad():
        for ckpt in MODEL_CKPTS:
            model = UNETR(
                in_channels=1, out_channels=2, img_size=IMG_SIZE, feature_size=16,
                hidden_size=768, mlp_dim=3072, num_heads=12, proj_type="perceptron",
                norm_name="instance", res_block=True
            ).to(DEVICE)
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            model.eval()
            logits = sliding_window_inference(img_tensor, IMG_SIZE, 1, model, overlap=0.5)
            softmax = torch.softmax(logits, dim=1)
            pred_probs.append(softmax[0, 1].cpu().numpy())
            del model
            torch.cuda.empty_cache()
    mean_prob = np.mean(np.stack(pred_probs, axis=0), axis=0)  # (D, H, W)
    mask_pred = (mean_prob > 0.5).astype(np.uint8)

    # --- Match shape to original image
    out_mask = match_shape(mask_pred, orig_shape)
    out_prob = match_shape(mean_prob, orig_shape)

    # --- Save outputs with original affine ---
    base_name = os.path.basename(img_path).replace('.nii.gz', '')
    nib.save(
        nib.Nifti1Image(out_mask.astype(np.uint8), orig_affine, orig_nii.header),
        os.path.join(OUTPUT_FOLDER, f"{base_name}_seg.nii.gz")
    )
    nib.save(
        nib.Nifti1Image(out_prob.astype(np.float32), orig_affine, orig_nii.header),
        os.path.join(OUTPUT_FOLDER, f"{base_name}_probmap.nii.gz")
    )

print("Inference complete! Masks and probability maps saved, aligned to original images.")

