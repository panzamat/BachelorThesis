import os
import numpy as np
import torch
import nibabel as nib
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, CropForegroundd
)
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNETR
from tqdm import tqdm
import SimpleITK as sitk

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (96, 96, 96)
SPACING = (0.455, 0.455, 0.4)
N_CLASSES = 2

MODEL_CKPTS = [
    "/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/slurm_jobs/UNETR/UNETR/model_output/final_outputs/fold_0/best_model_fold0.pth",
    "/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/slurm_jobs/UNETR/UNETR/model_output/final_outputs/fold_1/best_model_fold1.pth",
    "/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/slurm_jobs/UNETR/UNETR/model_output/final_outputs/fold_2/best_model_fold2.pth",
    "/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/slurm_jobs/UNETR/UNETR/model_output/final_outputs/fold_3/best_model_fold3.pth",
    "/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/slurm_jobs/UNETR/UNETR/model_output/final_outputs/fold_4/best_model_fold4.pth",
]


TEST_FOLDER = "/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/UNETR/Dataset_UNETR/imagesTs"
OUTPUT_FOLDER = "/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/slurm_jobs/UNETR/UNETR/model_output/final_outputs/unetr_test_outputs_correct2"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
MASK_FRAC = 0.5
CROP_LOSS_THRESH = 0.25

pre_trans = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=SPACING, mode="bilinear"),
    CropForegroundd(keys=["image"], source_key="image", allow_smaller=True),
])

def compute_central_mask(img, frac=MASK_FRAC):
    shape = img.shape
    box_shape = [int(s * frac) for s in shape]
    start = [(s - bs) // 2 for s, bs in zip(shape, box_shape)]
    end = [st + bs for st, bs in zip(start, box_shape)]
    mask = np.zeros(shape, dtype=bool)
    slices = tuple(slice(st, en) for st, en in zip(start, end))
    mask[slices] = True
    return mask

def normalize_img(img, mask):
    vals = img[mask]
    mean, std = vals.mean(), vals.std()
    if std < 1e-6: std = 1.0
    norm_img = np.zeros_like(img, dtype=np.float32)
    norm_img[mask] = (img[mask] - mean) / std
    return norm_img

def uncrop_to_original(cropped, orig_shape, bbox):
    out = np.zeros(orig_shape, dtype=cropped.dtype)
    slices = tuple(slice(start, end) for start, end in bbox)
    out[slices] = cropped
    return out

def resample_to_reference(arr, ref_nifti_path, is_probmap=False):
    itk_img = sitk.GetImageFromArray(arr.astype(np.float32 if is_probmap else np.uint8))
    ref_img = sitk.ReadImage(ref_nifti_path)
    res = sitk.Resample(
        itk_img,
        ref_img,
        sitk.Transform(),
        sitk.sitkLinear if is_probmap else sitk.sitkNearestNeighbor,
        0,
        itk_img.GetPixelID()
    )
    return sitk.GetArrayFromImage(res)

def get_crop_bbox(cropped_img):
    fg = cropped_img != 0
    coords = np.array(np.where(fg))
    minc = coords.min(axis=1)
    maxc = coords.max(axis=1) + 1
    return [(minc[i], maxc[i]) for i in range(3)]

nii_files = sorted([f for f in os.listdir(TEST_FOLDER) if f.endswith('.nii.gz')])
print(f"Found {len(nii_files)} images for inference.")

for img_name in tqdm(nii_files):
    img_path = os.path.join(TEST_FOLDER, img_name)
    orig_nii = nib.load(img_path)
    orig_affine = orig_nii.affine
    orig_shape = orig_nii.shape

    # -- Preproc --
    data = pre_trans({"image": img_path})
    img = data["image"]
    cropped_img = img[0]
    crop_bbox = get_crop_bbox(cropped_img)

    # -- Intensity norm --
    mask_nz = cropped_img != 0
    nvox_nz = mask_nz.sum()
    mask_center = compute_central_mask(cropped_img, frac=MASK_FRAC)
    mask_center_nz = mask_nz & mask_center
    crop_loss = 1 - (mask_center_nz.sum() / max(1, nvox_nz))
    use_mask = mask_center_nz if crop_loss >= CROP_LOSS_THRESH else mask_nz
    norm_img = normalize_img(cropped_img, use_mask)
    norm_img = norm_img[None]

    # -- Predict --
    img_tensor = torch.from_numpy(norm_img).unsqueeze(0).to(DEVICE)
    pred_probs = []
    with torch.no_grad():
        for ckpt in MODEL_CKPTS:
            model = UNETR(
                in_channels=1, out_channels=N_CLASSES, img_size=IMG_SIZE,
                feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12,
                proj_type="perceptron", norm_name="instance", res_block=True
            ).to(DEVICE)
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            model.eval()
            logits = sliding_window_inference(img_tensor, IMG_SIZE, 1, model, overlap=0.5)
            softmax = torch.softmax(logits, dim=1)
            pred_probs.append(softmax[0, 1].cpu().numpy())
            del model
            torch.cuda.empty_cache()
    mean_prob = np.mean(np.stack(pred_probs, axis=0), axis=0)
    mask_pred = (mean_prob > 0.5).astype(np.uint8)

    # -- Uncrop --
    uncropped_pred = uncrop_to_original(mask_pred, cropped_img.shape, crop_bbox)
    uncropped_prob = uncrop_to_original(mean_prob, cropped_img.shape, crop_bbox)

    # -- Resample to raw image space --
    mask_resampled = resample_to_reference(uncropped_pred, img_path, is_probmap=False)
    prob_resampled = resample_to_reference(uncropped_prob, img_path, is_probmap=True)

    # -- Save --
    base = img_name.replace('.nii.gz', '')
    nib.save(
        nib.Nifti1Image(mask_resampled.astype(np.uint8), orig_affine, orig_nii.header),
        os.path.join(OUTPUT_FOLDER, f"{base}_seg.nii.gz")
    )
    nib.save(
        nib.Nifti1Image(prob_resampled.astype(np.float32), orig_affine, orig_nii.header),
        os.path.join(OUTPUT_FOLDER, f"{base}_probmap.nii.gz")
    )

print("DONE! Masks and probability maps aligned with the raw image.")


