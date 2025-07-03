import os
import json
import numpy as np
import nibabel as nib
import pickle
from tqdm import tqdm
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, CropForegroundd
)

def compute_central_mask(img, frac=0.5):
    shape = img.shape
    box_shape = [int(s * frac) for s in shape]
    start = [(s - bs) // 2 for s, bs in zip(shape, box_shape)]
    end = [st + bs for st, bs in zip(start, box_shape)]
    mask = np.zeros(shape, dtype=bool)
    slices = tuple(slice(st, en) for st, en in zip(start, end))
    mask[slices] = True
    return mask, tuple(start), tuple(end)

def normalize_img(img, mask):
    vals = img[mask]
    mean, std = vals.mean(), vals.std()
    if std < 1e-6: std = 1.0
    norm_img = np.zeros_like(img, dtype=np.float32)
    norm_img[mask] = (img[mask] - mean) / std
    return norm_img, mean, std

def save_npy_and_metadata(img, label, meta, out_img_path, out_lbl_path, out_pkl_path):
    np.save(out_img_path, img.astype(np.float32))
    if label is not None:
        np.save(out_lbl_path, label.astype(np.uint8))
    with open(out_pkl_path, "wb") as f:
        pickle.dump(meta, f)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_json", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--mask_frac", type=float, default=0.5)
    parser.add_argument("--crop_loss_thresh", type=float, default=0.25)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Compose pipeline (add CropForegroundd here if you want to crop, like nnUNet)
    trans = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(0.455, 0.455, 0.4),
            mode=("bilinear", "nearest")
        ),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True)
    ])

    with open(args.dataset_json, "r") as f:
        dataset = json.load(f)

    for section in ["training", "validation", "test"]:
        if section not in dataset:
            continue
        for item in tqdm(dataset[section], desc=f"{section}"):
            image_path = os.path.join(args.data_root, item["image"]) if not os.path.isabs(item["image"]) else item["image"]
            label_path = os.path.join(args.data_root, item["label"]) if ("label" in item and not os.path.isabs(item["label"])) else item.get("label", None)
            data_dict = {"image": image_path}
            if label_path:
                data_dict["label"] = label_path

            out = trans(data_dict)
            img = out["image"][0] if out["image"].shape[0] == 1 else out["image"]
            label = out["label"][0] if "label" in out and out["label"].shape[0] == 1 else out.get("label", None)

            # Metadata extraction
            orig_nifti = nib.load(image_path)
            orig_affine = orig_nifti.affine
            orig_shape = orig_nifti.shape
            orig_spacing = orig_nifti.header.get_zooms()[:3]
            crop_bbox = out.get("image_meta_dict", {}).get("crop_bbox", None)  # None if not cropped

            mask_nz = img != 0
            nvox_nz = mask_nz.sum()
            mask_center, start, end = compute_central_mask(img, frac=args.mask_frac)
            mask_center_nz = mask_nz & mask_center
            crop_loss = 1 - (mask_center_nz.sum() / max(1, nvox_nz))
            use_mask = mask_center_nz if crop_loss >= args.crop_loss_thresh else mask_nz

            norm_img, mean, std = normalize_img(img, use_mask)
            meta = {
                "original_affine": orig_affine,
                "original_shape": orig_shape,
                "original_spacing": orig_spacing,
                "crop_bbox": crop_bbox,
                "normalize_mean": float(mean),
                "normalize_std": float(std),
                "used_mask": "central" if crop_loss >= args.crop_loss_thresh else "all_nonzero",
                "mask_frac": args.mask_frac,
                "crop_loss": float(crop_loss),
                "center_mask_start": start,
                "center_mask_end": end,
            }
            if label is not None:
                meta["label_shape"] = label.shape

            # Filenames (nnUNet-like: one per case, e.g., Tr_013.npy, Tr_013_seg.npy, Tr_013.pkl)
            base = os.path.splitext(os.path.basename(image_path))[0]
            out_img_path = os.path.join(args.out_dir, f"{base}.npy")
            out_lbl_path = os.path.join(args.out_dir, f"{base}_seg.npy") if label is not None else None
            out_pkl_path = os.path.join(args.out_dir, f"{base}.pkl")

            save_npy_and_metadata(norm_img, label, meta, out_img_path, out_lbl_path, out_pkl_path)
            print(f"{base:20s} | mean: {mean:.3f} std: {std:.3f} | mask: {meta['used_mask']} | crop_loss: {crop_loss:.2f}")

if __name__ == "__main__":
    main()

