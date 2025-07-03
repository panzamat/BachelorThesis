#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UNETR training with Dice+Focal loss, temperature-scaled softmax, robust MONAI metrics.
- 5-fold cross-validation: each run uses one fold as val, the rest as train (specified by --fold).
- Each epoch: train on random sample of train_patches_per_epoch from pool (75% fg, 25% bg).
- Validation each epoch: always the same fixed random 1/4 subset of val patches from the current fold (75% fg, 25% bg).
- Every 5 epochs: visualize prediction for a random val patch (from the subset).
- Sliding window validation: all val cases, every VAL_EVERY_N_EPOCHS.
- Performance metrics (incl. avg FG pred/label and ROC-AUC) printed each epoch, also saved to CSV at end.
- Best val Dice: save ROC-AUC curve as .png and store model.
- At end: compute best threshold for inference.
"""

import argparse
import os
import random
import re
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import load_decathlon_datalist
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from monai.networks.nets import UNETR
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandGaussianNoised,
    Spacingd,
    RandScaleIntensityd,
    RandShiftIntensityd,
)

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import csv
import nibabel as nib

# ------------------- Temperature-Scaled Softmax -------------------
def softmax_with_temp(logits, T=0.7):
    """Return softmax with temperature scaling."""
    return F.softmax(logits / T, dim=1)

# ------------------- Focal Loss -------------------
def focal_loss_custom(logits, target, gamma=2.0, reduction="mean"):
    if target.dtype != torch.long:
        target = target.long()
    ce_loss = F.cross_entropy(logits, target, reduction="none")
    pt = torch.exp(-ce_loss)
    focal_loss = (1 - pt) ** gamma * ce_loss
    if reduction == "mean":
        return focal_loss.mean()
    elif reduction == "sum":
        return focal_loss.sum()
    return focal_loss

# ------------------- Combo Loss (Dice + Focal, temp-scaled softmax) -------------------
def combo_loss(logits, labels, dice_fn):
    if logits.device != labels.device:
        labels = labels.to(logits.device)
    if labels.dtype != torch.long:
        labels = labels.long()
    dice = dice_fn(logits, labels)
    focal = focal_loss_custom(logits, labels.squeeze(1), gamma=2.0)
    return 0.7 * dice + 0.3 * focal

# ------------------- Save Predicted Patch and PNG -------------------
def save_patch_prediction(img, lbl, pred, outdir, cid):
    """
    Save image, label, and predicted probability as .nii.gz and optionally a middle-slice .png.
    - img, lbl: shape [1, 1, D, H, W]
    - pred: shape [1, D, H, W] or [1, 1, D, H, W]
    """
    img_np = img[0, 0].detach().cpu().numpy()
    lbl_np = lbl[0, 0].detach().cpu().numpy()
    pred_np = pred.squeeze().detach().cpu().numpy()

    aff = np.eye(4)
    nib.save(nib.Nifti1Image(img_np.astype(np.float32), aff), outdir / f"{cid}_img.nii.gz")
    nib.save(nib.Nifti1Image(lbl_np.astype(np.uint8), aff), outdir / f"{cid}_label.nii.gz")
    nib.save(nib.Nifti1Image(pred_np.astype(np.float32), aff), outdir / f"{cid}_pred.nii.gz")

    # Optional visualization (middle slice)
    z = img_np.shape[2] // 2
    plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(img_np[:, :, z], cmap='gray')
    plt.title("Image")
    plt.subplot(1, 3, 2)
    plt.imshow(lbl_np[:, :, z], cmap='gray')
    plt.title("Label")
    plt.subplot(1, 3, 3)
    plt.imshow(pred_np[:, :, z], cmap='hot')
    plt.title("Predicted")
    plt.suptitle(f"{cid} @ z={z}")
    plt.tight_layout()
    plt.savefig(outdir / f"{cid}_preview.png")
    plt.close()

# =============== Setup ===============
cli = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
cli.add_argument("--fold", type=int, default=0)
cli.add_argument("--n_folds", type=int, default=5)
cli.add_argument("--epochs", type=int, default=200)
cli.add_argument("--bs", type=int, default=4)
cli.add_argument("--seed", type=int, default=42)
cli.add_argument("--train_patches_per_epoch", type=int, default=500)
args = cli.parse_args()

ROOT = Path("/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/UNETR")
JSON_PATH = ROOT / "Dataset_UNETR/dataset.json"
PATCH_DIR = ROOT / "Dataset_UNETR/preprocessed_patches_zscore_monaistd_final"
VAL_IMGDIR = ROOT / "Dataset_UNETR/imagesTr"
VAL_LBLDIR = ROOT / "Dataset_UNETR/labelsTr"
OUT_DIR = ROOT / f"Dataset_UNETR/model_output/fold{args.fold}_96"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (96, 96, 96)
SPACING = (0.455, 0.455, 0.4)
AXCODES = "RAS"
N_CLASSES = 2
LR = 1e-4
ACCUM_STEPS = 2
VAL_EVERY_N_EPOCHS = 30         # sliding window interval
MAX_VAL_CASES = 10               # set to None for all cases

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEM = torch.cuda.is_available()

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(args.seed)

train_aug = Compose([
    RandFlipd(keys=["image", "label"], spatial_axis=[0, 1, 2], prob=0.5),
    RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.08),
    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.2),
    RandShiftIntensityd(keys=["image"], offsets=0.05, prob=0.2),
])

infer_img_tx = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes=AXCODES),
    Spacingd(keys=["image"], pixdim=SPACING, mode="bilinear"),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=False),
])
infer_lbl_tx = Compose([
    LoadImaged(keys=["label"]),
    EnsureChannelFirstd(keys=["label"]),
    Orientationd(keys=["label"], axcodes=AXCODES),
    Spacingd(keys=["label"], pixdim=SPACING, mode="nearest"),
])
foreground_dice = DiceMetric(include_background=False, reduction="mean")
precision_metric = ConfusionMatrixMetric(metric_name="precision", include_background=False)
recall_metric = ConfusionMatrixMetric(metric_name="recall", include_background=False)

# =============== Data loading & Patch Assignment ===============
datalist = load_decathlon_datalist(str(JSON_PATH), True, "training")
case_ids = sorted(os.path.basename(rec["image"]).replace(".nii.gz", "") for rec in datalist)

kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
train_idx, val_idx = list(kfold.split(case_ids))[args.fold]
train_cases = {case_ids[i] for i in train_idx}
val_cases = {case_ids[i] for i in val_idx}

def load_pretrained_weights(model, pretrain_path):
    state_dict = torch.load(pretrain_path, map_location=DEV)
    model_state = model.state_dict()
    filtered_dict = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_state and v.shape == model_state[k].shape:
            filtered_dict[k] = v
        else:
            skipped.append(k)
            print(f"Skipping key (no match or shape mismatch): {k}, "
                  f"pretrain: {getattr(v, 'shape', None)}, model: {model_state.get(k, None)}")
    model_state.update(filtered_dict)
    model.load_state_dict(model_state)
    print(f"Loaded {len(filtered_dict)} layers from pretrained weights.")
    print(f"Skipped {len(skipped)} layers due to mismatch.")

def extract_case_id_from_patch(fname: str) -> str:
    m = re.match(r"\d+_(Tr(?:_cropped)?_\d+_0000)_patch(?:_fg\d+|_bg\d+|\d+)\.npz", fname)
    if not m:
        raise ValueError(f"Patch filename does not match expected pattern: {fname}")
    return m.group(1)

all_patches: List[Path] = sorted(PATCH_DIR.glob("*.npz"))
train_patches, val_patches = [], []
for p in all_patches:
    try:
        caseid = extract_case_id_from_patch(p.name)
    except Exception:
        continue
    if caseid in train_cases:
        train_patches.append(p)
    elif caseid in val_cases:
        val_patches.append(p)

def count_fg_bg(patch_list):
    fg_patches, bg_patches, valid_patches = [], [], []
    for p in patch_list:
        arr = np.load(p)
        img_shape = arr["image"].shape
        lbl_shape = arr["label"].shape
        img_spatial = img_shape[-3:] if len(img_shape) == 4 else img_shape
        lbl_spatial = lbl_shape[-3:] if len(lbl_shape) == 4 else lbl_shape
        if img_spatial != IMG_SIZE or lbl_spatial != IMG_SIZE:
            continue
        valid_patches.append(p)
        if arr["label"].sum() > 0:
            fg_patches.append(p)
        else:
            bg_patches.append(p)
    return fg_patches, bg_patches, valid_patches

# Split fg/bg for train and val
fg_train, bg_train, train_patches = count_fg_bg(train_patches)
fg_val, bg_val, val_patches = count_fg_bg(val_patches)

def _assert_non_empty(name, lst):
    if not lst:
        raise RuntimeError(f"{name} split received 0 patches – check filename pattern or fold assignment.")
_assert_non_empty("TRAIN", train_patches)
_assert_non_empty("VAL", val_patches)

# ----------- FIXED VALIDATION SUBSET: 1/4 of val fold, 75% fg, 25% bg ------------
total_val_patches = len(fg_val) + len(bg_val)
val_subset_size = int(0.1 * total_val_patches)
val_fg_subset_size = int(0.75 * val_subset_size)
val_bg_subset_size = val_subset_size - val_fg_subset_size

# Make sure we don't sample more than available
val_fg_subset_size = min(val_fg_subset_size, len(fg_val))
val_bg_subset_size = min(val_bg_subset_size, len(bg_val))

np.random.seed(args.seed + 999)  # fixed for reproducibility
val_fg_subset = np.random.choice(fg_val, val_fg_subset_size, replace=False) if val_fg_subset_size > 0 else []
val_bg_subset = np.random.choice(bg_val, val_bg_subset_size, replace=False) if val_bg_subset_size > 0 else []
fixed_val_patches = np.concatenate([val_fg_subset, val_bg_subset])
np.random.shuffle(fixed_val_patches)  # randomize order, keep fixed

class NPZPatchDataset(Dataset):
    def __init__(self, files: List[Path], augment=False):
        self.files = files
        self.augment = augment
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        f = self.files[idx]
        d = np.load(f)
        img = torch.as_tensor(d["image"], dtype=torch.float32)
        lbl = torch.as_tensor(d["label"], dtype=torch.long)
        if img.ndim == 3:
            img = img.unsqueeze(0)
        if lbl.ndim == 3:
            lbl = lbl.unsqueeze(0)
        lbl = (lbl > 0).long()
        sample = {"image": img, "label": lbl}
        if self.augment and lbl.any():
            sample = train_aug(sample)
        return sample

model = UNETR(
    in_channels=1,
    out_channels=N_CLASSES,
    img_size=IMG_SIZE,
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    proj_type="perceptron",
    norm_name="instance",
    res_block=True,
).to(DEV)

PRETRAIN_PATH = ROOT / "Dataset_UNETR/UNETR_model_best_acc.pth"
if PRETRAIN_PATH.exists():
    print(f"Loading pretrained weights from {PRETRAIN_PATH}")
    load_pretrained_weights(model, PRETRAIN_PATH)
else:
    print(f"No pretrained model found at {PRETRAIN_PATH}, starting from scratch.")

dice_loss = DiceLoss(include_background=False, to_onehot_y=True, softmax=True, reduction="mean")

opt = torch.optim.AdamW(model.parameters(), LR, weight_decay=1e-5)
sched = ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=30, min_lr=1e-6, verbose=True)

def one_hot_5d(lbl, num_classes):
    if lbl.ndim == 5 and lbl.shape[1] == 1:
        lbl = lbl.squeeze(1)
    elif lbl.ndim == 4:
        pass
    else:
        raise RuntimeError(f"Label shape for one-hot: {lbl.shape}")
    return F.one_hot(lbl, num_classes=num_classes).permute(0, 4, 1, 2, 3).contiguous()

def save_prediction(cid, pred, outdir):
    pred_np = pred.argmax(1).cpu().numpy()[0].astype(np.uint8)
    dummy_affine = np.eye(4)
    nib.save(nib.Nifti1Image(pred_np, dummy_affine), outdir / f"{cid}_pred.nii.gz")

def get_first_or_scalar(val):
    if isinstance(val, (list, tuple)) and len(val) > 0:
        return float(val[0])
    elif hasattr(val, 'item'):
        return float(val.item())
    else:
        return float(val)

def compute_voxelwise_rocauc(probs, labels, out_png=None):
    auc = roc_auc_score(labels, probs)
    fpr, tpr, thresholds = roc_curve(labels, probs)
    if out_png is not None:
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC={auc:.3f}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Voxelwise ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(str(out_png), bbox_inches='tight')
        plt.close()
    return auc, fpr, tpr, thresholds

def patch_val(val_patches_sample, out_roc_png=None):
    model.eval()
    foreground_dice.reset()
    precision_metric.reset()
    recall_metric.reset()
    losses, tn_rates = [], []
    fg_pred_list = []
    fg_label_list = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        val_dl = DataLoader(
            NPZPatchDataset(val_patches_sample, augment=False),
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=PIN_MEM,
        )
        for batch in val_dl:
            img, lbl = batch["image"].to(DEV), batch["label"].to(DEV)
            logit = model(img)
            probs = softmax_with_temp(logit, T=0.7)[:, 1].cpu().numpy().flatten()
            labels = (lbl.cpu().numpy().flatten() > 0).astype(np.uint8)
            all_probs.append(probs)
            all_labels.append(labels)
            loss = combo_loss(logit, lbl, dice_loss)
            losses.append(loss.item())
            pred = logit.argmax(1)
            pred_oh = one_hot_5d(pred, N_CLASSES)
            lbl_oh = one_hot_5d(lbl, N_CLASSES)
            foreground_dice(pred_oh, lbl_oh)
            precision_metric(pred_oh, lbl_oh)
            recall_metric(pred_oh, lbl_oh)
            fg_pred = (pred == 1).float().mean().item()
            fg_label = (lbl == 1).float().mean().item()
            fg_pred_list.append(fg_pred)
            fg_label_list.append(fg_label)
            pred_idx, lbl_idx = pred, lbl.squeeze(1)
            tn = ((pred_idx == 0) & (lbl_idx == 0)).sum().item()
            fp = ((pred_idx == 1) & (lbl_idx == 0)).sum().item()
            if tn + fp:
                tn_rates.append(tn / (tn + fp))
    v_loss = float(np.mean(losses))
    v_dice = get_first_or_scalar(foreground_dice.aggregate())
    v_tn = float(np.mean(tn_rates) if tn_rates else np.nan)
    v_precision = get_first_or_scalar(precision_metric.aggregate())
    v_recall = get_first_or_scalar(recall_metric.aggregate())
    avg_fg_pred = float(np.mean(fg_pred_list))
    avg_fg_label = float(np.mean(fg_label_list))
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    auc, fpr, tpr, thresholds = compute_voxelwise_rocauc(all_probs, all_labels, out_png=out_roc_png)
    return v_loss, v_dice, v_tn, v_precision, v_recall, avg_fg_pred, avg_fg_label, auc, (fpr, tpr, thresholds, all_probs, all_labels)

# =============== Training Loop ===============
all_metrics = []
best_val_dice = -1.0
best_roc_data = None

for epoch in range(1, args.epochs + 1):
    model.train()
    ep_loss = 0.0
    fg_means = []

    np.random.seed(args.seed + epoch)
    num_train = min(args.train_patches_per_epoch, len(fg_train) + len(bg_train))
    n_fg = int(num_train * 0.75)
    n_bg = num_train - n_fg

    if len(fg_train) < n_fg or len(bg_train) < n_bg:
        raise RuntimeError(
            f"Not enough fg/bg patches for training epoch {epoch}: "
            f"need at least {n_fg} fg ({len(fg_train)} found), {n_bg} bg ({len(bg_train)} found)"
        )
    train_fg_sample = np.random.choice(fg_train, n_fg, replace=False)
    train_bg_sample = np.random.choice(bg_train, n_bg, replace=False)
    train_sample = np.concatenate([train_fg_sample, train_bg_sample])
    np.random.shuffle(train_sample)

    train_dl = DataLoader(
        NPZPatchDataset(train_sample, augment=True),
        batch_size=args.bs,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=PIN_MEM,
    )
    for batch in train_dl:
        img, lbl = batch["image"].to(DEV), batch["label"].to(DEV)
        logit = model(img)
        loss = combo_loss(logit, lbl, dice_loss) / ACCUM_STEPS
        loss.backward()
        if (len(fg_means) + 1) % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            opt.zero_grad()
        ep_loss += loss.item() * ACCUM_STEPS
        fg_means.append((lbl == 1).float().mean().item())
    ep_loss /= len(train_dl)

    # --- Validation on fixed validation subset ---
    v_loss, v_dice, v_tn, v_precision, v_recall, avg_fg_pred, avg_fg_label, auc, roc_data = patch_val(fixed_val_patches, out_roc_png=None)
    print(
        f"[Epoch {epoch:03d}] train={ep_loss:.5f}  valLoss={v_loss:.5f}  "
        f"valDice={v_dice:.5f}  valTN={v_tn:.5f}  Precision={v_precision:.5f}  Recall={v_recall:.5f}  "
        f"FG_pred={avg_fg_pred:.5f}  FG_label={avg_fg_label:.5f}  ROC-AUC={auc:.5f}  LR={opt.param_groups[0]['lr']:.2e}"
    )
    all_metrics.append({
        "epoch": epoch,
        "train_loss": ep_loss,
        "val_loss": v_loss,
        "val_dice": v_dice,
        "val_tn": v_tn,
        "val_precision": v_precision,
        "val_recall": v_recall,
        "fg_pred": avg_fg_pred,
        "fg_label": avg_fg_label,
        "roc_auc": auc,
    })

    sched.step(v_dice)

    # ---- Save patch prediction every 5 epochs on random val patch ----
    if epoch % 5 == 0 and len(fixed_val_patches) > 0:
        rand_patch_idx = np.random.randint(len(fixed_val_patches))
        rand_patch = fixed_val_patches[rand_patch_idx]
        rand_patch_id = rand_patch.stem
        patch_data = np.load(rand_patch)
        img = torch.as_tensor(patch_data["image"], dtype=torch.float32).unsqueeze(0).to(DEV)
        lbl = torch.as_tensor(patch_data["label"], dtype=torch.long).unsqueeze(0).to(DEV)
        with torch.no_grad():
            logits = model(img)
            pred = logits.argmax(1, keepdim=False)
        save_patch_prediction(img, lbl, pred, OUT_DIR, f"{rand_patch_id}_ep{epoch}")

    if v_dice > best_val_dice:
        best_val_dice = v_dice
        torch.save(model.state_dict(), OUT_DIR / "best_model.pth")
        out_roc_png = OUT_DIR / f"roc_curve_best_valdice.png"
        _, _, _, _, _, _, _, _, roc_data = patch_val(fixed_val_patches, out_roc_png=out_roc_png)
        best_roc_data = roc_data

    if epoch % VAL_EVERY_N_EPOCHS == 0 or epoch == args.epochs:
        model.eval()
        foreground_dice.reset()
        with torch.no_grad():
            for n, cid in enumerate(sorted(val_cases)):
                ip = VAL_IMGDIR / f"{cid}.nii.gz"
                lp = VAL_LBLDIR / f"{cid.rsplit('_0000',1)[0]}.nii.gz"
                img_np = infer_img_tx({"image": str(ip)})["image"].astype(np.float32)
                lbl_np = infer_lbl_tx({"label": str(lp)})["label"].astype(np.uint8)
                pred = sliding_window_inference(
                    torch.tensor(img_np)[None].to(DEV), IMG_SIZE, 4, model, overlap=0.25
                )
                pred_oh = one_hot_5d(pred.argmax(1), N_CLASSES)
                lbl_oh = one_hot_5d(torch.tensor(lbl_np)[None].long().to(DEV), N_CLASSES)
                foreground_dice(pred_oh, lbl_oh)
                if epoch == args.epochs:
                    save_prediction(cid, pred, OUT_DIR)
                if MAX_VAL_CASES is not None and (n + 1) >= MAX_VAL_CASES:
                    break
            full_dice = get_first_or_scalar(foreground_dice.aggregate())
            print(f"[SlidingWindow] FULL Dice={full_dice:.5f}")

    torch.save(model.state_dict(), OUT_DIR / "last_model.pth")

# Write CSV
csv_path = OUT_DIR / "metrics.csv"
fieldnames = [
    "epoch", "train_loss", "val_loss", "val_dice", "val_tn",
    "val_precision", "val_recall", "fg_pred", "fg_label", "roc_auc"
]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_metrics)

print("Training complete. Best val Dice: {:.5f}".format(best_val_dice))
print(f"Metrics saved to {csv_path}")

# =============== Find optimal threshold ===============
if best_roc_data is not None:
    fpr, tpr, thresholds, all_probs, all_labels = best_roc_data
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    best_thresh = thresholds[best_idx]
    print(f"Best ROC-AUC threshold (Youden's J): {best_thresh:.4f}")
    with open(OUT_DIR / "optimal_threshold.txt", "w") as f:
        f.write(f"Best threshold: {best_thresh:.4f}\n")
else:
    print("No ROC data found for best model.")

