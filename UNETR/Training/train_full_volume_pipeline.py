import os
import argparse
import numpy as np
import pickle
from tqdm import tqdm
import torch
from monai.data import DataLoader, Dataset
from monai.networks.nets import UNETR
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose, RandCropByPosNegLabeld, RandFlipd, RandScaleIntensityd, RandShiftIntensityd,
    AsDiscrete, SpatialPadd
)
from sklearn.model_selection import KFold
import torch.nn.functional as F
import random

# ----------- Config -----------
IMG_SIZE = (96, 96, 96)
N_CLASSES = 2
BATCH_SIZE = 2
EPOCHS = 300
NUM_FOLDS = 5
POS_PER_IMAGE = 2
NEG_PER_IMAGE = 1
NUM_SAMPLES_TRAIN = 4
NUM_SAMPLES_VAL = 1

def dice_ce_loss(logits, targets, soft_dice):
    dice = soft_dice(logits, targets)
    ce = F.cross_entropy(logits, targets.squeeze(1).long())
    return 0.5 * dice + 0.5 * ce

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def npy_loader(img_path, lbl_path, pkl_path):
    img = np.load(img_path)
    lbl = np.load(lbl_path) if lbl_path and os.path.exists(lbl_path) else None
    with open(pkl_path, "rb") as f:
        meta = pickle.load(f)
    if img.ndim == 3:
        img = img[None]
    if lbl is not None and lbl.ndim == 3:
        lbl = lbl[None]
    return {"image": img, "label": lbl, "meta": meta}

def list_cases(data_root):
    npy_files = sorted([f for f in os.listdir(data_root) if f.endswith(".npy") and "_seg" not in f])
    cases = []
    for npy_file in npy_files:
        case_id = npy_file.replace(".npy", "")
        img_path = os.path.join(data_root, f"{case_id}.npy")
        lbl_path = os.path.join(data_root, f"{case_id}_seg.npy")
        pkl_path = os.path.join(data_root, f"{case_id}.pkl")
        if not os.path.exists(pkl_path):
            continue
        case = {"image": img_path, "label": lbl_path, "meta": pkl_path, "id": case_id}
        cases.append(case)
    return cases

def load_partial_state_dict(model, pretrained_path, num_target_classes):
    print(f"Loading BTCV checkpoint from: {pretrained_path}")
    state_dict = torch.load(pretrained_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    filtered = {}
    own_state = model.state_dict()
    for k, v in state_dict.items():
        k_clean = k.replace("module.", "")
        if k_clean.startswith('out.') and own_state[k_clean].shape != v.shape:
            continue
        if k_clean in own_state and own_state[k_clean].shape == v.shape:
            filtered[k_clean] = v
    missing = set(own_state.keys()) - set(filtered.keys())
    print(f"Loaded {len(filtered)} layers from BTCV, skipped {len(missing)} (including final head).")
    model.load_state_dict(filtered, strict=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to dir with .npy/.pkl")
    parser.add_argument("--fold", type=int, required=True, help="0-4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--bs", type=int, default=BATCH_SIZE)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--btc_checkpoint", type=str, required=True, help="Path to BTCV .pth checkpoint")
    args = parser.parse_args()

    set_seed(args.seed)
    all_cases = list_cases(args.data_root)
    all_ids = [case["id"] for case in all_cases]
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=args.seed)
    splits = list(kf.split(all_ids))
    train_idx, val_idx = splits[args.fold]
    train_cases = [all_cases[i] for i in train_idx]
    val_cases = [all_cases[i] for i in val_idx]
    print(f"Fold {args.fold}: train {len(train_cases)}, val {len(val_cases)}")

    train_tf = Compose([
        SpatialPadd(keys=["image", "label"], spatial_size=IMG_SIZE),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=IMG_SIZE,
            pos=POS_PER_IMAGE,
            neg=NEG_PER_IMAGE,
            num_samples=NUM_SAMPLES_TRAIN,
            image_key="image",
            image_threshold=0,
            allow_smaller=True,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[1]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[2]),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.2),
        RandShiftIntensityd(keys=["image"], offsets=0.05, prob=0.2),
    ])
    val_tf = Compose([
        SpatialPadd(keys=["image", "label"], spatial_size=IMG_SIZE),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=IMG_SIZE,
            pos=POS_PER_IMAGE,
            neg=NEG_PER_IMAGE,
            num_samples=NUM_SAMPLES_VAL,
            image_key="image",
            image_threshold=0,
            allow_smaller=True,
        ),
    ])

    class NPYDataset(Dataset):
        def __init__(self, cases, transform=None):
            self.cases = cases
            self.transform = transform
        def __getitem__(self, idx):
            data = npy_loader(self.cases[idx]["image"], self.cases[idx]["label"], self.cases[idx]["meta"])
            sample = {"image": data["image"], "label": data["label"]}
            sample["image"] = torch.tensor(sample["image"], dtype=torch.float32)
            if sample["label"] is not None:
                sample["label"] = torch.tensor(sample["label"], dtype=torch.uint8)
            if self.transform:
                sample = self.transform(sample)
            return sample
        def __len__(self):
            return len(self.cases)

    train_ds = NPYDataset(train_cases, train_tf)
    val_ds = NPYDataset(val_cases, val_tf)
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNETR(
        in_channels=1, out_channels=N_CLASSES, img_size=IMG_SIZE,
        feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12,
        proj_type="perceptron", norm_name="instance", res_block=True
    ).to(device)

    load_partial_state_dict(model, args.btc_checkpoint, N_CLASSES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6, verbose=True)
    dice_loss = DiceLoss(to_onehot_y=True, softmax=True, include_background=False)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = AsDiscrete(argmax=True, to_onehot=N_CLASSES)
    post_label = AsDiscrete(to_onehot=N_CLASSES)
    best_val_loss = float('inf')
    patience_counter = 0
    EARLY_STOPPING_PATIENCE = 20
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
            imgs, lbls = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = dice_ce_loss(logits, lbls, dice_loss)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                imgs, lbls = batch["image"].to(device), batch["label"].to(device)
                logits = model(imgs)
                loss = dice_ce_loss(logits, lbls, dice_loss)
                val_loss += loss.item()
                preds = post_pred(logits)
                labels = post_label(lbls)
                dice_metric(preds, labels)
            val_loss /= len(val_loader)
            val_dice = dice_metric.aggregate().item()
            dice_metric.reset()
        scheduler.step(val_loss)
        print(f"[Fold {args.fold} Epoch {epoch:03d}] train_loss: {train_loss:.5f} | val_loss: {val_loss:.5f} | val_dice: {val_dice:.5f} | lr: {optimizer.param_groups[0]['lr']:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"best_model_fold{args.fold}.pth"))
            print(f"Best model updated at epoch {epoch:03d}.")
        else:
            patience_counter += 1
            print(f"No improvement in val_loss for {patience_counter} epochs.")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch:03d}. Best val_loss: {best_val_loss:.5f}")
                break
    print(f"Training complete. Best val Loss: {best_val_loss:.5f}")

if __name__ == "__main__":
    main()

