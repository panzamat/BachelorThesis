import os
import json
import random
from pathlib import Path

# === Input folders ===
data_root = Path("/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/UNETR/Dataset_UNETR")
imagesTr = data_root / "imagesTr"
imagesTs = data_root / "imagesTs"
labelsTr = data_root / "labelsTr"
labelsTs = data_root / "labelsTs"

# === Collect valid training files ===
all_train_files = sorted([f.name for f in imagesTr.glob("*.nii.gz")])
tr_all = [f for f in all_train_files if f.startswith("Tr_") and not f.startswith("Tr_cropped_")]
cropped_all = [f for f in all_train_files if f.startswith("Tr_cropped_")]

# === Shuffle and split ===
random.shuffle(tr_all)
random.shuffle(cropped_all)
tr_split = int(0.8 * len(tr_all))
cropped_split = int(0.8 * len(cropped_all))

tr_train, tr_val = tr_all[:tr_split], tr_all[tr_split:]
cropped_train, cropped_val = cropped_all[:cropped_split], cropped_all[cropped_split:]

# === Build entries ===
def build_entries(files, folder):
    entries = []
    for f in files:
        label_name = f.replace("_0000", "")
        label_path = data_root / "labelsTr" / label_name
        if label_path.exists():
            entries.append({
                "image": f"{folder}/{f}",
                "label": f"labelsTr/{label_name}"
            })
        else:
            print(f"⚠️ Missing label for {f}, skipping.")
    return entries

training = build_entries(tr_train + cropped_train, "imagesTr")
validation = build_entries(tr_val + cropped_val, "imagesTr")

# === Collect test files ===
test_files = sorted([f.name for f in imagesTs.glob("*.nii.gz")])

def build_test_entries(files):
    entries = []
    for f in files:
        label_name = f.replace("_0000", "")
        label_path = data_root / "labelsTs" / label_name
        if label_path.exists():
            entries.append({
                "image": f"imagesTs/{f}",
                "label": f"labelsTs/{label_name}"
            })
        else:
            print(f"⚠️ Missing test label for {f}, skipping.")
    return entries


testing = build_test_entries(test_files)

# === Final JSON ===
data_dict = {
    "name": "aneu_mr",
    "description": "Aneurysm multimodal MR dataset",
    "tensorImageSize": "3D",
    "reference": "ZHAW Comp-Health Lab",
    "licence": "ZHAW internal",
    "release": "1.0 05/2025",
    "modality": {
        "0": "CTMR"
    },
    "labels": {
        "0": "background",
        "1": "aneurysm"
    },
    "numTraining": len(training),
    "numTest": len(testing),
    "training": training,
    "validation": validation,
    "testing": testing
}

# === Save ===
output_path = data_root / "dataset2.json"
with open(output_path, "w") as f:
    json.dump(data_dict, f, indent=2)

print(f"✅ Saved dataset2.json to {output_path}")
print(f"📊 Training: {len(training)} | Validation: {len(validation)} | Testing: {len(testing)}")

