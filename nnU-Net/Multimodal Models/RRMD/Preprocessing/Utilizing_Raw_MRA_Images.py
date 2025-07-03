import os
import subprocess
from pathlib import Path
import nibabel as nib
import numpy as np

LABELS_DIR = Path("/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/nnUNet_data/nnUNet_raw/Dataset060_IA/labelsTr")
SOURCE_ROOT = Path("/cfs/earth/scratch/panzamat/ds003949/ds003949")
DEST_DIR = Path("/cfs/earth/scratch/icls/shared/comp-health-lab/data/aneu/nnUNet_Frame/Mattia_BSc_Thesis/multimodal_data/nnUNet_data/nnUNet_raw/imagesTr_correct")

DEST_DIR.mkdir(parents=True, exist_ok=True)

def load_props(path):
    img = nib.load(str(path))
    return img.shape, img.header.get_zooms(), img.affine[:3, 3]

for label_file in LABELS_DIR.glob("Tr_*.nii.gz"):
    subject_id = label_file.name.replace("Tr_", "").replace(".nii.gz", "")
    subject_dir = SOURCE_ROOT / f"sub-{subject_id}"
    if not subject_dir.exists():
        print(f"Subject dir not found: {subject_dir}")
        continue

    label_shape, label_spacing, label_origin = load_props(label_file)
    matched = False

    for ses in sorted(subject_dir.glob("ses-*")):
        anat_dir = ses / "anat"
        if not anat_dir.exists():
            continue

        for img_file in anat_dir.glob("*angio*.nii.gz"):
            try:
                shape, spacing, origin = load_props(img_file)
                if shape == label_shape and spacing == label_spacing and np.allclose(origin, label_origin):
                    dest_path = DEST_DIR / f"Tr_{subject_id}_0000.nii.gz"
                    subprocess.run(["cp", str(img_file), str(dest_path)], check=True)
                    print(f"Copied: {img_file} -> {dest_path}")
                    matched = True
                    break
            except Exception as e:
                print(f"Failed to process {img_file}: {e}")

        if matched:
            break

    if not matched:
        print(f"No match found for subject {subject_id}")

