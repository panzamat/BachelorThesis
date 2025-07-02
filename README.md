# BachelorThesis: Multimodal Deep Learning for Automatic Segmentation of Intracranial Aneurysms

This repository contains the code and documentation for the Bachelor's thesis

## 📌 Overview

Intracranial aneurysms (IAs) are pathological dilations of cerebral arteries and pose a major health risk if ruptured. This thesis explores deep learning methods for automatic segmentation of IAs in 3D medical images, comparing **unimodal** and **multimodal** approaches using CTA and MRA data.

### Objectives

- Evaluate nnU-Net models on unimodal (CTA/MRA) and multimodal datasets.
- Investigate whether multimodal inputs improve segmentation quality.
- Benchmark transformer-based UNETR models and analyze their limitations.
- Compare results with established models such as GLIA-Net.

## Methods

- **Data**:
  - 1,435 CTA images
  - 282 MRA images
- **Models**:
  - 2 unimodal nnU-Net models (CTA, MRA)
  - 2 multimodal nnU-Net models (baseline and refined)
  - 2 multimodal UNETR models
- **Evaluation**:
  - Target-wise (IoU, Recall, Precision, etc.)
  - Voxel-wise (DSC, TP, FP, FN, etc.)
  - Confidence interval analysis
  - Failure analysis and diagnostic visualizations

## Key Findings

- **Unimodal nnU-Net models** are strong and reliable baselines.
- **Multimodal nnU-Net models** did not improve target-wise detection, but the refined model showed **better voxel-wise segmentation** (especially vs. MRA).
- **UNETR models failed** to produce true positives, highlighting challenges in highly imbalanced data and model training stability.
- Compared to **GLIA-Net**, the nnU-Net models achieved **higher precision and DSC**, but lower recall.
- Model performance varied notably due to **randomness and fold instability**, especially in MRA folds.

## Repository Structure


## Model Weights
 - Model weights are stored in a Google Drive folder due to size constrictions of GitHub. The folder can be found under: https://drive.google.com/drive/folders/18eRfRkyUcop7-Lu-25G0-qgmonbOwtwO?usp=sharing
