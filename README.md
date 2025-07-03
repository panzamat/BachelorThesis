# Bachelor Thesis: Multimodal Deep Learning for Automatic Segmentation of Intracranial Aneurysms

This repository contains all source code, figures and needed environment requirements for the Bachelor's thesis **"Multimodal Deep Learning for Automatic Segmentation of Intracranial Aneurysms"** 

## Thesis Overview

This work investigates the potential of deep learning models, particularly nnU-Net and UNETR,for automatic IA segmentation from CTA and MRA images by exploring unimodal and multimodal approaches.

## Setup Instructions
### For nnU-Net:

```bash
# Create and activate a new Conda environment
conda create -n nnunet_env python=3.12.4 -y
conda activate nnunet_env

# Install dependencies
pip install -r environment.yml
```
### For UNETR:
```bash
# Create and activate a new Conda environment
conda create -n unetr_env python=3.10 -y
conda activate unetr_env

# Install dependencies
pip install -r environment-dev.yml
```

## Model Weights
Due to file size limitations, trained model weights are provided externally:

**Google Drive**  
[Link to model weights](https://drive.google.com/drive/folders/18eRfRkyUcop7-Lu-25G0-qgmonbOwtwO?usp=sharing)

## References

- Hatamizadeh, A., Tang, Y., Nath, V., Yang, D., Myronenko, A., Landman, B., Roth, H., & Xu, D. (2021). [UNETR: Transformers for 3D Medical Image Segmentation](http://arxiv.org/abs/2103.10504). *arXiv preprint arXiv:2103.10504*. https://doi.org/10.48550/arXiv.2103.10504

- Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., & Maier-Hein, K. H. (2021). [nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation](https://www.nature.com/articles/s41592-020-01008-z). *Nature Methods, 18*(2), 203–211. https://doi.org/10.1038/s41592-020-01008-z


