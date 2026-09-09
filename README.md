# Histopathology Nuclei Analysis

A deep learning pipeline for nuclei segmentation and quantitative analysis of histopathology images using U-Net and watershed-based instance separation.

## Overview

This project performs automated nuclei analysis on H&E-stained histopathology images.

The pipeline includes:

- XML annotation processing
- Ground-truth mask generation
- Histopathology image patch extraction
- U-Net based nuclei segmentation
- Dice and IoU based evaluation
- Watershed-based separation of touching nuclei
- Nuclear morphology quantification

## Dataset

The project uses the MoNuSeg histopathology nuclei segmentation dataset.

The dataset contains H&E-stained tissue images with manually annotated nuclear boundaries.

The data was divided at the original-image level to avoid data leakage:

- Training: 25 images
- Validation: 5 images
- Test: 7 images

256 × 256 patches were generated from the images.

## Model

A standard U-Net architecture was implemented using PyTorch for binary nuclei segmentation.

The model uses:

- Convolutional encoder
- Bottleneck layers
- Transpose-convolution decoder
- Skip connections
- Batch normalization
- ReLU activation

The training objective combines Binary Cross Entropy and Dice loss.

## Results

Final performance on the held-out test set:

| Metric | Score |
|---|---:|
| Dice | 0.8182 |
| IoU | 0.6924 |
| Precision | 0.7842 |
| Recall | 0.8554 |
| Specificity | 0.9316 |
| Pixel Accuracy | 0.9144 |

## Nuclei Quantification

Watershed post-processing was applied to separate touching nuclei.

Results across 63 test patches:

- Total detected nuclei: 1,570
- Average nuclei per patch: 24.92
- Average nuclear area: 575.41 px²
- Average perimeter: 91.85 px
- Average circularity: 0.7981
- Total nuclear area: 903,831 px²

## Project Structure

```text
Histopathology-Nuclei-Analysis/
│
├── models/
├── notebooks/
│   └── histopathology_nuclei_analysis.ipynb
│
├── results/
│   ├── nuclei_quantification_test.csv
│   └── figures/
│       └── test_nuclei_analysis.png
│
├── src/
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   ├── quantification.py
│   └── train.py
│
├── .gitignore
├── README.md
└── requirements.txt