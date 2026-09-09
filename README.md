# Histopathology Nuclei Analysis

A deep learning pipeline for nuclei segmentation, instance separation, and quantitative analysis of histopathology images using U-Net and watershed-based post-processing.

## Overview

This project performs automated nuclei analysis on H&E-stained histopathology images.

The pipeline includes:

- XML annotation to binary mask conversion
- Image patch extraction
- U-Net based nuclei segmentation
- Dice + Binary Cross-Entropy loss
- Pixel-level segmentation evaluation
- Watershed-based separation of touching nuclei
- Nuclei detection and morphological quantification

The project was developed using the MoNuSeg 2018 histopathology dataset.

## Pipeline

```text
Histopathology Image
        ↓
XML Annotations
        ↓
Binary Nuclear Masks
        ↓
256 × 256 Image Patches
        ↓
U-Net Segmentation
        ↓
Binary Nuclear Probability Map
        ↓
Thresholding + Morphological Processing
        ↓
Distance Transform
        ↓
Watershed Instance Separation
        ↓
Individual Nuclei
        ↓
Quantitative Analysis