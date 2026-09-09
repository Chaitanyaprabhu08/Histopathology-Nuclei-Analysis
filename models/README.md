# Trained Model

The trained U-Net model is not included directly in this repository because the model file is approximately 118 MB.

The model was trained using the MoNuSeg histopathology nuclei dataset.

Model details:

- Architecture: U-Net
- Framework: PyTorch
- Input size: 256 × 256
- Output: Binary nuclei segmentation
- Loss: BCE + Dice Loss
- Best validation Dice: 0.7949
- Final test Dice: 0.8182
- Final test IoU: 0.6924

The trained model can be added separately using a suitable model-hosting or release mechanism.