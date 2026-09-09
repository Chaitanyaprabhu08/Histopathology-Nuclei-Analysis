import numpy as np
import torch


def evaluate_model(model, test_loader, device):
    model.eval()

    all_predictions = []
    all_masks = []

    with torch.no_grad():
        for images, masks in test_loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            predictions = (
                torch.sigmoid(outputs) > 0.5
            ).float()

            all_predictions.append(
                predictions.cpu().numpy()
            )

            all_masks.append(
                masks.cpu().numpy()
            )

    predictions = np.concatenate(all_predictions)
    masks = np.concatenate(all_masks)

    pred = predictions.astype(np.uint8).flatten()
    true = masks.astype(np.uint8).flatten()

    intersection = np.sum(pred * true)

    pred_sum = np.sum(pred)
    true_sum = np.sum(true)

    dice = (
        2 * intersection + 1e-6
    ) / (
        pred_sum + true_sum + 1e-6
    )

    union = pred_sum + true_sum - intersection

    iou = (
        intersection + 1e-6
    ) / (
        union + 1e-6
    )

    tp = np.sum((pred == 1) & (true == 1))
    tn = np.sum((pred == 0) & (true == 0))
    fp = np.sum((pred == 1) & (true == 0))
    fn = np.sum((pred == 0) & (true == 1))

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    specificity = tn / (tn + fp + 1e-6)

    accuracy = (
        (tp + tn) /
        (tp + tn + fp + fn + 1e-6)
    )

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy
    }