import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from model import UNet
from dataset import NucleiDataset


def dice_loss(pred, target):
    pred = torch.sigmoid(pred)

    smooth = 1e-6
    intersection = (pred * target).sum(dim=(1, 2, 3))

    union = (
        pred.sum(dim=(1, 2, 3)) +
        target.sum(dim=(1, 2, 3))
    )

    dice = (2 * intersection + smooth) / (union + smooth)

    return 1 - dice.mean()


bce_loss = nn.BCEWithLogitsLoss()


def combined_loss(pred, target):
    return bce_loss(pred, target) + dice_loss(pred, target)


def calculate_metrics(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()

    intersection = (pred * target).sum(dim=(1, 2, 3))

    dice = (
        (2 * intersection + 1e-6) /
        (
            pred.sum(dim=(1, 2, 3)) +
            target.sum(dim=(1, 2, 3)) +
            1e-6
        )
    )

    union = (
        pred.sum(dim=(1, 2, 3)) +
        target.sum(dim=(1, 2, 3)) -
        intersection
    )

    iou = (intersection + 1e-6) / (union + 1e-6)

    return dice.mean().item(), iou.mean().item()


def train_model(
    train_image_dir,
    train_mask_dir,
    val_image_dir,
    val_mask_dir,
    model_path,
    epochs=20,
    batch_size=8,
    learning_rate=1e-4
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = NucleiDataset(
        train_image_dir,
        train_mask_dir
    )

    val_dataset = NucleiDataset(
        val_image_dir,
        val_mask_dir
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model = UNet().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    train_losses = []
    val_losses = []
    val_dices = []
    val_ious = []

    best_dice = 0.0

    for epoch in range(epochs):

        model.train()
        running_train_loss = 0.0

        for images, masks in train_loader:

            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = combined_loss(outputs, masks)

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        train_loss = running_train_loss / len(train_loader)

        model.eval()
        running_val_loss = 0.0
        dice_scores = []
        iou_scores = []

        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)

                loss = combined_loss(outputs, masks)
                running_val_loss += loss.item()

                dice, iou = calculate_metrics(
                    outputs,
                    masks
                )

                dice_scores.append(dice)
                iou_scores.append(iou)

        val_loss = running_val_loss / len(val_loader)
        val_dice = np.mean(dice_scores)
        val_iou = np.mean(iou_scores)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        val_ious.append(val_iou)

        if val_dice > best_dice:

            best_dice = val_dice

            torch.save(
                model.state_dict(),
                model_path
            )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Val IoU: {val_iou:.4f}"
        )

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_dices": val_dices,
        "val_ious": val_ious,
        "best_dice": best_dice,
        "device": device
    }