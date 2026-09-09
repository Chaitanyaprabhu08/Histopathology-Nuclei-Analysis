import cv2
import numpy as np
import pandas as pd
import torch

from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects


def quantify_nuclei(model, test_loader, device):

    model.eval()

    results = []

    with torch.no_grad():

        for images, masks in test_loader:

            images = images.to(device)

            outputs = model(images)

            predictions = torch.sigmoid(
                outputs
            ).cpu().numpy()

            for i in range(len(images)):

                binary_prediction = (
                    predictions[i, 0] > 0.5
                )

                clean_mask = remove_small_objects(
                    binary_prediction,
                    min_size=50
                )

                distance = cv2.distanceTransform(
                    clean_mask.astype(np.uint8),
                    cv2.DIST_L2,
                    5
                )

                coordinates = peak_local_max(
                    distance,
                    min_distance=8,
                    threshold_abs=2
                )

                markers = np.zeros(
                    distance.shape,
                    dtype=np.int32
                )

                for j, (y, x) in enumerate(
                    coordinates,
                    start=1
                ):
                    markers[y, x] = j

                labels = watershed(
                    -distance,
                    markers,
                    mask=clean_mask
                )

                regions = regionprops(labels)

                areas = [r.area for r in regions]
                perimeters = [
                    r.perimeter for r in regions
                ]

                circularities = []

                for area, perimeter in zip(
                    areas,
                    perimeters
                ):
                    if perimeter > 0:
                        circularities.append(
                            (4 * np.pi * area) /
                            (perimeter ** 2)
                        )

                results.append({
                    "patch_index": len(results) + 1,
                    "nuclei_count": len(regions),
                    "mean_area": np.mean(areas)
                    if areas else 0,
                    "median_area": np.median(areas)
                    if areas else 0,
                    "mean_perimeter": np.mean(perimeters)
                    if perimeters else 0,
                    "mean_circularity":
                        np.mean(circularities)
                        if circularities else 0,
                    "total_nuclear_area":
                        np.sum(areas)
                        if areas else 0
                })

    return pd.DataFrame(results)