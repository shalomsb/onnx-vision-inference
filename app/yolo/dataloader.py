import os
import glob
import cv2
import numpy as np

from .preprocess import preprocess


def load_images(images_dir: str = None, image: str = None) -> list[np.ndarray]:
    """Load images from a directory or a single image path."""
    images = []
    if images_dir:
        paths = sorted(glob.glob(os.path.join(images_dir, "*")))
        for p in paths:
            img = cv2.imread(p)
            if img is not None:
                images.append(img)
    elif image:
        img = cv2.imread(image)
        if img is not None:
            images.append(img)
    return images


def create_batch(
    images: list[np.ndarray],
    target_size: tuple[int, int] = (640, 640),
) -> tuple[np.ndarray, list[float], list[tuple[int, int]]]:
    """
    Preprocess images and stack into a batch.

    Returns:
        batch:  [N, 3, H, W] float32 — ready for onnxruntime
        scales: list of scale factors per image
        pads:   list of (pad_w, pad_h) per image
    """
    blobs, scales, pads = [], [], []
    for img in images:
        blob, scale, pad = preprocess(img, target_size)
        blobs.append(blob)
        scales.append(scale)
        pads.append(pad)

    batch = np.concatenate(blobs, axis=0)  # [N, 3, H, W]
    return batch, scales, pads
