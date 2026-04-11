import numpy as np
import cv2


# ImageNet normalization constants (SAM2 convention)
PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
PIXEL_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def preprocess(
    image: np.ndarray,
    input_size: int = 1024,
) -> tuple[np.ndarray, float, float]:
    """
    SAM2 image preprocessing: resize to 1024x1024, normalize with ImageNet stats.

    Args:
        image:      HxWxC BGR uint8 (raw cv2.imread output)
        input_size: model input size (default 1024)

    Returns:
        blob:    [1, 3, 1024, 1024] float32, ImageNet-normalized
        scale_x: width scale factor (input_size / orig_w)
        scale_y: height scale factor (input_size / orig_h)
    """
    orig_h, orig_w = image.shape[:2]
    scale_x = input_size / orig_w
    scale_y = input_size / orig_h

    # Resize to 1024x1024 (no letterbox — SAM2 convention)
    resized = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

    # BGR -> RGB
    rgb = resized[..., ::-1]

    # HWC -> CHW
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)

    # ImageNet normalize: (pixel - mean) / std
    chw[0] = (chw[0] - PIXEL_MEAN[0]) / PIXEL_STD[0] # Red channel
    chw[1] = (chw[1] - PIXEL_MEAN[1]) / PIXEL_STD[1] # Green channel
    chw[2] = (chw[2] - PIXEL_MEAN[2]) / PIXEL_STD[2] # Blue channel

    # Add batch dimension
    blob = chw[np.newaxis, ...]

    return blob, scale_x, scale_y


def transform_coords(
    coords: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    """
    Scale point coordinates from original image space to 1024x1024 model space.

    Args:
        coords:  [N, 2] array of (x, y) points
        scale_x: width scale factor
        scale_y: height scale factor

    Returns:
        scaled: [N, 2] array in model space
    """
    scaled = coords.copy().astype(np.float32)
    scaled[:, 0] *= scale_x
    scaled[:, 1] *= scale_y
    return scaled


def transform_boxes(
    boxes: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert xyxy boxes to SAM2 point_coords/point_labels format.

    Each box becomes two points: top-left (label=2) and bottom-right (label=3).

    Args:
        boxes:   [N, 4] array of (x1, y1, x2, y2) in original image coords
        scale_x: width scale factor
        scale_y: height scale factor

    Returns:
        point_coords: [N, 2, 2] float32 in model space
        point_labels: [N, 2] float32 (values: 2=top-left, 3=bottom-right)
    """
    n = boxes.shape[0]
    point_coords = np.zeros((n, 2, 2), dtype=np.float32)

    # Top-left corner
    point_coords[:, 0, 0] = boxes[:, 0] * scale_x
    point_coords[:, 0, 1] = boxes[:, 1] * scale_y
    # Bottom-right corner
    point_coords[:, 1, 0] = boxes[:, 2] * scale_x
    point_coords[:, 1, 1] = boxes[:, 3] * scale_y

    # Labels: 2 = top-left, 3 = bottom-right
    point_labels = np.full((n, 2), 2.0, dtype=np.float32)
    point_labels[:, 1] = 3.0

    return point_coords, point_labels
