import numpy as np
import cv2


def letterbox(
    image: np.ndarray,
    target_size: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (0, 0, 0),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Resize image with unchanged aspect ratio using padding.

    Args:
        image:       HxWxC BGR uint8
        target_size: (width, height) target canvas
        color:       padding color (SCRFD convention: black)

    Returns:
        padded:  target_size HxWxC BGR uint8
        scale:   scale factor applied to original image
        pad:     (pad_w, pad_h) padding added to each side
    """
    src_h, src_w = image.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / src_w, target_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (target_w - new_w) / 2
    pad_h = (target_h - new_h) / 2

    top    = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left   = int(round(pad_w - 0.1))
    right  = int(round(pad_w + 0.1))

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )

    return padded, scale, (left, top)


def preprocess(
    image: np.ndarray,
    target_size: tuple[int, int] = (640, 640),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Full SCRFD preprocessing pipeline.

    Args:
        image:       HxWxC BGR uint8  (raw cv2.imread output)
        target_size: model input size, default 640x640

    Returns:
        blob:    [1, 3, H, W] float32 normalized with (pixel - 127.5) / 128.0
        scale:   scale factor (used in postprocess to map boxes back)
        pad:     (pad_w, pad_h) padding offsets (used in postprocess)
    """
    padded, scale, pad = letterbox(image, target_size)

    # BGR -> RGB
    rgb = padded[..., ::-1]

    # HWC -> CHW
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))

    # Normalize: (pixel - 127.5) / 128.0
    normalized = (chw.astype(np.float32) - 127.5) / 128.0

    # Add batch dimension -> [1, 3, H, W]
    blob = normalized[np.newaxis, ...]

    return blob, scale, pad
