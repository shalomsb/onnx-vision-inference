import numpy as np
import cv2


def letterbox(
    image: np.ndarray,
    target_size: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Resize image with unchanged aspect ratio using padding.

    Args:
        image:       HxWxC BGR uint8
        target_size: (width, height) target canvas
        color:       padding color (YOLO standard: 114)

    Returns:
        padded:  target_size HxWxC BGR uint8
        scale:   scale factor applied to original image
        pad:     (pad_w, pad_h) padding added to each side
    """
    src_h, src_w = image.shape[:2]
    target_w, target_h = target_size

    # Scale to fit inside target canvas, preserve aspect ratio
    # For example, if src is 1280x720 and target is 640x640, we get w_scale=0.5 and h_scale=0.888.
    # we take the smaller one to ensure the whole image fits inside target -> scale=0.5, new size is 640x360.
    # if we took the larger one (0.888), we would get 1137*640, which exceeds target size.
    scale = min(target_w / src_w, target_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Padding to reach target canvas
    # notice: only one of pad_w or pad_h is zero, because we scale to fit inside target.
    pad_w = (target_w - new_w) / 2
    pad_h = (target_h - new_h) / 2

    # if pad_h is 141, for example, we get each side 70.5. after round we get 70 and 70.
    # 70 + 70 = 140, so we are 1 pixel short of target. to fix this, we can add a small epsilon before rounding.
    # so we get 70.5 + 0.1 = 70.6 → round to 71, and 70.5 - 0.1 = 70.4 → round to 70, now we have 71 + 70 = 141.
    top    = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left   = int(round(pad_w - 0.1))
    right  = int(round(pad_w + 0.1))

    # we take the resized image and pad it to the target size. 
    # for example, we take the 640x360 resized image and add 70 pixels top and bottom,
    # and 0 pixels left and right, to get a 640x640 padded image.
    # args: the resized image before padding, the number of pixels to pad on each side, the padding color (BGR)
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
    Full YOLO preprocessing pipeline.

    Args:
        image:       HxWxC BGR uint8  (raw cv2.imread output)
        target_size: model input size, default 640x640

    Returns:
        blob:    [1, 3, H, W] float32 in [0.0, 1.0]  — ready for onnxruntime
        scale:   scale factor (used in postprocess to map boxes back)
        pad:     (pad_w, pad_h) padding offsets (used in postprocess)
    """
    padded, scale, pad = letterbox(image, target_size)

    # BGR → RGB without copy (padded is HxWx3 BGR uint8, we just change the view to RGB by reversing the last channel)
    rgb = padded[..., ::-1]  # HxWx3, view (no copy)

    # HWC → CHW with copy (we need to transpose the array and make it contiguous in memory for onnxruntime)
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))  # 3xHxW

    # uint8 → float32, normalize to [0, 1]
    normalized = chw.astype(np.float32) / 255.0

    # Add batch dimension → [1, 3, H, W]
    blob = normalized[np.newaxis, ...]

    return blob, scale, pad
