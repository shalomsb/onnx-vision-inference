import numpy as np
import cv2
from transformers import AutoTokenizer


# Grounding DINO input resolution (HuggingFace ONNX export)
INPUT_SIZE = 800

# ImageNet normalization (applied after /255)
PIXEL_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
PIXEL_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def preprocess_image(
    image: np.ndarray,
    input_size: int = INPUT_SIZE,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Resize to 800x800 and ImageNet-normalize.

    Args:
        image:      HxWxC BGR uint8
        input_size: model input size (800)

    Returns:
        blob:       [1, 3, 800, 800] float32, ImageNet-normalized
        pixel_mask: [1, 800, 800] int64 (all ones)
        orig_size:  (orig_h, orig_w)
    """
    orig_h, orig_w = image.shape[:2]

    resized = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

    # BGR -> RGB
    rgb = resized[..., ::-1]

    # HWC -> CHW, uint8 -> float32, normalize
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
    chw = (chw - PIXEL_MEAN) / PIXEL_STD

    blob = chw[np.newaxis, ...]  # [1, 3, 800, 800]

    # pixel_mask: all ones (no padding — we resize the full image)
    pixel_mask = np.ones((1, input_size, input_size), dtype=np.int64)

    return blob, pixel_mask, (orig_h, orig_w)


def load_tokenizer(tokenizer_name: str = "bert-base-uncased"):
    """Load a BERT tokenizer for text preprocessing."""
    return AutoTokenizer.from_pretrained(tokenizer_name)


def _clean_text(text: str) -> str:
    """Normalize grounding prompt: lowercase, ensure trailing dot."""
    text = text.strip().lower()
    if not text.endswith("."):
        text += " ."
    return text


def preprocess_text(
    tokenizer,
    text: str,
) -> dict[str, np.ndarray]:
    """
    Tokenize a grounding prompt and build text ONNX inputs.

    Args:
        tokenizer:  AutoTokenizer instance
        text:       grounding prompt, e.g. "cat . dog ."

    Returns:
        dict with:
            input_ids:      int64 [1, L]
            token_type_ids: int64 [1, L]
            attention_mask: int64 [1, L]
    """
    text = _clean_text(text)
    enc = tokenizer(
        text,
        return_tensors="np",
    )

    input_ids = enc["input_ids"].astype(np.int64)
    attention_mask = enc["attention_mask"].astype(np.int64)
    token_type_ids = enc.get(
        "token_type_ids", np.zeros_like(input_ids)
    ).astype(np.int64)

    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
    }


def map_token_to_category(
    tokenizer,
    text: str,
    token_ids: np.ndarray,
) -> list[str]:
    """
    Map token indices back to category names from the grounding prompt.

    For prompt "person . car .", tokens are like:
        [CLS] person . car . [SEP]
        0     1      2  3   4  5
    Token 1 → "person", token 3 → "car"

    Args:
        tokenizer: AutoTokenizer instance
        text:      original grounding prompt
        token_ids: [N] int array of token indices from postprocess

    Returns:
        list of category name strings, one per detection
    """
    text = _clean_text(text)
    tokens = tokenizer.tokenize(text)

    # Build mapping: token position → category name
    # Categories are separated by "." in the prompt
    # Token positions are offset by 1 because of [CLS]
    token_to_cat = {}
    current_cat_tokens = []
    cat_start_idx = 0

    for i, tok in enumerate(tokens):
        token_pos = i + 1  # +1 for [CLS]
        if tok == ".":
            # All tokens in current_cat_tokens belong to this category
            cat_name = tokenizer.convert_tokens_to_string(current_cat_tokens).strip()
            for idx in range(cat_start_idx + 1, token_pos + 1):  # +1 for [CLS]
                token_to_cat[idx] = cat_name
            current_cat_tokens = []
            cat_start_idx = i + 1
        else:
            current_cat_tokens.append(tok)

    # Handle remaining tokens (if text doesn't end with ".")
    if current_cat_tokens:
        cat_name = tokenizer.convert_tokens_to_string(current_cat_tokens).strip()
        for idx in range(cat_start_idx + 1, len(tokens) + 1):
            token_to_cat[idx] = cat_name

    # Map each detection's token_id to its category
    labels = []
    for tid in token_ids:
        labels.append(token_to_cat.get(int(tid), "object"))

    return labels
