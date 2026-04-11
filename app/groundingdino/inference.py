import numpy as np
import onnxruntime as ort


def load_model(
    model_path: str,
    providers: list[str] = None,
) -> ort.InferenceSession:
    """
    Load the Grounding DINO ONNX model.

    Args:
        model_path: path to .onnx file
        providers:  e.g. ["CUDAExecutionProvider", "CPUExecutionProvider"]

    Returns:
        ort.InferenceSession
    """
    if providers is None:
        providers = ["CPUExecutionProvider"]

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    return ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)


def run_inference(
    session: ort.InferenceSession,
    image_blob: np.ndarray,
    pixel_mask: np.ndarray,
    text_inputs: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run Grounding DINO inference.

    Args:
        session:     ONNXRuntime InferenceSession
        image_blob:  [1, 3, 800, 800] float32
        pixel_mask:  [1, 800, 800] int64
        text_inputs: dict with input_ids, token_type_ids, attention_mask

    Returns:
        logits:     [1, 900, 256] float32 (raw logits per token)
        pred_boxes: [1, 900, 4] float32 (cx, cy, w, h normalized)
    """
    # Build feed dict from all inputs
    all_inputs = {
        "pixel_values": image_blob,
        "pixel_mask": pixel_mask,
        **text_inputs,
    }

    # Match to actual ONNX input names
    input_names = [inp.name for inp in session.get_inputs()]
    feed = {name: all_inputs[name] for name in input_names}

    output_names = [o.name for o in session.get_outputs()]
    outputs = session.run(output_names, feed)

    out = dict(zip(output_names, outputs))
    logits_key = next(k for k in out if "logit" in k.lower())
    boxes_key = next(k for k in out if "box" in k.lower())

    return out[logits_key], out[boxes_key]
