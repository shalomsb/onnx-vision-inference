import numpy as np
import onnxruntime as ort


def load_models(
    model_dir: str,
    providers: list[str] = None,
) -> dict[str, ort.InferenceSession]:
    """
    Load SAM2 image encoder and mask decoder ONNX models.

    Args:
        model_dir: directory containing image_encoder.onnx and mask_decoder.onnx
        providers: e.g. ["CUDAExecutionProvider", "CPUExecutionProvider"]

    Returns:
        dict with "encoder" and "decoder" InferenceSession objects
    """
    if providers is None:
        providers = ["CPUExecutionProvider"]

    encoder_path = f"{model_dir}/image_encoder.onnx"
    decoder_path = f"{model_dir}/mask_decoder.onnx"

    print(f"Loading encoder: {encoder_path}")
    encoder = ort.InferenceSession(encoder_path, providers=providers)
    print(f"Loading decoder: {decoder_path}")
    decoder = ort.InferenceSession(decoder_path, providers=providers)

    return {"encoder": encoder, "decoder": decoder}


def run_encoder(
    session: ort.InferenceSession,
    blob: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Run SAM2 image encoder.

    Args:
        session: encoder InferenceSession
        blob:    [1, 3, 1024, 1024] float32

    Returns:
        dict with feature tensors:
            pix_feat:          [1, 256, 64, 64]
            high_res_feat0:    [1, 32, 256, 256]
            high_res_feat1:    [1, 64, 128, 128]
            vision_feats:      encoder backbone features
            vision_pos_embed:  positional embeddings
    """
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    outputs = session.run(output_names, {input_name: blob})

    return {name: out for name, out in zip(output_names, outputs)}


def run_decoder(
    session: ort.InferenceSession,
    point_coords: np.ndarray,
    point_labels: np.ndarray,
    image_embed: np.ndarray,
    high_res_feat0: np.ndarray,
    high_res_feat1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run SAM2 mask decoder.

    Args:
        session:         decoder InferenceSession
        point_coords:    [N, num_pts, 2] float32
        point_labels:    [N, num_pts] float32
        image_embed:     [N, 256, 64, 64] float32
        high_res_feat0:  [1, 32, 256, 256] float32
        high_res_feat1:  [1, 64, 128, 128] float32

    Returns:
        pred_mask: [N, 1, H, W] float32 (logits or sigmoid depending on model)
        iou:       [N, 1] float32
    """
    # Build input dict from model's expected input names
    input_names = [inp.name for inp in session.get_inputs()]
    inputs = {
        "point_coords": point_coords,
        "point_labels": point_labels,
        "image_embed": image_embed,
        "high_res_feats_0": high_res_feat0,
        "high_res_feats_1": high_res_feat1,
    }
    feed = {name: inputs[name] for name in input_names}

    output_names = [o.name for o in session.get_outputs()]
    outputs = session.run(output_names, feed)

    # Find pred_mask and iou in outputs
    output_dict = {name: out for name, out in zip(output_names, outputs)}
    pred_mask = output_dict["pred_mask"]
    iou = output_dict["iou"]

    return pred_mask, iou
