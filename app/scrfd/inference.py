import numpy as np
import onnxruntime as ort


def load_model(
    model_path: str,
    providers: list[str] = None,
) -> ort.InferenceSession:
    """
    Load an ONNX model into an ONNXRuntime InferenceSession.

    Args:
        model_path: path to .onnx file
        providers:  e.g. ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    defaults to CPU only
    """
    if providers is None:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(model_path, providers=providers)


def run_inference(
    session: ort.InferenceSession,
    blob: np.ndarray,
) -> list[np.ndarray]:
    """
    Run inference on a single image blob.

    Args:
        session: ONNXRuntime InferenceSession
        blob:    [1, 3, H, W] float32

    Returns:
        outputs: list of 9 tensors (3 score maps, 3 bbox maps, 3 keypoint maps)
    """
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: blob})
