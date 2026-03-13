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
    batch: np.ndarray,
) -> np.ndarray:
    """
    Run inference on a batch.

    Args:
        session: ONNXRuntime InferenceSession
        batch:   [N, 3, H, W] float32

    Returns:
        output:  raw model output [N, num_classes + 4, num_detections]
    """
    input_name = session.get_inputs()[0].name
    # None -> return all outputs.
    # {input_name: batch} -> input feed dict, we can have multiple inputs for some models.
    output = session.run(None, {input_name: batch})
    # we have only one output, so we take the first element of the output list.
    return output[0]
