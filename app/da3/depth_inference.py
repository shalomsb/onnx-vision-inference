"""
DA3METRIC-LARGE - Standalone inference with ONNXRuntime
One script, one image, see the result.

Usage:
    python depth_inference.py --image photo.jpg --model DA3METRIC-LARGE.onnx
    python depth_inference.py --image photo.jpg --model DA3METRIC-LARGE.onnx --pointcloud --color-cloud
"""

import argparse

import cv2
import numpy as np
import onnxruntime as ort

# ── Camera intrinsics (Logitech C920 defaults) ────────────────────────────────
FX, FY = 578.57, 578.76
CX, CY = 305.79, 181.83

# ── Model expects dimensions divisible by 14 (ViT patch size) ─────────────────
MODEL_H, MODEL_W = 280, 504   # HuggingFace ONNX model resolution

# ── ImageNet normalization ─────────────────────────────────────────────────────
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(bgr_image, model_h, model_w):
    """
    BGR uint8 image → [1, 3, H, W] float32 tensor

    Steps:
    1. Resize to model resolution (bicubic)
    2. BGR → RGB
    3. Normalize with ImageNet mean/std
    4. HWC → CHW → add batch dim
    """
    resized = cv2.resize(bgr_image, (model_w, model_h), interpolation=cv2.INTER_CUBIC)
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    norm    = (rgb.astype(np.float32) / 255.0 - MEAN) / STD  # [H, W, 3]
    chw     = norm.transpose(2, 0, 1)                         # [3, H, W]
    return chw[np.newaxis]                                     # [1, 3, H, W]


def postprocess(depth_raw, sky_raw, orig_h, orig_w, model_h, model_w):
    """
    Raw model outputs → metric depth image [orig_h, orig_w] float32

    depth_raw : [1, 1, model_h, model_w] float32
    sky_raw   : [1, 1, model_h, model_w] float32
    """
    depth = depth_raw.squeeze()   # [model_h, model_w]
    sky   = sky_raw.squeeze()     # [model_h, model_w]

    depth = np.clip(depth, 0.0, None)

    # ── Step 1: Metric scaling ─────────────────────────────────────────────────
    # Scale intrinsics to model resolution, then apply DA3 formula
    fx_s = FX * (model_w / orig_w)
    fy_s = FY * (model_h / orig_h)
    focal = 0.5 * (fx_s + fy_s)
    depth = depth * (focal / 300.0)

    # ── Step 2: Sky masking ────────────────────────────────────────────────────
    non_sky = sky < 0.3
    valid_depths = depth[non_sky & (depth > 0)]
    if valid_depths.size > 0:
        far = np.percentile(valid_depths, 99)
        far = min(far, 200.0)
        depth[~non_sky] = far

    # ── Step 3: Resize to original resolution ─────────────────────────────────
    depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    return depth  # [orig_h, orig_w] float32, meters


def colorize(depth):
    """
    Metric depth [H, W] float32 → BGR uint8 with plasma colormap

    Uses percentile normalization to ignore outliers.
    """
    valid = (depth > 0) & np.isfinite(depth)
    if valid.sum() < 100:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    p_low  = np.percentile(depth[valid], 1)
    p_high = np.percentile(depth[valid], 99)

    norm = np.clip((depth - p_low) / (p_high - p_low + 1e-6), 0.0, 1.0)
    gray = (norm * 255).astype(np.uint8)

    return cv2.applyColorMap(gray, cv2.COLORMAP_PLASMA)


def build_pointcloud(depth_metric, orig_h, orig_w, bgr=None):
    """
    Metric depth [H, W] float32 → Open3D PointCloud object

    bgr: if provided, colors each point with the original image color
    """
    import open3d as o3d

    # ── Build pixel grid ───────────────────────────────────────────────────────
    vs, us = np.meshgrid(
        np.arange(orig_h, dtype=np.float32),
        np.arange(orig_w, dtype=np.float32),
        indexing='ij'
    )  # both [H, W]

    d = depth_metric  # [H, W]

    # ── Filter invalid pixels ──────────────────────────────────────────────────
    valid = (d > 0.0) & np.isfinite(d)

    d_v  = d[valid]
    us_v = us[valid]
    vs_v = vs[valid]

    # ── Backproject: pixel → 3D ────────────────────────────────────────────────
    X = (us_v - CX) * d_v / FX
    Y = (vs_v - CY) * d_v / FY
    Z = d_v

    points = np.stack([X, Y, Z], axis=-1)  # [N, 3]
    print(f"Point cloud: {len(points):,} points")

    # ── Color ──────────────────────────────────────────────────────────────────
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if bgr is not None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        colors = rgb[valid]  # [N, 3]
    else:
        z_norm = (Z - Z.min()) / (Z.max() - Z.min() + 1e-6)
        colors = np.stack([1 - z_norm, z_norm * 0.5, z_norm], axis=-1)

    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def save_pointcloud(pcd, path):
    """Save Open3D PointCloud to a .ply file."""
    import open3d as o3d
    o3d.io.write_point_cloud(path, pcd)
    print(f"Saved point cloud: {path}")


def render_pointcloud(pcd, path, orig_w, orig_h):
    """
    Render a stereo pair (left/right eye) side-by-side, like a 3D movie frame.
    The camera shifts sideways to create parallax — near objects shift more than far ones.
    """
    import open3d as o3d

    EYE_SEPARATION = 0.65  # ~65mm, average human interpupillary distance (meters)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(orig_w, orig_h, FX, FY, CX, CY)

    left_img = _render_eye(pcd, intrinsic, orig_w, orig_h, -EYE_SEPARATION / 2)
    right_img = _render_eye(pcd, intrinsic, orig_w, orig_h, +EYE_SEPARATION / 2)

    stereo = np.hstack([left_img, right_img])
    cv2.imwrite(path, stereo)
    print(f"Saved stereo render: {path}  ({stereo.shape[1]}x{stereo.shape[0]})")


def _render_eye(pcd, intrinsic, width, height, x_offset):
    """Render from a camera shifted horizontally by x_offset meters."""
    import open3d as o3d

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background(np.array([0.0, 0.0, 0.0, 1.0]))

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.point_size = 2.0
    renderer.scene.add_geometry("cloud", pcd, mat)

    # Extrinsic: shift camera along X axis
    extrinsic = np.eye(4)
    extrinsic[0, 3] = x_offset  # translate camera left/right
    renderer.setup_camera(intrinsic, extrinsic)

    img = np.asarray(renderer.render_to_image())
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def show_pointcloud(pcd):
    """Show Open3D PointCloud in a GUI window."""
    import open3d as o3d
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="DA3METRIC — Point Cloud",
        width=1280, height=720
    )


def main():
    parser = argparse.ArgumentParser(description="DA3METRIC-LARGE depth inference")
    parser.add_argument("--image", required=True,  help="Input image path")
    parser.add_argument("--model", required=True,  help="DA3METRIC-LARGE.onnx path")
    parser.add_argument("--save",  default="depth_output.png", help="Output path")
    parser.add_argument("--show",  action="store_true", help="Show result window")
    parser.add_argument("--pointcloud", action="store_true", help="Generate 3D point cloud")
    parser.add_argument("--save-ply",   default="pointcloud.ply", help="Point cloud output path")
    parser.add_argument("--save-render", default="pointcloud_render.png", help="Render point cloud to 2D image")
    parser.add_argument("--show-cloud", action="store_true", help="Show 3D point cloud in GUI window")
    parser.add_argument("--color-cloud", action="store_true", help="Color cloud with image RGB")
    args = parser.parse_args()

    # ── Load image ─────────────────────────────────────────────────────────────
    bgr = cv2.imread(args.image)
    if bgr is None:
        raise FileNotFoundError(f"Cannot open image: {args.image}")
    orig_h, orig_w = bgr.shape[:2]
    print(f"Image: {orig_w}x{orig_h}")

    # ── Load ONNX model ────────────────────────────────────────────────────────
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3  # suppress warnings (0=verbose, 1=info, 2=warn, 3=error)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session   = ort.InferenceSession(args.model, sess_options=sess_opts, providers=providers)

    # Print what the model expects / outputs
    for inp in session.get_inputs():
        print(f"Input  : {inp.name}  {inp.shape}  {inp.type}")
    for out in session.get_outputs():
        print(f"Output : {out.name}  {out.shape}  {out.type}")

    # Grab actual model resolution from ONNX input shape
    inp_shape = session.get_inputs()[0].shape
    model_h   = inp_shape[2] if isinstance(inp_shape[2], int) else MODEL_H
    model_w   = inp_shape[3] if isinstance(inp_shape[3], int) else MODEL_W
    print(f"Model resolution: {model_w}x{model_h}")

    # ── Preprocess ─────────────────────────────────────────────────────────────
    tensor = preprocess(bgr, model_h, model_w)
    print(f"Input tensor: {tensor.shape}  min={tensor.min():.2f}  max={tensor.max():.2f}")

    # ── Inference ──────────────────────────────────────────────────────────────
    input_name = session.get_inputs()[0].name
    outputs    = session.run(None, {input_name: tensor})

    depth_raw = outputs[0]   # [1, 1, H, W]
    sky_raw   = outputs[1]   # [1, 1, H, W]
    print(f"Raw depth: min={depth_raw.min():.4f}  max={depth_raw.max():.4f}")
    print(f"Raw sky  : min={sky_raw.min():.4f}   max={sky_raw.max():.4f}")

    # ── Postprocess ────────────────────────────────────────────────────────────
    depth_metric = postprocess(depth_raw, sky_raw, orig_h, orig_w, model_h, model_w)
    print(f"Metric depth: min={depth_metric.min():.2f}m  max={depth_metric.max():.2f}m")

    # ── Visualize ──────────────────────────────────────────────────────────────
    depth_color = colorize(depth_metric)

    # Side-by-side: original | depth
    side_by_side = np.hstack([bgr, depth_color])
    cv2.imwrite(args.save, side_by_side)
    print(f"Saved: {args.save}")

    if args.show:
        cv2.imshow("Original | Metric Depth (plasma)", side_by_side)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ── 3D point cloud ─────────────────────────────────────────────────────────
    if args.pointcloud:
        rgb_for_cloud = bgr if args.color_cloud else None
        pcd = build_pointcloud(depth_metric, orig_h, orig_w, rgb_for_cloud)
        save_pointcloud(pcd, args.save_ply)
        render_pointcloud(pcd, args.save_render, orig_w, orig_h)
        if args.show_cloud:
            show_pointcloud(pcd)


if __name__ == "__main__":
    main()
