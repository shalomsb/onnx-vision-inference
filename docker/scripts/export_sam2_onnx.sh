#!/usr/bin/env bash
# Export SAM2.1 ONNX models (image_encoder + mask_decoder) for all sizes.
# Clones sam2-onnx-tensorrt, installs deps, exports, copies to /app/sam2/onnx/<size>/

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" 2>/dev/null && pwd || pwd)"
REPO_DIR="/opt/deepstream_tools"
EXPORT_DIR="${REPO_DIR}/sam2-onnx-tensorrt"
REPO_URL="https://github.com/NVIDIA-AI-IOT/deepstream_tools.git"
SAM2_SIZES=("tiny" "small" "base_plus" "large")

# Check if all sizes are already exported
ALL_EXIST=true
for SIZE in "${SAM2_SIZES[@]}"; do
    DEST="/app/sam2/onnx/${SIZE}"
    if [[ ! -f "$DEST/image_encoder.onnx" || ! -f "$DEST/mask_decoder.onnx" ]]; then
        ALL_EXIST=false
        break
    fi
done

if [[ "$ALL_EXIST" == "true" ]]; then
    echo "All SAM2 ONNX models already exist — skipping export."
    exit 0
fi

# ── Clone repo ────────────────────────────────────────────────────────
if [[ ! -d "$EXPORT_DIR" ]]; then
    echo "Cloning deepstream_tools (contains sam2-onnx-tensorrt)..."
    git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$EXPORT_DIR"

# ── Install dependencies ──────────────────────────────────────────────
echo "Installing PyTorch and SAM2 dependencies..."
pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip3 install -e .

# ── Download checkpoints ─────────────────────────────────────────────
echo "Downloading SAM2.1 checkpoints..."
cd checkpoints
bash download_ckpts.sh
cd ..

# ── Patch mask_decoder.py: repeat_interleave -> tile ─────────────────
# This fixes ONNX export compatibility (repeat_interleave not well supported)
MASK_DECODER_PY="sam2/modeling/sam2_utils.py"
if grep -q "repeat_interleave" "$MASK_DECODER_PY" 2>/dev/null; then
    echo "Patching repeat_interleave -> tile in $MASK_DECODER_PY..."
    sed -i 's/\.repeat_interleave(\([^,]*\), dim=\([0-9]*\))/.repeat(\*[1]*\2 + [\1] + [1]*(len(.shape)-\2-1))/g' "$MASK_DECODER_PY" || true
fi

# Also patch Module.py if needed
MODULE_PY="src/Module.py"
if grep -q "repeat_interleave" "$MODULE_PY" 2>/dev/null; then
    echo "Patching repeat_interleave -> tile in $MODULE_PY..."
    sed -i 's/obj_pos.repeat_interleave(4, dim=0)/obj_pos.repeat(4, 1, 1)/g' "$MODULE_PY"
fi

# ── Export all sizes ──────────────────────────────────────────────────
for SIZE in "${SAM2_SIZES[@]}"; do
    DEST="/app/sam2/onnx/${SIZE}"
    if [[ -f "$DEST/image_encoder.onnx" && -f "$DEST/mask_decoder.onnx" ]]; then
        echo "SAM2 ${SIZE} already exported — skipping."
        continue
    fi

    echo ""
    echo "============================================"
    echo "Exporting SAM2 ${SIZE}..."
    echo "============================================"

    mkdir -p "checkpoints/${SIZE}"
    python3 export_sam2_onnx.py --model "$SIZE"

    # Copy only image_encoder and mask_decoder (we don't need memory modules for image-only mode)
    mkdir -p "$DEST"
    cp "checkpoints/${SIZE}/image_encoder.onnx" "$DEST/"
    cp "checkpoints/${SIZE}/mask_decoder.onnx" "$DEST/"
    echo "Copied to $DEST"
done

# ── Restore onnxruntime-gpu (torch may have overwritten it) ──────────
echo "Restoring onnxruntime-gpu..."
pip3 install --no-cache-dir onnxruntime-gpu

echo ""
echo "SAM2 ONNX export complete. Models at /app/sam2/onnx/"
ls -la /app/sam2/onnx/*/
