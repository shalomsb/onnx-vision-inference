#!/usr/bin/env bash
# Download Grounding DINO ONNX model from HuggingFace (onnx-community)

ONNX_DIR="/app/groundingdino/onnx"
ONNX_FILE="$ONNX_DIR/model.onnx"
URL="https://huggingface.co/onnx-community/grounding-dino-tiny-ONNX/resolve/main/onnx/model.onnx"

if [[ -f "$ONNX_FILE" ]]; then
    echo "Grounding DINO ONNX model already exists: $ONNX_FILE — skipping."
    exit 0
fi

mkdir -p "$ONNX_DIR"

echo "Downloading Grounding DINO ONNX model (~719 MB) ..."
wget -q --show-progress -O "$ONNX_FILE" "$URL"

if [[ $? -eq 0 && -f "$ONNX_FILE" ]]; then
    echo "Download complete: $ONNX_FILE"
else
    echo "ERROR: Failed to download Grounding DINO ONNX model."
    rm -f "$ONNX_FILE"
    exit 1
fi
