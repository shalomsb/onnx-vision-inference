#!/usr/bin/env bash

function usage
{
    echo "usage: ./entrypoint.sh [-b/-r/-d]"
    echo "Choose action from:"
    echo "  -b | Build ONNX models (export)"
    echo "  -r | Run inference pipeline"
    echo "  -d | Develop using bash terminal"
    echo "  -h | Help"
}

ACTION=""

if [[ $# -ne 1 ]]; then
    usage && exit;
fi

while [[ "$1" != "" ]]; do
    case $1 in
        -b | -r | -d )  ACTION=$1   ;;
        -h )          usage && exit ;;
        * )           usage && exit ;;
    esac
    shift;
done

if [[ $ACTION == '-b' ]]; then
    cd /opt/scripts
    bash export_onnx.sh
elif [[ $ACTION == '-r' ]]; then
    cd /app
    python3 -m yolo.main --images_dir /app/assets/images --model yolo/onnx/yolo11n.onnx --gpu
elif [[ $ACTION == '-d' ]]; then
    /bin/bash
else
    usage && exit;
fi
