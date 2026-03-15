#!/usr/bin/env bash

function usage
{
    echo "usage: ./entrypoint.sh [-b/-r/-d] [model]"
    echo "Choose action from:"
    echo "  -b [model]| Build/download ONNX models (model: yolo, da3, scrfd, sam2; default: all)"
    echo "  -r [model]| Run inference pipeline (model: yolo, da3, scrfd, sam2, yolo+sam2; default: yolo)"
    echo "  -d        | Develop using bash terminal"
    echo "  -h        | Help"
}

ACTION=""
MODEL=""

if [[ $# -lt 1 ]]; then
    usage && exit;
fi

while [[ "$1" != "" ]]; do
    case $1 in
        -b | -r | -d )  ACTION=$1   ;;
        -h )          usage && exit ;;
        * )
            if [[ ($ACTION == '-r' || $ACTION == '-b') && -z $MODEL ]]; then
                MODEL=$1
            else
                usage && exit
            fi
            ;;
    esac
    shift;
done

if [[ $ACTION == '-b' ]]; then
    cd /opt/scripts
    if [[ -z $MODEL ]]; then
        # Build all models
        bash export_onnx.sh
        bash download_da3_onnx.sh
        bash download_scrfd_onnx.sh
        bash export_sam2_onnx.sh
    else
        case $MODEL in
            yolo)
                bash export_onnx.sh
                ;;
            da3)
                bash download_da3_onnx.sh
                ;;
            scrfd)
                bash download_scrfd_onnx.sh
                ;;
            sam2)
                bash export_sam2_onnx.sh
                ;;
            *)
                echo "Unknown model: $MODEL (supported: yolo, da3, scrfd, sam2)"
                exit 1
                ;;
        esac
    fi
elif [[ $ACTION == '-r' ]]; then
    # Default to yolo if no model specified
    MODEL=${MODEL:-yolo}
    cd /app
    case $MODEL in
        yolo)
            python3 -m yolo.main --images_dir /app/assets/images --model yolo/onnx/yolo11n.onnx --gpu
            ;;
        da3)
            # python3 /app/da3/depth_inference.py --image /app/assets/images/image1.jpg --model /app/da3/onnx/DA3METRIC-LARGE.onnx --pointcloud --color-cloud
            python3 /app/da3/depth_inference.py --image /app/assets/images/image1.jpg --model /app/da3/onnx/DA3METRIC-LARGE.onnx
            ;;
        scrfd)
            python3 -m scrfd.main --image /app/assets/images/image1.jpg --model scrfd/onnx/det_10g.onnx --gpu
            ;;
        sam2)
            python3 -m sam2.main --image /app/assets/images/image1.jpg --model-dir sam2/onnx/small --gpu --box 100,100,400,400
            ;;
        yolo+sam2)
            python3 sam2/pipeline_yolo.py --image /app/assets/images/image1.jpg --yolo-model yolo/onnx/yolo11n.onnx --sam2-model-dir sam2/onnx/small --gpu
            ;;
        *)
            echo "Unknown model: $MODEL (supported: yolo, da3, scrfd, sam2, yolo+sam2)"
            exit 1
            ;;
    esac
elif [[ $ACTION == '-d' ]]; then
    /bin/bash
else
    usage && exit;
fi
