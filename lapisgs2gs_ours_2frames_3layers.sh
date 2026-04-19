#!/bin/bash
# This script runs a pipeline of training, rendering, and metrics for each experiment.
# It does not exit on the first error, but continues to the next experiment.
# set -e

export CUDA_VISIBLE_DEVICES=0

SCENE_NAME="longdress"
TOTAL_FRAMES=3
GS_TYPE="lapisgs"
TOTAL_LAYERS=3

for FRAME_IDX in $(seq 0 $((TOTAL_FRAMES - 1))); do
    PADDED_IDX=$(printf "%04d" ${FRAME_IDX})
    echo "Processing frame index: ${PADDED_IDX}"
    for LAYER_IDX in $(seq 0 $((TOTAL_LAYERS - 1))); do
        echo "Processing layer index: ${LAYER_IDX}"
        python lapisgs2gs.py \
            --input_root ./dataset/ours/${SCENE_NAME}_${GS_TYPE}/frame_${PADDED_IDX} \
            --output_root ./dataset/ours2gs/${SCENE_NAME}_${GS_TYPE}/frame_${PADDED_IDX} \
            --target_layer ${LAYER_IDX}
    done
done