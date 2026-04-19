#!/bin/bash
# This script runs a pipeline of training, rendering, and metrics for each experiment.
# It does not exit on the first error, but continues to the next experiment.
# set -e

export CUDA_VISIBLE_DEVICES=0

SCENE_NAME="longdress"
TOTAL_FRAMES=12
TILING_METHOD="uniform"
GS_TYPE="gs"

for FRAME_IDX in $(seq 0 $((TOTAL_FRAMES - 1))); do
    PADDED_IDX=$(printf "%04d" ${FRAME_IDX})
    echo "Processing frame index: ${PADDED_IDX}"
    python merge_tiles.py \
        --input_root ./tiling_output/${SCENE_NAME}_${GS_TYPE}_tiled/${TILING_METHOD}/frame_${PADDED_IDX}/lod0 \
        --output_root ./merged_output/${SCENE_NAME}_${GS_TYPE}/frame_${PADDED_IDX}/lod0/point_cloud/iteration_30000/point_cloud.ply \
        --gs_type ${GS_TYPE} \
        --tiling_method ${TILING_METHOD} \
        --selected_tiles all
done