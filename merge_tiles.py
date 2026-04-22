import numpy as np
import math
import json
from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict
import sys
import time

import tiling

sys.path.append('/mnt/data1/syjintw/GS-Interface')

import io_3dgs
from io_3dgs import GaussianModelV2

# [Notes]: 
# Structure of tiles_info: 
# {
# "tile_idxs": tile_idxs,
# "tile_keys": tile_keys,
# "min_corners": min_corners,
# "max_corners": max_corners,
# "index_offsets": index_offsets,
# "flat_indices": flat_indices,
# "grid_shape": grid_shape,
# "scene_min": scene_min,
# "scene_max": scene_max,
# "layer_idx": layer_idx
# }

def load_custom_selected_tiles_json(custom_selection_path):
    with open(custom_selection_path, 'r') as f:
        selected_tiles = json.load(f)
    return selected_tiles

def merge_tiles(tiles_path_list):
    gs_merged = None
    for tile_path in tiles_path_list:
        # Load each tile and merge them
        tile_gs = GaussianModelV2(tile_path)
        if gs_merged is None:
            gs_merged = tile_gs
        else:
            gs_merged.concatenate_gs(tile_gs)
    return gs_merged


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--input_root", default="./tiling_output/longdress_gs_tiled/uniform/frame_0000/lod0", type=str)
    parser.add_argument("--output_root", default="./merged_output/longdress_gs/frame_0000/lod0/point_cloud/iteration_30000/point_cloud.ply", type=str)
    
    # parser.add_argument("--frame_idx", default=0, type=int)
    # parser.add_argument("--start_frame", default=0, type=int)
    # parser.add_argument("--total_frames", default=12, type=int)
    
    parser.add_argument("--gs_type", default="gs", type=str,
                        choices=["gs", "lapisgs", "dlapisgs"], help="Type of Gaussian scene representation")
    
    # # gs_type is "lapisgs" or "dlapisgs"
    # parser.add_argument("--total_layers", default=1, type=int, help="Total number of layers in the Gaussian scene (only for lapisgs)")
    # # gs_type is "dlapisgs"
    # parser.add_argument("--frame_in_group", default=2, type=int, help="Number of frames in each group for dynamic tiling (only for dlapisgs)")
    
    parser.add_argument("--tiling_method", default="uniform", type=str, 
                        choices=["uniform", "uniform_dynamic"])
    
    # # When tiling_method is "uniform" 
    # parser.add_argument("--grid_shape", nargs=3, type=int, default=[4, 4, 4], help="Number of tiles along each axis (x, y, z)")
    
    parser.add_argument("--selected_tiles", type=str, default="all", 
                        choices=["all", "custom", "all_from_input"], help="Merging all tiles or only custom selected tiles, if custom, need to specify the tile indices in a json file")
    parser.add_argument("--custom_selection_path", type=str, default="./tools/custom_tiling_selection/custom.json", help="Path to the JSON file containing custom selected tile keys (only used when --selected_tiles is 'custom')")
    args = parser.parse_args()
    
    start_time = time.time()
    if args.tiling_method == "uniform":
        if args.gs_type == "gs" or args.gs_type == "lapisgs":
            # Only have one layer, so total_layers must be 1
            # Load metadata first
            metadata_path = Path(args.input_root) / "tiles_metadata.npz"
            tiles_info = tiling.load_tiles_from_npz(metadata_path)
            
            if args.selected_tiles == "all":
                selected_tile_keys = [key for key in tiles_info["tile_keys"]]
                tiles_path_list = [Path(args.input_root) / f"tile_{k[0]}_{k[1]}_{k[2]}.ply" for k in selected_tile_keys]
            
            elif args.selected_tiles == "custom":
                selected_tiles = load_custom_selected_tiles_json(args.custom_selection_path)
                selected_tile_keys = selected_tiles["selected_tile_keys"]
                tiles_path_list = [Path(args.input_root) / f"tile_{k[0]}_{k[1]}_{k[2]}.ply" for k in selected_tile_keys]
            
            elif args.selected_tiles == "all_from_input":
                # Get all the .ply files in the input_root folder
                tiles_path_list = list(Path(args.input_root).glob("*.ply"))
                
            merged_gs = merge_tiles(tiles_path_list)
            
            # Export the merged Gaussian model to PLY format
            output_path = Path(args.output_root)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            merged_gs.export_gs_to_ply(output_path)
    
    end_time = time.time()
    print(f"Merging completed in {end_time - start_time:.2f} seconds.")
        
"""
# Example usage:

## GS
python merge_tiles.py \
--input_root ./tiling_output/materials_gs_tiled/uniform/frame_0000/lod0 \
--output_root ./merged_output/materials_gs/all/frame_0000/lod0/point_cloud/iteration_15000/point_cloud.ply \
--gs_type gs \
--tiling_method uniform \
--selected_tiles all

## LapisGS
### LOD0
python merge_tiles.py \
--input_root ./tiling_output/lego_lapisgs_tiled/uniform/frame_0000/lod0 \
--output_root ./merged_output/lego_lapisgs/all/frame_0000/lod0/point_cloud/iteration_30000/point_cloud.ply \
--gs_type lapisgs \
--tiling_method uniform \
--selected_tiles all

### LOD1
python merge_tiles.py \
--input_root ./tiling_output/lego_lapisgs_tiled/uniform/frame_0000/lod1 \
--output_root ./merged_output/lego_lapisgs/all/frame_0000/lod1/point_cloud/iteration_30000/point_cloud.ply \
--gs_type lapisgs \
--tiling_method uniform \
--selected_tiles all

### LOD2
python merge_tiles.py \
--input_root ./tiling_output/lego_lapisgs_tiled/uniform/frame_0000/lod2 \
--output_root ./merged_output/lego_lapisgs/all/frame_0000/lod2/point_cloud/iteration_30000/point_cloud.ply \
--gs_type lapisgs \
--tiling_method uniform \
--selected_tiles all

### LOD3
python merge_tiles.py \
--input_root ./tiling_output/lego_lapisgs_tiled/uniform/frame_0000/lod3 \
--output_root ./merged_output/lego_lapisgs/all/frame_0000/lod3/point_cloud/iteration_30000/point_cloud.ply \
--gs_type lapisgs \
--tiling_method uniform \
--selected_tiles all

# D-LapisGS (Not Yet Implemented)
"""

"""
## LapisGS
### LOD0
python merge_tiles.py \
--input_root ./tiling_output/lego_lapisgs_tiled/uniform/frame_0000/lod0 \
--output_root ./merged_output/lego_lapisgs/custom/frame_0000/lod0/point_cloud/iteration_30000/point_cloud.ply \
--gs_type lapisgs \
--tiling_method uniform \
--selected_tiles custom \
--custom_selection_path ./tools/custom_tiling_selection/custom.json
"""

"""
## LapisGS
### LOD0
python merge_tiles.py \
--input_root ./tiling_output/lego_lapisgs_tiled/uniform/frame_0000/lod0 \
--output_root ./merged_output/lego_lapisgs/all_from_input/frame_0000/lod0/point_cloud/iteration_30000/point_cloud.ply \
--gs_type lapisgs \
--tiling_method uniform \
--selected_tiles all_from_input
"""