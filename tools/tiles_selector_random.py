import json
import time
import random
from pathlib import Path
from argparse import ArgumentParser

import sys
sys.path.append('..')
import tiling

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

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--input_root", default="../tiling_output/lego_lapisgs_tiled/uniform/frame_0000/lod0", type=str)
    parser.add_argument("--output_path", default="./custom_tiling_selection/custom.json", type=str)
    args = parser.parse_args()
    
    # Load tiling metadata
    metadata_path = Path(args.input_root) / "tiles_metadata.npz"    
    tiles_info = tiling.load_tiles_from_npz(metadata_path)
    
    print(tiles_info["tile_idxs"])
    print(tiles_info["tile_keys"])
    
    selection = random.sample(list(tiles_info["tile_idxs"]), len(tiles_info["tile_idxs"]) // 2)  # Randomly select half of the tile keys
    print(selection)

    # Save the selected tile keys to a JSON file
    print(tiles_info["tile_keys"][selection])
    selected_tiles = {
        "selected_tile_idxs": tiles_info["tile_idxs"][selection].tolist(),
        "selected_tile_keys": tiles_info["tile_keys"][selection].tolist()
    }
    print(selected_tiles)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(selected_tiles, f, indent=4)
    
    