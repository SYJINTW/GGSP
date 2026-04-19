from argparse import ArgumentParser
from pathlib import Path

import sys
sys.path.append('/mnt/data1/syjintw/GS-Interface')
import io_3dgs
from io_3dgs import GaussianModelV2


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--input_root", default="./merged_output/longdress_lapisgs/frame_0000", type=str)
    parser.add_argument("--output_root", default="./lapisgs2gs_output/longdress_lapisgs/frame_0000", type=str)
    parser.add_argument("--target_layer", default=3, type=int)
    args = parser.parse_args()
    
    # Need to merge from the lowest layer to the target layer
    gs_merged = None
    for layer_idx in range(args.target_layer + 1):
        layer_path = Path(args.input_root) / f"lod{layer_idx}" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
        tile_gs = GaussianModelV2(layer_path)
        if gs_merged is None:
            gs_merged = tile_gs
        else:
            gs_merged.concatenate_gs(tile_gs)
    
    # Need to handle the opacity
    # Generate the correct "opacity" attribute according to the target layer
    gs_merged.add_attribute(f"opacity", "f4", gs_merged.data[f"opacity_lod{args.target_layer}"]["data"])
    
    # Delete the redundant "opacity_lod0", "opacity_lod1", etc. attributes if they exist
    # Get the keys that is "opacity_lod*"
    lod_keys_to_delete = [
        key for key in gs_merged.data.keys() 
        if key.startswith("opacity_lod")
    ]
    # Delete the keys
    for key in lod_keys_to_delete:
        gs_merged.delete_attribute(key)
    
    output_path = Path(args.output_root) / f"lod{args.target_layer}" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gs_merged.export_gs_to_ply(output_path)
    
    
"""
# Example usage:

python lapisgs2gs.py \
--input_root ./dataset/ours/longdress_lapisgs/frame_0000 \
--output_root ./dataset/ours2gs/longdress_lapisgs/frame_0000 \
--target_layer 3

python lapisgs2gs.py \
--input_root ./merged_output/longdress_lapisgs/frame_0000 \
--output_root ./lapisgs2gs_output/longdress_lapisgs/frame_0000 \
--target_layer 3

python lapisgs2gs.py \
--input_root ./merged_output/longdress_lapisgs/frame_0001 \
--output_root ./lapisgs2gs_output/longdress_lapisgs/frame_0001 \
--target_layer 3
"""

    