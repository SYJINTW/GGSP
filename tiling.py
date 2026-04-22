import numpy as np
import math
import json
from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict
import sys
import time

# sys.path.append('/mnt/data1/syjintw/GS-Interface')
sys.path.append('/home/syjintw/Desktop/NUS/GS-Interface')

import io_3dgs
from io_3dgs import GaussianModelV2


def tiling_uniform_layered_gs(gs_layers, grid_shape=(2, 2, 2)):
    """
    根據指定的 grid_shape (x, y, z) 將多層 Gaussian 場景均勻切割。
    gs_layers: 0, 1, 2, ... N-1 層的 Gaussian 場景列表 (每層為一個 GaussianModelV2 物件)。
    
    Inputs:
        gs_layers (list): 包含多層 GaussianModelV2 物件的列表。
        grid_shape (tuple): 指定切割成多少個 Tile (n_tiles_x, n_tiles_y, n_tiles_z)。
        
    Returns:
        tile_aabbs (dict): Key 為 tile 的 3D 索引 (ix, iy, iz)，Value 為 (min_corner, max_corner)。
        tile_indices (dict): Key 為 tile 的 3D 索引，Value 為 list of numpy arrays (每個 layer 中屬於該 tile 的 indices)。
    """
    
    l_min_list = []
    l_max_list = []
    
    # --- 1. 計算所有 Layer 的 Global AABB ---
    for layer in gs_layers:
        l_min, l_max = layer.get_AABB()
        l_min_list.append(l_min)
        l_max_list.append(l_max)
    
    scene_min = np.min(np.array(l_min_list), axis=0)
    scene_max = np.max(np.array(l_max_list), axis=0)
    scene_extent = scene_max - scene_min

    # --- 2. 根據 Grid Shape 決定每個 Tile 的尺寸 (Step Size) ---
    n_tiles_x, n_tiles_y, n_tiles_z = grid_shape
    step_size = scene_extent / np.array(grid_shape)
    
    print(f"Grid Shape: {n_tiles_x}x{n_tiles_y}x{n_tiles_z} | Step Size (x,y,z): {step_size}")
    
    # 準備儲存結果的字典
    tile_aabbs = {}
    tile_indices = {}
    
    # --- 3. 遍歷 Grid，擷取每個 Tile 的 Indices 與 AABB ---
    for ix in range(n_tiles_x):
        for iy in range(n_tiles_y):
            for iz in range(n_tiles_z):
                tile_idx = (ix, iy, iz)
                
                # 計算當前 Tile 的幾何邊界
                x0 = scene_min[0] + ix * step_size[0]
                y0 = scene_min[1] + iy * step_size[1]
                z0 = scene_min[2] + iz * step_size[2]
                
                # 為了避免浮點數誤差，最後一個 Grid 直接對齊 scene_max
                x1 = scene_min[0] + (ix + 1) * step_size[0] if ix < n_tiles_x - 1 else scene_max[0]
                y1 = scene_min[1] + (iy + 1) * step_size[1] if iy < n_tiles_y - 1 else scene_max[1]
                z1 = scene_min[2] + (iz + 1) * step_size[2] if iz < n_tiles_z - 1 else scene_max[2]
                
                min_corner = np.array([x0, y0, z0])
                max_corner = np.array([x1, y1, z1])
                
                layer_indices_list = []
                
                # 找出每個 layer 中，落在這個 Tile 範圍內的 Gaussians
                for layer in gs_layers:
                    x_data = layer.data["x"]["data"]
                    y_data = layer.data["y"]["data"]
                    z_data = layer.data["z"]["data"]
                    
                    # 使用條件過濾 (注意：為了包含邊界上的點，最後一個 grid 會用 <=)
                    cond_x = (x_data >= x0) & (x_data <= x1 if ix == n_tiles_x - 1 else x_data < x1)
                    cond_y = (y_data >= y0) & (y_data <= y1 if iy == n_tiles_y - 1 else y_data < y1)
                    cond_z = (z_data >= z0) & (z_data <= z1 if iz == n_tiles_z - 1 else z_data < z1)
                    
                    indices = np.where(cond_x & cond_y & cond_z)[0]
                    layer_indices_list.append(indices)
                
                # 如果這個 Tile 在所有 Frame/Layer 中都是空的，就直接跳過不儲存
                if all(len(indices) == 0 for indices in layer_indices_list):
                    continue
                
                # 儲存非空 Tile 的 AABB 和 對應的 Indices
                tile_aabbs[tile_idx] = {"min_corner": min_corner, "max_corner": max_corner}
                tile_indices[tile_idx] = layer_indices_list
                
    return tile_aabbs, tile_indices, scene_min, scene_max


def save_tiles_to_npz(tile_aabbs, tile_indices, output_path, 
                    grid_shape=None, 
                    scene_min=None, scene_max=None,
                    layer_idx=0
                    ):
    """
    將 tile AABB 與對應的 GS indices 存成 npz 檔案，格式適合後續 pipeline 使用。

    Parameters
    ----------
    tile_aabbs : dict
        格式例如:
        {
            (ix, iy, iz): {
                "min_corner": np.array([x0, y0, z0]),
                "max_corner": np.array([x1, y1, z1])
            },
            ...
        }

    tile_indices : dict
        格式例如:
        {
            (ix, iy, iz): [indices_layer0, indices_layer1, ...]
        }

        若目前只用 frame 0 做 tile membership，則通常每個 value 只有一個 array，
        即 tile_indices[tile_idx][0] 是該 tile 對應的 GS indices。

    output_path : str or Path
        輸出的 .npz 路徑

    grid_shape : tuple or list, optional
        例如 (4, 4, 4)

    scene_min : array-like, optional
        場景總 AABB 的 min corner

    scene_max : array-like, optional
        場景總 AABB 的 max corner

    layer_idx : int, optional
        要保存的 layer 索引
    """

    # 固定順序，避免每次 dict iteration 順序不一致
    sorted_tile_keys = sorted(tile_aabbs.keys())

    tile_idxs_list = []
    tile_keys_list = []
    min_corners_list = []
    max_corners_list = []
    flat_indices_list = []
    index_offsets = [0]

    for i, tile_idx in enumerate(sorted_tile_keys):
        aabb = tile_aabbs[tile_idx]

        gs_idx = tile_indices[tile_idx][layer_idx]

        tile_idxs_list.append(i)
        tile_keys_list.append(tile_idx)
        min_corners_list.append(aabb["min_corner"])
        max_corners_list.append(aabb["max_corner"])

        flat_indices_list.append(gs_idx.astype(np.int64))
        index_offsets.append(index_offsets[-1] + len(gs_idx))

    # 轉成 numpy array
    tile_idxs_arr = np.asarray(tile_idxs_list, dtype=np.int32)          # (N, 3)
    tile_keys_arr = np.asarray(tile_keys_list, dtype=np.int32)          # (N, 3)
    min_corners_arr = np.asarray(min_corners_list, dtype=np.float32)    # (N, 3)
    max_corners_arr = np.asarray(max_corners_list, dtype=np.float32)    # (N, 3)
    index_offsets_arr = np.asarray(index_offsets, dtype=np.int64)        # (N+1,)

    if len(flat_indices_list) > 0:
        flat_indices_arr = np.concatenate(flat_indices_list).astype(np.int64)
    else:
        flat_indices_arr = np.empty((0,), dtype=np.int64)

    save_dict = {
        "tile_idxs": tile_idxs_arr,
        "tile_keys": tile_keys_arr,
        "min_corners": min_corners_arr,
        "max_corners": max_corners_arr,
        "index_offsets": index_offsets_arr,
        "flat_indices": flat_indices_arr,
    }

    if grid_shape is not None:
        save_dict["grid_shape"] = np.asarray(grid_shape, dtype=np.int32)

    if scene_min is not None:
        save_dict["scene_min"] = np.asarray(scene_min, dtype=np.float32)

    if scene_max is not None:
        save_dict["scene_max"] = np.asarray(scene_max, dtype=np.float32)

    np.savez(output_path, **save_dict)

    print(f"Saved tile metadata to: {output_path}")
    print(f"Number of non-empty tiles: {len(tile_keys_arr)}")
    print(f"Total number of stored GS indices: {len(flat_indices_arr)}")


def load_tiles_from_npz(npz_path):
    data = np.load(npz_path)

    tile_idxs = data["tile_idxs"]
    tile_keys = data["tile_keys"]
    min_corners = data["min_corners"]
    max_corners = data["max_corners"]
    index_offsets = data["index_offsets"]
    flat_indices = data["flat_indices"]

    grid_shape = data["grid_shape"] if "grid_shape" in data else None
    scene_min = data["scene_min"] if "scene_min" in data else None
    scene_max = data["scene_max"] if "scene_max" in data else None
    layer_idx = data["layer_idx"] if "layer_idx" in data else None
    
    return {
        "tile_idxs": tile_idxs,
        "tile_keys": tile_keys,
        "min_corners": min_corners,
        "max_corners": max_corners,
        "index_offsets": index_offsets,
        "flat_indices": flat_indices,
        "grid_shape": grid_shape,
        "scene_min": scene_min,
        "scene_max": scene_max,
        "layer_idx": layer_idx
    }
    

def export_tiles_to_ply(gs, tile_indices, layer_idx, output_dir):
    for tile_idx, gs_indices in tile_indices.items():
        # print(f"Exporting tile {tile_idx} with {len(gs_indices[layer_idx])} Gaussians...")
        tile_gs = gs.extract_gaussians(gs_indices[layer_idx])
        save_path = output_dir / f"tile_{tile_idx[0]}_{tile_idx[1]}_{tile_idx[2]}.ply"
        tile_gs.export_gs_to_ply(save_path)
        # print(f"Saved tile {tile_idx} with {tile_gs.num_of_point} Gaussians to {save_path}")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--input_root", default="./dataset/ours/longdress_gs", type=str)
    parser.add_argument("--output_root", default="./tiling_output/longdress_gs_tiled", type=str)
    
    parser.add_argument("--iteration", default=30000, type=int)
    
    parser.add_argument("--start_frame", default=0, type=int)
    parser.add_argument("--total_frames", default=12, type=int)
    
    parser.add_argument("--gs_type", default="gs", type=str,
                        choices=["gs", "lapisgs", "dlapisgs"], help="Type of Gaussian scene representation")
    # gs_type is "lapisgs" or "dlapisgs"
    parser.add_argument("--total_layers", default=1, type=int, help="Total number of layers in the Gaussian scene (only for lapisgs)")
    # gs_type is "dlapisgs"
    parser.add_argument("--frame_in_group", default=2, type=int, help="Number of frames in each group for dynamic tiling (only for dlapisgs)")
    
    
    parser.add_argument("--tiling_method", default="uniform", type=str, 
                        choices=["uniform", "uniform_dynamic"])
    
    # When tiling_method is "uniform" 
    parser.add_argument("--grid_shape", nargs=3, type=int, default=[4, 4, 4], help="Number of tiles along each axis (x, y, z)")
    
    args = parser.parse_args()
    
    start_time = time.time()
    if args.tiling_method == "uniform":  
        if args.gs_type == "gs" or args.gs_type == "lapisgs":
            # "gs" and "lapisgs" can both use the same tiling method, 
            # since the tiling is done independently for each frame and each layer, 
            # without considering temporal consistency. 
            # The only difference is that for "lapisgs" we have multiple layers to process, 
            # while for "gs" we only have one layer (layer 0).
            
            # Each frame is cutting into tiles independently, without considering temporal consistency
            for i in range(args.start_frame, args.start_frame + args.total_frames):
                print(f"Processing frame {i}...")
                gs_path_list = [f"{args.input_root}/frame_{i:04d}/lod{layer_idx}/point_cloud/iteration_{args.iteration}/point_cloud.ply" for layer_idx in range(args.total_layers)]
                gs_list = [GaussianModelV2(gs_path) for gs_path in gs_path_list]
                tile_aabbs, tile_indices, scene_min, scene_max = tiling_uniform_layered_gs(gs_list, args.grid_shape)
                
                for layer_idx in range(args.total_layers):
                    print(f"Processing layer {layer_idx} of frame {i}...")
                    output_dir = Path(args.output_root) / f"frame_{i:04d}" / f"lod{layer_idx}"
                    output_dir.mkdir(parents=True, exist_ok=True)
                
                    save_tiles_to_npz(
                        tile_aabbs=tile_aabbs,
                        tile_indices=tile_indices,
                        output_path=output_dir / "tiles_metadata.npz",
                        grid_shape=args.grid_shape,
                        scene_min=scene_min,
                        scene_max=scene_max,
                        layer_idx=layer_idx
                    )
                    
                    export_tiles_to_ply(gs_list[layer_idx], tile_indices, layer_idx, output_dir)
                    
        elif args.gs_type == "dlapisgs":
            # Handle "dlapisgs" type if needed
            # Only use the first frame for tiling, since the dynamic tiling will be applied in the later pipeline stage
            # Definitely not the best solution, but follow LTS paper design for now, since the dynamic tiling is only applied to the first frame in their paper.
            # Each frame is cutting into tiles independently, without considering temporal consistency
            
            # Check the relationship between total_frames and frame_in_group
            if args.frame_in_group <= 0:
                raise ValueError("frame_in_group should be a positive integer.")
            elif args.frame_in_group > args.total_frames:
                raise ValueError("frame_in_group should not be greater than total_frames.")
            # Check how many groups
            if args.total_frames % args.frame_in_group != 0:
                raise ValueError("Warning: total_frames is not divisible by frame_in_group. The last group will have fewer frames.")
            
            total_group = int(args.total_frames // args.frame_in_group)
            print(f"Total groups: {total_group}")
            for group_offset in range(total_group):
                group_start_frame = args.start_frame + group_offset * args.frame_in_group
                print(f"Processing group {group_offset} (frames {group_start_frame} to {group_start_frame + args.frame_in_group - 1})...")
                
                # Intra frame tiling for the first frame in the group, and then apply the same tiling to the rest frames in the group
                gs_path_list = [f"{args.input_root}/frame_{group_start_frame:04d}/lod{layer_idx}/point_cloud/iteration_30000/point_cloud.ply" for layer_idx in range(args.total_layers)]
                gs_list = [GaussianModelV2(gs_path) for gs_path in gs_path_list]
                # [Important] Only use the first frame for tiling, since the dynamic tiling will be applied in the later pipeline stage.
                tile_aabbs, tile_indices, scene_min, scene_max = tiling_uniform_layered_gs(gs_list, args.grid_shape)
                
                for layer_idx in range(args.total_layers):
                    print(f"Processing layer {layer_idx} of frame {group_start_frame}...")
                    output_dir = Path(args.output_root) / f"frame_{group_start_frame:04d}" / f"lod{layer_idx}"
                    output_dir.mkdir(parents=True, exist_ok=True)
                
                    save_tiles_to_npz(
                        tile_aabbs=tile_aabbs,
                        tile_indices=tile_indices,
                        output_path=output_dir / "tiles_metadata.npz",
                        grid_shape=args.grid_shape,
                        scene_min=scene_min,
                        scene_max=scene_max,
                        layer_idx=layer_idx
                    )
                    
                    export_tiles_to_ply(gs_list[layer_idx], tile_indices, layer_idx, output_dir)
                        
                # Inter frame tiling for the rest frames in the group, directly use the tile_aabbs and tile_indices from the first frame, without considering temporal consistency
                for frame_offset in range(1, args.frame_in_group):
                    frame_idx = group_start_frame + frame_offset
                    
                    gs_path_list = [f"{args.input_root}/frame_{frame_idx:04d}/lod{layer_idx}/point_cloud/iteration_30000/point_cloud.ply" for layer_idx in range(args.total_layers)]
                    gs_list = [GaussianModelV2(gs_path) for gs_path in gs_path_list]
                    # [Important] Skip tiling for the rest frames in the group, directly use the tile_aabbs and tile_indices from the first frame, without considering temporal consistency 
                
                    for layer_idx in range(args.total_layers):
                        print(f"Processing layer {layer_idx} of frame {frame_idx}...")
                        output_dir = Path(args.output_root) / f"frame_{frame_idx:04d}" / f"lod{layer_idx}"
                        output_dir.mkdir(parents=True, exist_ok=True)
                    
                        save_tiles_to_npz(
                            tile_aabbs=tile_aabbs,
                            tile_indices=tile_indices,
                            output_path=output_dir / "tiles_metadata.npz",
                            grid_shape=args.grid_shape,
                            scene_min=scene_min,
                            scene_max=scene_max,
                            layer_idx=layer_idx
                        )
                        
                        export_tiles_to_ply(gs_list[layer_idx], tile_indices, layer_idx, output_dir)
                    
        else:
            raise ValueError(f"Unsupported gs_type: {args.gs_type}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total tiling time: {elapsed_time:.2f} seconds")

"""
# Example usage:

## GS
python tiling.py \
--input_root ./dataset/ours/materials_gs \
--output_root ./tiling_output/materials_gs_tiled/uniform \
--iteration 15000 \
--start_frame 0 --total_frames 1 \
--gs_type gs \
--tiling_method uniform \
--grid_shape 2 2 2

## LapisGS
python tiling.py \
--input_root ./dataset/ours/lego_lapisgs \
--output_root ./tiling_output/lego_lapisgs_tiled/uniform \
--iteration 30000 \
--start_frame 0 --total_frames 1 \
--gs_type lapisgs \
--total_layers 4 \
--tiling_method uniform \
--grid_shape 2 2 2

# DLapisGS (Not Yet Implemented)
python tiling.py \
--input_root ./dataset/ours/longdress_dlapisgs \
--output_root ./tiling_output/longdress_dlapisgs_tiled/uniform \
--start_frame 0 --total_frames 12 \
--gs_type dlapisgs \
--frame_in_group 3 \
--total_layers 4 \
--tiling_method uniform \
--grid_shape 2 2 2
"""
