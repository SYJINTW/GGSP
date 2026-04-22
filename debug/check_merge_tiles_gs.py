from pathlib import Path

import sys
sys.path.append('/home/syjintw/Desktop/NUS/GS-Interface')
import io_3dgs
from io_3dgs import GaussianModelV2

scene_name = "materials"
iteration = 15000

our_gs_root = Path(f"/home/syjintw/Desktop/NUS/GGSP/dataset/ours/{scene_name}_gs/frame_0000")
merged_gs_root = Path(f"/home/syjintw/Desktop/NUS/GGSP/merged_output/{scene_name}_gs/frame_0000")

our_gs_path = our_gs_root / f"lod0" / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
merged_gs_path = merged_gs_root / f"lod0" / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"

our_gs = GaussianModelV2(our_gs_path)
dataset_gs = GaussianModelV2(merged_gs_path)

is_diff = io_3dgs.is_diff(dataset_gs, our_gs)
print(f"Merging result: {is_diff}") # False means they are the same, True means they are different      
