from pathlib import Path

import sys
sys.path.append('/home/syjintw/Desktop/NUS/GS-Interface')
import io_3dgs
from io_3dgs import GaussianModelV2

scene_name = "lego"

our_gs_root = Path(f"/home/syjintw/Desktop/NUS/GGSP/dataset/ours/lego_lapisgs/frame_0000")
merged_gs_root = Path(f"/home/syjintw/Desktop/NUS/GGSP/merged_output/lego_lapisgs/frame_0000")

lod_list = [0, 1, 2, 3] # from low to high

for layer in lod_list:
    our_gs_path = our_gs_root / f"lod{layer}" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    merged_gs_path = merged_gs_root / f"lod{layer}" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    
    our_gs = GaussianModelV2(our_gs_path)
    dataset_gs = GaussianModelV2(merged_gs_path)
    
    is_diff = io_3dgs.is_diff(dataset_gs, our_gs)
    print(f"Layer {layer}: {is_diff}")        
    