from pathlib import Path

import sys
sys.path.append('/home/syjintw/Desktop/NUS/GS-Interface')
import io_3dgs
from io_3dgs import GaussianModelV2

scene_name = "lego"

dataset_gs_root = Path(f"/home/syjintw/Desktop/NUS/GGSP/dataset/lapisgs/lego/opacity")
our_gs_root = Path(f"/home/syjintw/Desktop/NUS/GGSP/dataset/ours2gs/lego_lapisgs/frame_0000")

res_list = [8, 4, 2, 1] # from low to high
lod_list = [0, 1, 2, 3] # from low to high

for layer, res in zip(lod_list, res_list):
    dataset_gs_path = dataset_gs_root / f"lego_res{res}" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    our_gs_path = our_gs_root / f"lod{layer}" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    
    dataset_gs = GaussianModelV2(dataset_gs_path)
    our_gs = GaussianModelV2(our_gs_path)
    
    is_diff = io_3dgs.is_diff(dataset_gs, our_gs)
    print(f"Layer {layer} / Resolution {res}: {is_diff}")        
    