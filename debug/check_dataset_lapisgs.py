from pathlib import Path

import sys
sys.path.append('/mnt/data1/syjintw/GS-Interface')
import io_3dgs
from io_3dgs import GaussianModelV2

frame = 1080
res_list = [4, 2, 1] # from low to high

input_root = Path("/mnt/data1/syjintw/NUS/code/dataset/dlapisgs/longdress/opacity")

gs_list = []
num_of_gs_list = []
for res in res_list:
    gs_path = input_root / f"longdress_res{res}" / f"dynamic_{frame:04d}" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    gs = GaussianModelV2(gs_path)
    gs_list.append(gs)
    num_of_gs_list.append(gs.num_of_point)

print(f"Number of Gaussians for each resolution: {num_of_gs_list}")

# Check if the first few gs points are the same
for i in range(1, len(gs_list)):
    print(f"Compare res{res_list[i]} with res{res_list[i-1]}, num of points: {num_of_gs_list[i-1]}")
    diff_keys = []
    for key in gs_list[i].data.keys():
        if key in gs_list[i-1].data:
            data_i = gs_list[i].data[key]["data"][:num_of_gs_list[i-1]]
            data_i_1 = gs_list[i-1].data[key]["data"][:num_of_gs_list[i-1]]
            is_diff = not (data_i == data_i_1).all()
            if is_diff:
                diff_keys.append(key)
    print("Diff keys: ", diff_keys)