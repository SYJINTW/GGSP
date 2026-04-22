from pathlib import Path

import sys
sys.path.append('/home/syjintw/Desktop/NUS/GS-Interface')
import io_3dgs
from io_3dgs import GaussianModelV2

res_list = [8, 4, 2, 1] # from low to high
scene_name = "longdress"

# input_root = Path(f"../dataset/dlapisgs/{scene_name}/opacity")
input_root = Path(f"/home/syjintw/Desktop/NUS/dynamic-lapis-gs/model/8i/longdress/my-dynamic-lapis")

start_frame = 1051
group_of_frames = 2
total_groups = 1

iterations = 3000

for group_idx in range(total_groups):
    print(f"Group {group_idx}: frames {start_frame + group_idx*group_of_frames} to {start_frame + (group_idx+1)*group_of_frames - 1}")

    for frame_offset in range(group_of_frames):
        frame_idx = start_frame + group_idx*group_of_frames + frame_offset
        print(f"Frame {frame_idx}")
        
        # Check LapisGS policy
        gs_list = []
        num_of_gs_list = []
        for res in res_list:
            if frame_offset == 0:
                gs_path = input_root / f"{scene_name}_res{res}" / f"{frame_idx:04d}" / f"point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
            else:
                gs_path = input_root / f"{scene_name}_res{res}" / f"dynamic_{frame_idx:04d}" / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
            
            # gs_path = input_root / f"{scene_name}_res{res}" / f"dynamic_{frame_idx:04d}" / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
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