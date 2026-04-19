import sys
sys.path.append('/mnt/data1/syjintw/GS-Interface')
import io_3dgs
from io_3dgs import GaussianModelV2

for frame in range(12):
    gs1 = GaussianModelV2(f"./dataset/ours/longdress_gs/frame_{frame:04d}/lod0/point_cloud/iteration_30000/point_cloud.ply")
    gs2 = GaussianModelV2(f"./merged_output/longdress_gs/frame_{frame:04d}/lod0/point_cloud/iteration_30000/point_cloud.ply")

    is_diff = io_3dgs.is_diff(gs1, gs2)
    print(f"Frame {frame}: {is_diff}")

# keys = [i for i in gs1.data.keys()]
# gs1.sort_by_attributes(keys)
# gs2.sort_by_attributes(keys)
# gs1.export_gs_to_ply("./check_diff/gs1.ply")
# gs2.export_gs_to_ply("./check_diff/gs2.ply")


# print(io_3dgs.is_diff(gs1, gs2))