import sys
sys.path.append('/mnt/data1/syjintw/GS-Interface')
import io_3dgs
from io_3dgs import GaussianModelV2

for frame in range(12):
    for layer in range(4):
        gs1 = GaussianModelV2(f"./dataset/ours/longdress_lapisgs/frame_{frame:04d}/lod{layer}/point_cloud/iteration_30000/point_cloud.ply")
        gs2 = GaussianModelV2(f"./merged_output/longdress_lapisgs/frame_{frame:04d}/lod{layer}/point_cloud/iteration_30000/point_cloud.ply")

        is_diff = io_3dgs.is_diff(gs1, gs2)
        print(f"Frame {frame}, Layer {layer}: {is_diff}")

