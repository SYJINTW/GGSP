import sys
sys.path.append('/mnt/data1/syjintw/GS-Interface')
import io_3dgs
from io_3dgs import GaussianModelV2

original_start_frame_idx = 1051

layer_list = [0, 1, 2, 3]
res_list = [8, 4, 2, 1]

for frame in range(1, 2):
    for layer, res in zip(layer_list, res_list):
        gs1 = GaussianModelV2(f"./dataset/dlapisgs/longdress/opacity/longdress_res{res}/dynamic_{original_start_frame_idx+frame:04d}/point_cloud/iteration_30000/point_cloud.ply")
        gs2 = GaussianModelV2(f"./lapisgs2gs_output/longdress_lapisgs/frame_{frame:04d}/lod{layer}/point_cloud/iteration_30000/point_cloud.ply")

        is_diff = io_3dgs.is_diff(gs1, gs2)
        print(f"Frame {frame}, Layer {layer} / Resolution {res}: {is_diff}")
        
        # print("Different keys in gs1 and gs2: ", set(gs1.data.keys()) - set(gs2.data.keys()))
        # print("Different keys in gs2 and gs1: ", set(gs2.data.keys()) - set(gs1.data.keys()))
        # print("Keys in gs1: ", set(gs1.data.keys()))
        # print("Keys in gs2: ", set(gs2.data.keys()))

