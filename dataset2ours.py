import os
from pathlib import Path
import shutil
import numpy as np
from argparse import ArgumentParser
import time

import sys
# sys.path.append('/mnt/data1/syjintw/GS-Interface')
sys.path.append('/home/syjintw/Desktop/NUS/GS-Interface')

import io_3dgs
from io_3dgs import GaussianModelV2


def lapisgs_separate_level(gs_list):
    """
    Input:
    - gs_list: list of GaussianModelV2 objects. 
            The order should be from the lowest level of detail to the highest level of detail. 
            We expect all GaussianModelV2 have the same attributes.
    
    Output:
    - separated_gs_list: list of GaussianModelV2 objects.
                    The order is from the lowest level of detail to the highest level of detail.
                    Each GaussianModelV2 object contains the Gaussians for that level of detail, 
                    and the opacity for every level of detail.
    
    Description:
    (1) Get the opacity 
    (2) Separate higher level differences from lower level
    
    The highest level have all the Gaussians.
    But the opacity need to add more.
    
    """
    
    num_of_level = len(gs_list)
    num_of_point_list = [gs.num_of_point for gs in gs_list]
    # print("num_of_point_list: ", num_of_point_list)
    
    # # Debug: check if the dataset is correct by checking x
    # for gs in gs_list:
    #     print("x: ", gs.data["x"]["data"][:10])
        
    # Opacity
    for i in range(num_of_level):
        # print("Generate opacity for level ", i)
        opacity = np.zeros(num_of_point_list[-1], dtype=np.float32)
        # print(opacity.shape)
        opacity[:num_of_point_list[i]] = gs_list[i].data["opacity"]["data"]
        gs_list[-1].add_attribute(f"opacity_lod{i}", "f4", opacity)
    
    # Delete original opacity
    gs_list[-1].delete_attribute("opacity")
    
    # Separate based on the number of points
    separated_gs_list = []
    for i in range(num_of_level):
        # print("Separate level ", i)
        start_idx = num_of_point_list[i-1] if i > 0 else 0
        end_idx = num_of_point_list[i]
        # print(f"start_idx: {start_idx}, end_idx: {end_idx}")
        extracted_idx = [i for i in range(start_idx, end_idx)]
        separated_gs_list.append(gs_list[-1].extract_gaussians(extracted_idx))
          
    return separated_gs_list


def main_gs(scene_name,
            input_root, output_root,
            iteration,
            output_start_frame):
    """
    This is for processing the original Gaussians without separation.
    Just copy the original Gaussians to the output folder, and rename the folder structure to match our expected format.
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    
    output_frame = output_start_frame
    print(f"Processing frame --> {output_frame}")
        
    input_gs_path = input_root / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply" #! Change to your correct path
    output_gs_path = output_root / f"frame_{output_frame:04d}" / "lod0" / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    output_gs_path.parent.mkdir(parents=True, exist_ok=True)
    
    shutil.copy(input_gs_path, output_gs_path)


def main_lapisgs(scene_name,
                input_root, output_root,
                iteration,
                output_start_frame, 
                res_list, lod_list):
    input_root = Path(input_root)
    output_root = Path(output_root)
    
    output_frame = output_start_frame
    print(f"Processing frame --> {output_frame}")
    
    gs_list = []
    for res in res_list:
        #! Change to your correct path
        gs_path = input_root / f"{scene_name}_res{res}" / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply" 
        gs = io_3dgs.GaussianModelV2(gs_path)
        gs_list.append(gs)

    separated_gs_list = lapisgs_separate_level(gs_list)
    
    for i, gs in enumerate(separated_gs_list):
        output_path = output_root / f"frame_{output_frame:04d}" / f"lod{lod_list[i]}"
        output_gs_path = output_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
        output_gs_path.parent.mkdir(parents=True, exist_ok=True)
        gs.export_gs_to_ply(output_gs_path)
        

def main_dlapisgs(scene_name,
                input_root, output_root, 
                input_start_frame, output_start_frame, 
                res_list, lod_list, 
                total_group, frame_in_group):
    
    input_root = Path(input_root)
    output_root = Path(output_root)
    
    for group_offset in range(total_group):
        input_frame = input_start_frame + group_offset * frame_in_group
        output_frame = output_start_frame + group_offset * frame_in_group
        print(f"Intra: {input_frame} --> {output_frame}")
        
        # intraframe
        gs_list = []
        for res in res_list:
            gs_path = input_root / f"{scene_name}_res{res}" / f"dynamic_{input_frame:04d}" / "point_cloud" / "iteration_30000" / "point_cloud.ply" #! Change to your correct path
            gs = io_3dgs.GaussianModelV2(gs_path)
            gs_list.append(gs)

        intra_separated_gs_list = lapisgs_separate_level(gs_list)
        
        for i, gs in enumerate(intra_separated_gs_list):
            output_path = output_root / f"frame_{output_frame:04d}" / f"lod{lod_list[i]}"
            output_gs_path = output_path / "point_cloud" / "iteration_30000" / "point_cloud.ply"
            output_gs_path.parent.mkdir(parents=True, exist_ok=True)
            gs.export_gs_to_ply(output_gs_path)

        # interframe
        for frame_offset in range(1, frame_in_group):
            input_frame = input_start_frame + group_offset * frame_in_group + frame_offset
            output_frame = output_start_frame + group_offset * frame_in_group + frame_offset
            print(f"Inter: {input_frame} --> {output_frame}")
            
            gs_list = []
            for res in res_list:    
                gs = io_3dgs.GaussianModelV2(input_root / f"longdress_res{res}" / f"dynamic_{input_frame:04d}" / "point_cloud" / "iteration_30000" / "point_cloud.ply")
                gs_list.append(gs)
                
            inter_separated_gs_list = lapisgs_separate_level(gs_list)
            
            # For debugging: check the difference between intra_separated_gs_list and inter_separated_gs_list
            # # position diff
            # print(intra_separated_gs_list[0].data["x"]["data"][:10] - inter_separated_gs_list[0].data["x"]["data"][:10])
            # # sh same
            # print(intra_separated_gs_list[0].data["f_dc_0"]["data"][:10] - inter_separated_gs_list[0].data["f_dc_0"]["data"][:10])
            # # scale same
            # print(intra_separated_gs_list[0].data["scale_0"]["data"][:10] - inter_separated_gs_list[0].data["scale_0"]["data"][:10])
            # # rot diff
            # print(intra_separated_gs_list[0].data["rot_0"]["data"][:10] - inter_separated_gs_list[0].data["rot_0"]["data"][:10])
            # # opacity_lod0 same
            # print(intra_separated_gs_list[0].data["opacity_lod0"]["data"][:10] - inter_separated_gs_list[0].data["opacity_lod0"]["data"][:10])
            # # opacity_lod1 same
            # print(intra_separated_gs_list[0].data["opacity_lod1"]["data"][:10] - inter_separated_gs_list[0].data["opacity_lod1"]["data"][:10])
            # # opacity_lod2 same
            # print(intra_separated_gs_list[0].data["opacity_lod2"]["data"][:10] - inter_separated_gs_list[0].data["opacity_lod2"]["data"][:10])
            
            for i in range(len(inter_separated_gs_list)):
                inter_separated_gs_list[i].data["x"]["data"] -= intra_separated_gs_list[i].data["x"]["data"]
                inter_separated_gs_list[i].data["y"]["data"] -= intra_separated_gs_list[i].data["y"]["data"]
                inter_separated_gs_list[i].data["z"]["data"] -= intra_separated_gs_list[i].data["z"]["data"]
                inter_separated_gs_list[i].data["rot_0"]["data"] -= intra_separated_gs_list[i].data["rot_0"]["data"]
                
                # Clean unused attributes
                keys_to_remove = [k for k in inter_separated_gs_list[i].data.keys() if not (k.startswith(("x", "y", "z", "rot_", "opacity_lod")))]# Remove all attributes except position, rotation, and opacity
                for key in keys_to_remove:
                    inter_separated_gs_list[i].delete_attribute(key)
            
            for i, gs in enumerate(inter_separated_gs_list):
                output_path = output_root / f"frame_{output_frame:04d}" / f"lod{lod_list[i]}"
                output_gs_path = output_path / "point_cloud" / "iteration_30000" / "point_cloud.ply"
                output_gs_path.parent.mkdir(parents=True, exist_ok=True)
                gs.export_gs_to_ply(output_gs_path)


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--scene_name", default="longdress", type=str)
    parser.add_argument("--input_root", default="./dataset/dlapisgs/longdress/opacity", type=str)
    parser.add_argument("--output_root", default="./dataset/ours/longdress_dlapisgs", type=str)
    
    parser.add_argument("--iteration", default=30000, type=int)
    
    parser.add_argument("--input_start_frame", default=0, type=int) # Only dlapisgs need input_start_frame
    parser.add_argument("--output_start_frame", default=0, type=int)
    parser.add_argument("--total_frames", default=1, type=int)
    
    parser.add_argument("--gs_type", default="gs", type=str)
    
    # If gs_type is "gs", then we expect the input is the original Gaussians
    # If gs_type is "lapisgs"
    parser.add_argument("--res_list", nargs='*', default=[8, 4, 2, 1], type=int)
    parser.add_argument("--lod_list", nargs='*', default=[], type=int)
    
    # If gs_type is "dlapisgs"
    parser.add_argument("--frame_in_group", default=2, type=int)
    
    args = parser.parse_args()
    
    start_time = time.time()
    if args.gs_type == "gs":
        main_gs(args.scene_name,
                args.input_root, args.output_root, 
                args.iteration,
                args.output_start_frame)
    
    elif args.gs_type == "lapisgs" or args.gs_type == "dlapisgs":
        # Check if the number of res_list and lod_list are the same
        if len(args.lod_list) == 0: # Let lod start from 0 (lowest level) and increase by 1
            args.lod_list = list(range(len(args.res_list)))
        elif len(args.lod_list) != len(args.res_list):
            raise ValueError("The number of lod_list should match the number of res_list.")
        
        # Process the lapisgs
        if args.gs_type == "lapisgs":
            main_lapisgs(args.scene_name,
                        args.input_root, args.output_root, 
                        args.iteration,
                        args.output_start_frame, 
                        args.res_list, args.lod_list)
            
        elif args.gs_type == "dlapisgs":
            pass
            # # Check frame_in_group
            # if args.frame_in_group <= 0:
            #     raise ValueError("frame_in_group should be a positive integer.")
            # elif args.frame_in_group > args.total_frames:
            #     raise ValueError("frame_in_group should not be greater than total_frames.")
            # # Check how many groups
            # if args.total_frames % args.frame_in_group != 0:
            #     raise ValueError("Warning: total_frames is not divisible by frame_in_group. The last group will have fewer frames.")
            # total_group = int(args.total_frames // args.frame_in_group)
            # print(f"Total groups: {total_group}")
            
            # main_dlapisgs(args.scene_name,
            #     args.input_root, args.output_root, 
            #     args.input_start_frame, args.output_start_frame,
            #     args.res_list, args.lod_list, 
            #     total_group, args.frame_in_group)
    
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")

           
"""
# Example usage:

# Static
## GS
python dataset2ours.py \
--scene_name materials \
--input_root ./dataset/gs/materials \
--output_root ./dataset/ours/materials_gs \
--iteration 15000 \
--output_start_frame 0 \
--gs_type gs
    
## LapisGS
python dataset2ours.py \
--scene_name lego \
--input_root ./dataset/lapisgs/lego/opacity \
--output_root ./dataset/ours/lego_lapisgs \
--iteration 30000 \
--output_start_frame 0 \
--gs_type lapisgs \
--res_list 8 4 2 1 \
--lod_list 0 1 2 3

## DLapisGS (Not Yet Implemented)
python dataset2ours.py \
--scene_name longdress \
--input_root ./dataset/dlapisgs/longdress/opacity \
--output_root ./dataset/ours/longdress_dlapisgs \
--input_start_frame 1051 --output_start_frame 0 --total_frames 12 \
--gs_type dlapisgs \
--res_list 8 4 2 1 \
--lod_list 0 1 2 3 \
--frame_in_group 3
"""

