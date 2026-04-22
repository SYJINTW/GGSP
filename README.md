# GGSP

Dataset results' format to our format

Handle single frame
```bash
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
```


Divide Gaussian scene into tiles
```bash
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
```

Merging all the tiles back to a single 3DGS
```bash
## GS
python merge_tiles.py \
--input_root ./tiling_output/materials_gs_tiled/uniform/frame_0000/lod0 \
--output_root ./merged_output/materials_gs/frame_0000/lod0/point_cloud/iteration_15000/point_cloud.ply \
--gs_type gs \
--tiling_method uniform \
--selected_tiles all

## LapisGS
### LOD0
python merge_tiles.py \
--input_root ./tiling_output/lego_lapisgs_tiled/uniform/frame_0000/lod0 \
--output_root ./merged_output/lego_lapisgs/frame_0000/lod0/point_cloud/iteration_30000/point_cloud.ply \
--gs_type lapisgs \
--tiling_method uniform \
--selected_tiles all

### LOD1
python merge_tiles.py \
--input_root ./tiling_output/lego_lapisgs_tiled/uniform/frame_0000/lod1 \
--output_root ./merged_output/lego_lapisgs/frame_0000/lod1/point_cloud/iteration_30000/point_cloud.ply \
--gs_type lapisgs \
--tiling_method uniform \
--selected_tiles all
```