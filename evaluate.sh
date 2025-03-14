#!/bin/sh
# export hydra_run_dir=${hydra_run_dir}
# export weights_path=${weights_path}
# export CUDA_VISIBLE_DEVICES=7
# export CUDA_LAUNCH_BLOCKING=1
# # # # # # # re10k testing
# python evaluate.py \
#     hydra.run.dir=exp/re10k_v2/ \
#     hydra.job.chdir=true \
#     +experiment=layered_re10k \
#     +dataset.crop_border=true \
#     dataset.test_split_path=splits/re10k_mine_filtered/test_files.txt \
#     model.depth.version=v1 \
#     run.weights_path=/home/wuxianzu/Projects/flash3d/man/flash3d/exp/re10k_v2/checkpoints/model_0240000.pth \
#     ++eval.save_vis=false
# python e.py \
#     hydra.run.dir=$1 \
#     hydra.job.chdir=true \
#     +experiment=layered_re10k \
#     +dataset.crop_border=true \
#     dataset.test_split_path=splits/re10k_mine_filtered/test_files.txt \
#     model.depth.version=v1 \
#     ++eval.save_vis=false

# re10k testevalayer
# python evaluate_layer.py \
#     hydra.run.dir=${hydra_run_dir} \
#     hydra.job.chdir=true \
#     +experiment=layered_re10k \
#     +dataset.crop_border=true \
#     dataset.test_split_path=splits/re10k_mine_filtered/test.txt \
#     model.depth.version=v1 \
#     run.weights_path=${weights_path} \
#     ++eval.save_vis=false
# re10k testing layer
# python evaluate_layer.py \
#     hydra.run.dir=${hydra_run_dir} \
#     hydra.job.chdir=true \
#     +experiment=layered_re10k \
#     +dataset.crop_border=true \
#     dataset.test_split_path=splits/re10k_mine_filtered/test_files.txt \
#     model.depth.version=v1 \
#     run.weights_path=${weights_path} \
#     ++eval.save_vis=false
# python evaluate.py \
#     hydra.run.dir=$1 \
#     hydra.job.chdir=true \
#     +experiment=layered_re10k \
#     +dataset.crop_border=true \
#     dataset.test_split_path=splits/re10k_mine_filtered/test_files.txt \
#     model.depth.version=v1 \
#     ++eval.save_vis=false
# re10k testing Comparison with Two-view Methods Extrapolation
# python evaluate.py \
#     hydra.run.dir=exp/re10k_v2/ \
#     hydra.job.chdir=true \
#     +experiment=layered_re10k \
#     +dataset.crop_border=true \
#     dataset.test_split_path=splits/re10k_latentsplat/test_closer_as_src.txt \
#     model.depth.version=v1 \
#     run.weights_path=/home/wuxianzu/Projects/flash3d/man/flash3d/exp/re10k_v2/checkpoints/model_0240000.pth \
#     ++eval.save_vis=false

# # re10k stating Comparison with Two-view Methods Extrapolation
# python stat.py \
#     hydra.run.dir=${hydra_run_dir} \
#     hydra.job.chdir=true \
#     +experiment=layered_re10k \
#     +dataset.crop_border=true \
#     dataset.test_split_path=splits/re10k_latentsplat/test_closer_as_src.txt \
#     model.depth.version=v2 \
#     run.weights_path=${weights_path} \
#     ++eval.save_vis=false

KITTI testing
python evaluate.py \
    hydra.run.dir=exp/kitti \
    hydra.job.chdir=true \
    +experiment=layered_kitti \
    +dataset.crop_border=true \
    dataset.split_path=/home/wuxianzu/Projects/flash3d/man/flash3d/splits/\
    run.weights_path=/home/wuxianzu/Projects/flash3d/man/flash3d/exp/re10k_v2/checkpoints/model_re10k_v54.pth \
    model.depth.version=v1 \
    ++eval.save_vis=false

# # NYUV2 testing
# python evaluate.py \
#     hydra.run.dir=$1 \
#     hydra.job.chdir=true \
#     +experiment=layered_nyuv2 \
#     +dataset.crop_border=true \
#     dataset.split_path=splits/nyuv2/val_files.txt \
#     model.depth.version=v1 \
#     ++eval.save_vis=false