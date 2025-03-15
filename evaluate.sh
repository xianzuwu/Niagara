#!/bin/sh
# export hydra_run_dir=${hydra_run_dir}
# export weights_path=${weights_path}
# export CUDA_VISIBLE_DEVICES=7
# export CUDA_LAUNCH_BLOCKING=1
# # # # # # # re10k testing
python evaluate.py \
    hydra.run.dir=exp/re10k_v2/ \
    hydra.job.chdir=true \
    +experiment=layered_re10k \
    +dataset.crop_border=true \
    dataset.test_split_path=splits/re10k_mine_filtered/test_files.txt \
    model.depth.version=v1 \
    run.weights_path=/home/wuxianzu/Projects/man/Niagara/exp/re10k_v2/checkpoints/model_re10k_v54.pth \
    ++eval.save_vis=false

# # # # # # #  KITTI testing
# python evaluate.py \
#     hydra.run.dir=exp/kitti \
#     hydra.job.chdir=true \
#     +experiment=layered_kitti \
#     +dataset.crop_border=true \
#     dataset.split_path=/home/wuxianzu/Projects/man/Niagara/splits/\
#     run.weights_path=/home/wuxianzu/Projects/man/Niagara/exp/re10k_v2/checkpoints/model_re10k_v54.pth \
#     model.depth.version=v1 \
#     ++eval.save_vis=false
