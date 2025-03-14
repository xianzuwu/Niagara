#!/bin/bash

# set tmp
export TMPDIR=/home/wuxianzu/Projects/flash3d/flash3d/tmp

# Set visible GPU devices (e.g., use GPUs #0-3)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

# Defines the action to be performed when Ctrl + C is pressed
cleanup() {
  echo "The process is being cleaned up..."
  if [ -n "$YOUR_PID" ]; then
    kill -- -$YOUR_PID 2>/dev/null
    echo "The process group $YOUR_PID has ended."
  fi
  exit 0
}

trap cleanup SIGINT

python train.py -m \
  hydra/launcher=basic \
  +hydra.job.tag=gaussian2_unidepthv1 \
  +experiment=layered_re10k \
  model.depth.version=v1 \
  train.logging=true

YOUR_PID=$!
echo "The background process group $YOUR_PID has been started."

wait $YOUR_PID

cleanup