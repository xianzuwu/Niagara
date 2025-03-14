
terminate() {
    echo "Terminating all background jobs..."
    jobs -p | xargs -r kill
    wait
    exit 1
}

TOTAL_GPUS=8

for ((i=0; i<$TOTAL_GPUS; i++))
do
    echo "Starting process on GPU $i for data partition $i"
    CUDA_VISIBLE_DEVICES=$i nohup python kunkun.py $i $i > output_gpu_$i.log 2>&1 &
done