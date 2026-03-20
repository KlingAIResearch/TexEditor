NUM_INSTANCES=8

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_PREFIX="more_${TIMESTAMP}"

for i in $(seq 0 $((NUM_INSTANCES - 1)))
do
    echo "启动第 $i 个"
    nohup env CUDA_VISIBLE_DEVICES=$i \
    python zb_scripts/scores/make_more.py --num_splits $NUM_INSTANCES --split_index $i \
    > logs/${LOG_PREFIX}_texture_edit_log_$i.out 2>&1 & 
    sleep 1
done