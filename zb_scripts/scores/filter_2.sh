NUM_INSTANCES=8

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_PREFIX="filter_${TIMESTAMP}"

for i in $(seq 0 $((NUM_INSTANCES - 1)))
do
    echo "启动第 $i 个"
    nohup env CUDA_VISIBLE_DEVICES=$i \
    python /mmu-vcg/zb08/codes/UniWorld-main/UniWorld-V2/zb_scripts/scores/filter_texture.py --num_splits $NUM_INSTANCES --split_index $i --dst_img_p /mmu-vcg/zb08/outputs/texture_edit/sft_base/2026.01.07_17.18_0_eval_train_more_e1/edited  --dst_foldr /mmu-vcg/zb08/outputs/texture_edit/final_old_scores/tain_more_e1 \
    > logs/${LOG_PREFIX}_texture_edit_log_$i.out 2>&1 & 
    sleep 1
done


sleep 1200


for i in $(seq 0 $((NUM_INSTANCES - 1)))
do
    echo "启动第 $i 个"
    nohup env CUDA_VISIBLE_DEVICES=$i \
    python /mmu-vcg/zb08/codes/UniWorld-main/UniWorld-V2/zb_scripts/scores/filter_texture.py --num_splits $NUM_INSTANCES --split_index $i --dst_img_p /mmu-vcg/zb08/outputs/texture_edit/sft_base/2026.01.07_17.18_0_eval_train_more_e2/edited --dst_foldr /mmu-vcg/zb08/outputs/texture_edit/final_old_scores/tain_more_e2 \
    > logs/${LOG_PREFIX}_texture_edit_log_$i.out 2>&1 & 
    sleep 1
done