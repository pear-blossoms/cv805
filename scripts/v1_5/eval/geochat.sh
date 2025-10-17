#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-order3-newdomain-atom-art-astro-agri-chem-climate-llava_med-GeoChat_Instruct"
SPLIT="geochat_val"
exp_name='newdomain_order3'
mkdir -p "eval/geochat/answers/${SPLIT}/${exp_name}_${CKPT}"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python  llava/eval/geochat_refering.py \
        --model-path ./checkpoints/$exp_name/$CKPT \
        --question-file /data/repos/Continual-LLaVA-NeXT/eval/geochat/GeoChat-Bench/region_captioning.jsonl \
        --image-folder /data/repos/dataset/GeoChat_Instruct/GeoChat_Instruct/share/softwares/kartik/GeoChat_finetuning/final_images_llava \
        --answers-file eval/geochat/answers/$SPLIT/${exp_name}_${CKPT}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

# --model-path liuhaotian/llava-v1.5-7b

wait

output_file=./eval/geochat/answers/$SPLIT/${exp_name}_${CKPT}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./eval/geochat/answers/$SPLIT/${exp_name}_${CKPT}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done
python3 llava/eval/eval_video_qa.py --num_tasks 8 --output_dir ./eval/geochat/answers/$SPLIT/${exp_name}_${CKPT}/gpt_outputdir  --output_json  ./eval/geochat/answers/$SPLIT/${exp_name}_${CKPT}/gt_pred_for_gpt_eval.json --pred_path $output_file

