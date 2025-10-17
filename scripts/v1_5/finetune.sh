#!/bin/bash

# cd /fsx/meng.cao/code/LLaVA
# HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 

# --model_name_or_path ./checkpoints/vicuna-7b-v1.5

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ptuningoutput/Continual-LLaVA-usePretrainMLP/Continual-LLaVA-usePretrainMLP-chartqa \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_mix665k.json \
    --cl_data_clss "chartqa" \
    --image_folder ./playground/data \
    --vision_tower ./checkpoints/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./output/llava-continual-v1.5-7b-vicuna-debug \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard 

# deepspeed --include localhost:1 llava/train/train_mem.py