#!/bin/bash
# cd /fsx/meng.cao/code/LLaVA
# HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 
set -e
export MIOPEN_CUSTOM_CACHE_DIR=".cache/miopen_cache"
export MIOPEN_USER_DB_PATH=".cache/miopen_db"
exp_name="vqa-rad"
# task_ids=("GeoChat_Instruct" "llava_med" "atom" "art" "astro" "agri" "chem" "climate")
# task_ids=("art" "climate" "llava_med" "GeoChat_Instruct" "agri" "chem" "astro" "atom")
# task_ids=("CT" "CXR" "Histopathology" "MRI")
# task_ids=("MRI" "CXR" "CT")
# declare -A file_map=(
#     ["CT"]="CT_data_3995"
#     ["CXR"]="CXR_data_4088"
#     ["Histopathology"]="Histopathology_data_3404"
#     ["MRI"]="MRI_data_4416"
# )

# declare -A lr_map=(
#     ["CT"]=2e-5
#     ["CXR"]=2e-5
#     ["Histopathology"]=2e-5
#     ["MRI"]=2e-5
# )

task_ids=("chest" "abd" "head")
declare -A file_map=(
    ["abd"]="train_abd"
    ["chest"]="train_chest"
    ["head"]="train_head"
)

declare -A lr_map=(
    ["abd"]=2e-5
    ["chest"]=2e-5
    ["head"]=2e-5
)

# task_ids=("sct" "smri" "sxray")
# declare -A file_map=(
#     ["sct"]="train_modality_CT"
#     ["smri"]="train_modality_MRI"
#     ["sxray"]="train_modality_X-Ray"
# )

# declare -A lr_map=(
#     ["sct"]=2e-5
#     ["smri"]=2e-5
#     ["sxray"]=2e-5
# )

init_model_name_or_path="/llava/output/llava-v1.5-7b"
output_name="llava_vqa-rad2"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
for task_id in "${task_ids[@]}"; do
    current_file_name=${file_map[$task_id]}
    current_lr=${lr_map[$task_id]}

    output_name="${output_name}_${task_id}"
    # image_folder="data/llava_med_for_cv805/upload_hf/training_images/${task_id}_data"
    # json_path="data/llava_med_for_cv805/upload_hf/training/${current_file_name}.json"

    image_folder="data/vqa-rad/data/train"
    json_path="data/vqa-rad/data/${current_file_name}.json"
    
    # image_folder="data/SLAKE/imgs"
    # json_path="data/SLAKE/${current_file_name}.json"
    deepspeed llava/train/train_mem.py \
        --deepspeed ./scripts/zero3.json \
        --model_name_or_path $init_model_name_or_path \
        --version v1 \
        --data_path $json_path \
        --cl_data_clss $task_id \
        --image_folder $image_folder \
        --vision_tower /llava/output/clip-vit-large-patch14-336 \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir /llava/output/$output_name \
        --num_train_epochs 1 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 50000 \
        --save_total_limit 1 \
        --learning_rate $current_lr \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 False \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \
        --disable_task_id False \
        --dataset_type $exp_name \
        --report_to wandb \
        --retriever_state_dict /llava/output/prompt-key/chest_abd_head.pth \


    # # For quick check
    # echo "==============="
    # echo "task_id: $task_id"
    # echo "init_model_name_or_path: $init_model_name_or_path"
    # echo "data_path: $data_path"
    # echo "output:"./output/"$exp_name"/llava-squential-v1.5-7b-vicuna"$output_name"
    # echo "==============="

    init_model_name_or_path="/llava/output/${output_name}"
done