#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.
export MIOPEN_CUSTOM_CACHE_DIR=".cache/miopen_cache"
export MIOPEN_USER_DB_PATH=".cache/miopen_db"
# --- Configuration Section (Please EDIT these paths!) ---


MODEL_PATHS=(
    "/llava/output/llava_vqa-rad2_chest"
    "/llava/output/llava_vqa-rad2_chest_abd"
    "/llava/output/llava_vqa-rad2_chest_abd_head"
)
QUESTION_FILE="data/vqa-rad/data/test_vqa_rad.jsonl"
IMAGE_FOLDER="data/vqa-rad/data/test"
EVAL_SPLIT_NAME="vqa-rad_eval"

# MODEL_PATHS=(
#     "/llava/output/llava-slake1_sct"
#     "/llava/output/llava-slake1_sct_smri"
#     "/llava/output/llava-slake1_sct_smri_sxray"
# )
# QUESTION_FILE="data/SLAKE/test_slake.jsonl"
# IMAGE_FOLDER="data/SLAKE/imgs"
# EVAL_SPLIT_NAME="slake_eval"

# MODEL_PATHS=(
#     "/llava/output/llava-med5_MRI"
#     "/llava/output/llava-med5_MRI_CXR"
#     "/llava/output/llava-med5_MRI_CXR_CT"
# )
# QUESTION_FILE="data/llava_med_for_cv805/upload_hf/llava_med_eval_qa50_qa.jsonl"
# IMAGE_FOLDER="data/llava_med_for_cv805/upload_hf/eval_images"
# EVAL_SPLIT_NAME="llava_med_eval"




# 5. Base name for the experiment (used for subdirectories)
EXP_NAME="my_continual_eval"

# 6. Conversation mode (should match the mode used during training, e.g., "vicuna_v1")
CONV_MODE="vicuna_v1"

# --- End Configuration Section ---


# --- Script Logic ---
# Get GPU list from environment variable or default to GPU 0
gpu_list="0,1,2,3,4,5,6,7"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

echo "Using GPUs: ${GPULIST[@]}"
echo "Number of parallel chunks: $CHUNKS"

for MODEL_PATH in "${MODEL_PATHS[@]}"; do

    echo "=========================================================================="
    echo "========= PROCESSING MODEL: $MODEL_PATH =========="
    echo "=========================================================================="

    # Extract the base model name for directory naming
    MODEL_BASENAME=$(basename "$MODEL_PATH")

    # Define the main evaluation directory
    EVAL_DIR="./eval/${EVAL_SPLIT_NAME}/answers/${EXP_NAME}_${MODEL_BASENAME}"
    echo "Evaluation results will be stored in: $EVAL_DIR"

    # Create necessary directories
    mkdir -p "$EVAL_DIR"

    # Step 1: Run parallel inference using model_vqa_loader
    echo "Starting parallel inference for $MODEL_BASENAME..."
    for IDX in $(seq 0 $((CHUNKS-1))); do
        GPU_IDX=${GPULIST[$IDX]}
        ANS_FILE="$EVAL_DIR/${CHUNKS}_${IDX}.jsonl"

        echo "Launching inference chunk $IDX on GPU $GPU_IDX, outputting to $ANS_FILE"

        HIP_VISIBLE_DEVICES=$GPU_IDX python -m llava.eval.model_vqa_loader \
            --model-path "$MODEL_PATH" \
            --question-file "$QUESTION_FILE" \
            --image-folder "$IMAGE_FOLDER" \
            --answers-file "$ANS_FILE" \
            --num-chunks "$CHUNKS" \
            --chunk-idx "$IDX" \
            --temperature 0 \
            --conv-mode "$CONV_MODE"
    done

    # Wait for all background inference jobs to complete
    wait
    echo "Parallel inference finished for $MODEL_BASENAME."

    # Step 2: Merge the results
    MERGED_ANSWER_FILE="$EVAL_DIR/merged_predictions.jsonl"
    echo "Merging prediction files into $MERGED_ANSWER_FILE..."

    # Clear out the merged file if it exists.
    > "$MERGED_ANSWER_FILE"

    # Loop through the indices and concatenate each chunk's result file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CHUNK_FILE="$EVAL_DIR/${CHUNKS}_${IDX}.jsonl"
        if [[ -f "$CHUNK_FILE" ]]; then
            cat "$CHUNK_FILE" >> "$MERGED_ANSWER_FILE"
            echo "Appended $CHUNK_FILE"
        else
            echo "Warning: Chunk file $CHUNK_FILE not found."
        fi
    done
    echo "Merging complete for $MODEL_BASENAME."

    # Step 3: Run evaluation using the modified script with NLP metrics
    SCORES_FILE="$EVAL_DIR/nlp_metrics_scores.json"
    echo "Running evaluation using NLP metrics for $MODEL_BASENAME..."

    python llava/eval/eval_video_qa.py \
        --pred-path "$MERGED_ANSWER_FILE" \
        --gt-file "$QUESTION_FILE" \
        --output-json "$SCORES_FILE"

    echo "-------------------------------------"
    echo "Evaluation complete for $MODEL_BASENAME."
    echo "NLP metrics scores saved to: $SCORES_FILE"
    echo "You can view the scores by running: cat $SCORES_FILE"
    echo "-------------------------------------"

done

echo "=========================================================================="
echo "========= ALL MODELS HAVE BEEN EVALUATED. =========="
echo "=========================================================================="