#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.
export MIOPEN_CUSTOM_CACHE_DIR="/vast/users/xiaodan/haokunlin/.cache/miopen_cache"
export MIOPEN_USER_DB_PATH="/vast/users/xiaodan/haokunlin/.cache/miopen_db"
# --- Configuration Section (Please EDIT these paths!) ---

# 1. Path to your FINISHED continual learning model directory
#    (e.g., the one created after training CT, CXR, Histopathology, AND MRI)
MODEL_PATH="/vast/users/xiaodan/haokunlin/Continual_LLaVA/llava/output/llava-v1.5-7b" # <--- MUST EDIT: Update with your final model path

# 2. Path to the evaluation question file (the .jsonl you provided)
QUESTION_FILE="/vast/users/xiaodan/haokunlin/data/llava_med_for_cv805/upload_hf/llava_med_eval_qa50_qa.jsonl" # <--- MUST EDIT: Update with the actual path

# 3. Path to the directory containing the evaluation images
IMAGE_FOLDER="/vast/users/xiaodan/haokunlin/data/llava_med_for_cv805/upload_hf/eval_images" # <--- MUST EDIT: Update with the actual path if different

# 4. Name for this specific evaluation run (used for subdirectories)
EVAL_SPLIT_NAME="llava_med_eval_qa50"

# 5. Base name for the experiment (used for subdirectories)
EXP_NAME="my_continual_eval" # Or keep 'newdomain_order3' if preferred

# 6. Conversation mode (should match the mode used during training, e.g., "vicuna_v1")
CONV_MODE="vicuna_v1" # <--- Verify this matches your training setup

# --- End Configuration Section ---


# --- Script Logic ---
# Get GPU list from environment variable or default to GPU 0
gpu_list="0,1,2,3,4,5,6,7"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

echo "Using GPUs: ${GPULIST[@]}"
echo "Number of parallel chunks: $CHUNKS"

# Extract the base model name for directory naming
MODEL_BASENAME=$(basename "$MODEL_PATH")

# Define the main evaluation directory
EVAL_DIR="./eval/${EVAL_SPLIT_NAME}/answers/${EXP_NAME}_${MODEL_BASENAME}"
echo "Evaluation results will be stored in: $EVAL_DIR"

# Create necessary directories
mkdir -p "$EVAL_DIR"

# Step 1: Run parallel inference using model_vqa_loader
echo "Starting parallel inference..."
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
        --conv-mode "$CONV_MODE" \
        --batch-size 1 &

done

# Wait for all background inference jobs to complete
wait
echo "Parallel inference finished."

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
echo "Merging complete."

# Step 3: Run evaluation using the modified script with NLP metrics
SCORES_FILE="$EVAL_DIR/nlp_metrics_scores.json"
echo "Running evaluation using NLP metrics..."

python llava/eval/eval_video_qa.py \
    --pred-path "$MERGED_ANSWER_FILE" \
    --gt-file "$QUESTION_FILE" \
    --output-json "$SCORES_FILE"

echo "-------------------------------------"
echo "Evaluation complete."
echo "NLP metrics scores saved to: $SCORES_FILE"
echo "You can view the scores by running: cat $SCORES_FILE"
echo "-------------------------------------"