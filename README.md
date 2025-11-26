
# Continual Learning for Medical VQA with Prompt-Key Retrieval

This project implements a **Continual Learning (CL)** framework for Medical Visual Question Answering (Med-VQA). It adapts the **LLaVA-Med** architecture using a **Prompt-Key Retrieval** mechanism combined with **LoRA (Low-Rank Adaptation)**. This framework enables the model to learn new tasks (e.g., different medical domains like VQA-RAD) sequentially without suffering from Catastrophic Forgetting.

## 📂 Codebase Structure & Component Description

The codebase is organized as follows:

### Core Model Components
* **`llava/model/language_model/llava_llama.py`**: Defines the `LlavaLlamaForCausalLM` class. It integrates the `Retriever` module into the standard LLaVA architecture and handles the retrieval of task-specific prompts during the forward pass.
* **`llava/model/language_model/retriever.py`**: Implements the `Retriever` class. This module manages the "Prompt Keys" (learnable vectors) and "Weight Offsets" (LoRA parameters). It calculates similarity scores between input queries and keys to select the appropriate adapters.
* **`llava/model/llava_arch.py`**: Abstract architecture definition for LLaVA, handling vision tower encoding and multimodal projection logic.

### Training & Processing Scripts
* **`train_prompt_key.py`**: **[Stage 1 Training]** This script trains the retrieval keys independently. It maps input questions to specific task IDs (e.g., mapping an abdominal question to the "Abdomen" task slot).
* **`merge_prompt_key.py`**: A utility script to merge independently trained key checkpoints (e.g., Chest, Abdomen, Head) into a single `merged_prompt_key.pth` file. This is **crucial** for assembling the final CL system.
* **`finetune_continual.sh`**: **[Stage 2 Training]** The main shell script for continual instruction tuning. It iterates through tasks, loads the `merged_prompt_key`, and fine-tunes the LoRA weights using DeepSpeed Zero-3.
* **`fusion.py`**: Data preprocessing script. It reads raw CSV/JSON metadata (e.g., from VQA-RAD) and converts it into the specific JSONL format required by LLaVA (`question_id`, `image`, `text`, `gpt4_answer`, `domain`).

### Evaluation
* **`cv805.sh`**: The main evaluation entry point. It runs inference on test sets using `model_vqa_loader.py` and then calculates metrics using `eval_video_qa.py`.
* **`llava/eval/model_vqa_loader.py`**: Handles data loading and inference generation during evaluation. **Note:** This script requires the `--model-base` argument to correctly load the vision tower configuration.
* **`llava/eval/eval_video_qa.py`**: Calculates standard NLP metrics (BLEU, ROUGE, METEOR, CIDEr) for the generated answers compared to ground truth, broken down by domain (e.g., Abdomen, Chest, Head).

### Configuration
* **`scripts/zero3.json`**: DeepSpeed configuration file. **Note:** Ensure parameters like `stage3_prefetch_bucket_size` are set to integers (not floats) to avoid validation errors.

---

## 🛠️ Dependencies and Installation

This project relies on PyTorch, DeepSpeed, and the LLaVA ecosystem.

### Prerequisites
* Python >= 3.10
* CUDA 11.8 or 12.x (or ROCm 6.x for AMD users)

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-folder>
    ```

2.  **Create a Conda environment:**
    ```bash
    conda create -n llava python=3.10 -y
    conda activate llava
    ```

3.  **Install Core Dependencies:**
    ```bash
    pip install --upgrade pip
    # Adjust the following command for your specific CUDA or ROCm version
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    ```

4.  **Install Project Dependencies:**
    ```bash
    pip install transformers==4.37.2
    pip install accelerate
    pip install deepspeed
    pip install einops
    pip install opencv-python
    pip install shortuuid
    pip install datasets
    pip install pandas
    pip install pycocoevalcap
    ```

5.  **Fixing Common Environment Issues (Important):**
    * **Scikit-learn/Numpy ABI Conflict:** If you encounter `numpy.dtype size changed` errors during evaluation:
        ```bash
        pip install scikit-learn --no-cache-dir --force-reinstall
        ```
    * **Bitsandbytes (Quantization):**
        * For **NVIDIA** GPUs:
            ```bash
            pip install bitsandbytes
            ```
        * For **AMD (ROCm)** GPUs: You must verify your ROCm version and install the compatible wheel or compile from source. The standard `pip install bitsandbytes` may not work.
            ```bash
            # Example for ROCm (verify version first)
            pip install bitsandbytes-rocm
            ```

---

## 📊 Data Preparation

### 1. Download Data
This project uses the **VQA-RAD** dataset (or similar medical VQA datasets).
* **Download Link:** [VQA-RAD Dataset on HuggingFace](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) or [Original Source](https://osf.io/89k6j/).
* Place images in: `data/vqa-rad/images/`
* Place metadata in: `data/train/metadata.csv`

### 2. Preprocess Data
Use the `fusion.py` script to convert the raw metadata into the LLaVA-Med JSONL format required for training and evaluation.

```bash
python fusion.py
```

  * **Input:** `metadata.csv` and reference JSONs (`vqa_ABD.json`, etc.).
  * **Output:** `data/train_vqa_rad_combined.jsonl` (contains fields like `question_id`, `text`, `image`, `domain`).

-----

## 🚀 Running the Code

### Step 1: Train Prompt Keys

Train the retrieval keys for each specific task (e.g., 'abd', 'chest', 'head'). This allows the model to route questions to the correct adapter.

```bash
python train_prompt_key.py \
    --task 'abd' \
    --data_paths "data/train_abd.jsonl" \
    --save_path "output/prompt-key/abd.pth"

# Repeat for 'chest' and 'head' tasks
```

### Step 2: Merge Prompt Keys

Combine the independently trained keys into a single file containing both the keys and the initial LoRA weight offsets.

```bash
python merge_prompt_key.py
```

  * **Output:** `output/prompt-key/merged_prompt_key.pth`

### Step 3: Continual Fine-tuning

Run the main training loop to fine-tune the model sequentially on the tasks.
**Note:** Ensure `HIP_VISIBLE_DEVICES` (for AMD) or `CUDA_VISIBLE_DEVICES` (for NVIDIA) is set correctly in the script.

```bash
bash finetune_continual.sh
```

### Step 4: Evaluation

Evaluate the trained model on the test set.

**CRITICAL NAMING CONVENTION:** The model folder names in `cv805.sh` **MUST** contain the string `"llava"` (e.g., `llava-vqa-rad_abd`). If they do not, the model loader will fail to initialize the vision tower correctly.

```bash
bash cv805.sh
```

-----

## 📝 Demo File (Sample Input/Output)

**File:** `demo_sample.jsonl`

**Input (JSONL format):**

```json
{
    "question_id": 101,
    "image": "test_img_00101.jpg",
    "pair_id": "test_img_00101",
    "text": "is there evidence of air in the peritoneal cavity?\n<image>",
    "domain": {"abd": true, "chest": false, "head": false}
}
```

**Output (Generated Prediction):**

```json
{
    "question_id": 101,
    "prompt": "is there evidence of air in the peritoneal cavity?\n<image>",
    "text": "no",
    "answer_id": "uuid-1234-5678",
    "model_id": "llava-vqa-rad_abd",
    "metadata": {}
}
```

-----

## 📜 Credits and Differences

This project is built upon the codebase of **[LLaVA (Large Language-and-Vision Assistant)](https://github.com/haotian-liu/LLaVA)** and **[LLaVA-Med](https://github.com/microsoft/LLaVA-Med)**.

**Key Differences & Contributions:**

1.  **Prompt-Key Retrieval Mechanism:** Introduced `retriever.py` and modified `llava_llama.py` to support dynamic prompt retrieval. This allows the model to select task-specific parameter offsets (`weight_offset`) based on the input question embedding.
2.  **Task Masking:** Added logic to handle `pool_mask` in `llava_llama.py` to restrict parameter updates to specific task slots during the training phase.
3.  **Data Pipeline:** Implemented custom `fusion.py` scripts to handle domain-specific data splitting and formatting (e.g., for VQA-RAD).
4.  **Evaluation Logic:** Modified `eval_video_qa.py` to compute metrics (BLEU, CIDEr, ROUGE) specifically categorized by medical domains (Abdomen, Chest, Head) for more granular performance analysis.