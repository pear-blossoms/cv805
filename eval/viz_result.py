import json
import pandas as pd
import matplotlib.pyplot as plt
import os

# Base directory where evaluation results are stored


### llava-med
# eval_base_dir = "./eval/llava_med_eval_qa50/answers/my_continual_eval_"
# model_dir_suffixes = [
#     "llava-v1.5-7b",              
#     "llava-med3_CXR",                   
#     "llava-med3_CXR_CT",                  
#     "llava-med3_CXR_CT_MRI",  
#     # "llava-med2_CXR_CT_MRI_Histopathology" 
# ]
# model_stage_labels = [
#     "Baseline",
#     "Learned CXR",
#     "Learned CT",
#     "Learned MRI",
#     # "Learned Histo",
# ]
# plot_save_dir = "./v/evaluation_plots_llava-med3"
# categories_to_plot = ["overall", "chest_xray", "ct_scan", "mri"]
# category_labels = {
#     "overall": "Overall",
#     "chest_xray": "cxr Task",
#     "ct_scan": "ct Task",
#     "mri": "mri Task"
# }

### vqa-rad
eval_base_dir = "./eval/vqa-rad_eval/answers/my_continual_eval_"
model_dir_suffixes = [
    "llava-v1.5-7b",            
    "llava_vqa-rad2_chest",                   
    "llava_vqa-rad2_chest_abd",                 
    "llava_vqa-rad2_chest_abd_head",  
]
plot_save_dir = "./v/evaluation_plots_vqa-rad2"
model_stage_labels = [
    "Baseline",
    "Learned chest",
    "Learned abd",
    "Learned head",
]
categories_to_plot = ["overall", "chest", "abd", "head"]
category_labels = {
    "overall": "Overall",
    "chest": "Chest Task",
    "abd": "Abd Task",
    "head": "Head Task"
}

### slake
# model_dir_suffixes = [
#     "llava-v1.5-7b",            
#     "llava-slake1_sct",                   
#     "llava-slake1_sct_smri",                 
#     "llava-slake1_sct_smri_sxray",  
# ]
# plot_save_dir = "./v/evaluation_plots_slake1"
# model_stage_labels = [
#     "Baseline",
#     "Learned ct",
#     "Learned mri",
#     "Learned xray",
# ]
# categories_to_plot = ["overall", "chest_xray", "ct_scan", "mri"]
# category_labels = {
#     "overall": "Overall",
#     "chest_xray": "cxr Task",
#     "ct_scan": "ct Task",
#     "mri": "mri Task"
# }

score_file_name = "nlp_metrics_scores.json"

assert len(model_dir_suffixes) == len(model_stage_labels), \
    f"List length mismatch! model_dir_suffixes ({len(model_dir_suffixes)}) vs model_stage_labels ({len(model_stage_labels)})"


# Ensure we only plot categories present in the configuration
categories_to_plot = [cat for cat in categories_to_plot if cat in category_labels]

# Metrics to plot (ensure these keys match those within each category's dictionary in the JSON)
metrics_to_plot = ["Bleu_4", "METEOR", "ROUGE_L", "CIDEr"]

os.makedirs(plot_save_dir, exist_ok=True)

# --- Removed Chinese Font Setting Block ---

# --- Data Loading and Processing ---
all_data = []
for i, model_suffix in enumerate(model_dir_suffixes):
    # Construct the path to the JSON file using base directory and suffix
    json_path = os.path.join(eval_base_dir + model_suffix, score_file_name)

    print(f"Attempting to load: {json_path}")
    try:
        with open(json_path, 'r') as f:
            scores_data = json.load(f)
            # Check basic structure
            if not isinstance(scores_data, dict):
                 print(f"Warning: File {json_path} is not a valid JSON object. Skipping.")
                 continue

            for category in categories_to_plot:
                if category in scores_data and isinstance(scores_data[category], dict):
                    category_scores = scores_data[category]
                    metrics_found = {metric: category_scores.get(metric) for metric in metrics_to_plot if metric in category_scores}

                    # Check if metrics exist and are numeric, fill with None otherwise
                    valid_metrics = True
                    row_metrics = {}
                    for metric in metrics_to_plot:
                        value = metrics_found.get(metric)
                        if value is None or not isinstance(value, (int, float)):
                            print(f"Warning: File {json_path}, category '{category}', metric '{metric}' is missing or non-numeric.")
                            row_metrics[metric] = None # Use None for missing/invalid data
                        else:
                            row_metrics[metric] = value

                    # Append data
                    row_data = {
                        'model_stage_label': model_stage_labels[i],
                        'model_stage_index': i,
                        'category': category,
                        **row_metrics
                    }
                    all_data.append(row_data)

                else:
                    print(f"Warning: Category '{category}' not found or is not a dictionary in file {json_path}.")

    except FileNotFoundError:
        print(f"Error: Score file not found at {json_path}. Skipping this model stage.")
    except json.JSONDecodeError:
        print(f"Error: Cannot parse JSON from {json_path}. Skipping this model stage.")
    except Exception as e:
         print(f"An unexpected error occurred loading {json_path}: {e}. Skipping this model stage.")

# --- Convert to DataFrame ---
if not all_data:
    print("Error: No valid data loaded. Cannot generate plots.")
else:
    df = pd.DataFrame(all_data)
    print("\nLoaded and Processed Data DataFrame:")
    print(df)
    print("\nDataFrame Info:")
    df.info()

    # --- Plotting ---
    print(f"\nGenerating plots for metrics: {metrics_to_plot}")
    available_metrics = [m for m in metrics_to_plot if m in df.columns]
    if not available_metrics:
        print("Error: No requested metric columns found in the DataFrame.")
    else:
        if len(available_metrics) < len(metrics_to_plot):
             missing = set(metrics_to_plot) - set(available_metrics)
             print(f"Warning: The following metrics are missing from the data: {list(missing)}")

        for metric in available_metrics:
            # Check if there's any valid data for this metric
            if df[metric].isnull().all():
                print(f"Warning: All values for metric '{metric}' are invalid (None/NaN). Skipping plot.")
                continue

            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(12, 7))

            # Plot one line per category
            for category in categories_to_plot:
                category_data = df[df['category'] == category].sort_values('model_stage_index')
                if not category_data.empty:
                    valid_data = category_data.dropna(subset=[metric])
                    if not valid_data.empty:
                        ax.plot(valid_data['model_stage_label'], valid_data[metric],
                                marker='o', linestyle='-', markersize=5,
                                label=category_labels.get(category, category))
                    else:
                        print(f"Info: No valid data points for metric '{metric}' in category '{category}'. Not plotted.")

            # Add English titles and labels
            ax.set_title(f'{metric} Score Across Continual Learning Stages', fontsize=16)
            ax.set_xlabel('Model Trained Up To Task', fontsize=12)
            ax.set_ylabel(f'{metric} Score', fontsize=12)

            # Ensure all X-axis ticks and labels are shown
            ax.set_xticks(range(len(model_stage_labels)))
            ax.set_xticklabels(model_stage_labels, rotation=15, ha="right", fontsize=10)

            ax.tick_params(axis='y', labelsize=10)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(title="Evaluation Category", fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))

            # Adjust Y-axis limits dynamically
            valid_metric_values = df[metric].dropna()
            if not valid_metric_values.empty:
                min_val = valid_metric_values.min()
                max_val = valid_metric_values.max()
                if pd.api.types.is_number(min_val) and pd.api.types.is_number(max_val):
                    padding = (max_val - min_val) * 0.1 if max_val > min_val else 0.1
                    y_lower = max(0, min_val - padding) if min_val >= 0 else min_val - padding
                    y_upper = max_val + padding
                    if y_lower < y_upper:
                         ax.set_ylim(y_lower, y_upper)
                    else:
                         ax.set_ylim(min_val - 0.05, min_val + 0.05)

            # Save the plot
            plot_filename = os.path.join(plot_save_dir, f'{metric}_by_task_trend.png')
            plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
            plt.savefig(plot_filename)
            print(f"Saved plot: {plot_filename}")
            plt.close(fig)

        print("\nAll plots generated and saved.")