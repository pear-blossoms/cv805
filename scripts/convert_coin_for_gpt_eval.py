import json

def read_jsonl_file(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def read_json_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return {entry['question_id']: entry['answer'] for entry in data}

def write_jsonl_file(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as file:
        for entry in data:
            file.write(json.dumps(entry, ensure_ascii=False) + '\n')

jsonl_filepath = "./playground/data/eval/GQA/GQA_val.jsonl"
json_filepath = "./playground/data/eval/GQA/answers_upload/GQA_val/llava-fulltune-coin-ScienceQA-TextVQA-ImageNet-GQA-VizWiz-Grounding-VQAv2-OCRVQA.json"
output_filepath = './playground/data/eval/gpt_eval/llava-fulltune-coin-ScienceQA-TextVQA-ImageNet-GQA-VizWiz-Grounding-VQAv2-OCRVQA-GQA_gt_pred_for_gpt_eval.jsonl'
jsonl_filepath = "./playground/data/eval/ScienceQA/ScienceQA_val.jsonl"
json_filepath = "./playground/data/eval/ScienceQA/answers_upload/ScienceQA_val/llava-fulltune-coin-ScienceQA-TextVQA-ImageNet-GQA-VizWiz-Grounding-VQAv2-OCRVQA.json"
output_filepath = './playground/data/eval/gpt_eval/llava-fulltune-coin-ScienceQA-TextVQA-ImageNet-GQA-VizWiz-Grounding-VQAv2-OCRVQA-ScienceQA_gt_pred_for_gpt_eval.jsonl'
jsonl_filepath = "./playground/data/eval/TextVQA/TextVQA_val.jsonl"
json_filepath = "./playground/data/eval/TextVQA/answers_upload/TextVQA_val/llava-fulltune-coin-ScienceQA-TextVQA-ImageNet-GQA-VizWiz-Grounding-VQAv2-OCRVQA.json"
output_filepath = './playground/data/eval/gpt_eval/llava-fulltune-coin-ScienceQA-TextVQA-ImageNet-GQA-VizWiz-Grounding-VQAv2-OCRVQA-TextVQA_gt_pred_for_gpt_eval.jsonl'
jsonl_filepath = "./playground/data/eval/VQAv2/VQAv2_val.jsonl"
json_filepath = "./playground/data/eval/VQAv2/answers_upload/VQAv2_val/llava-fulltune-coin-ScienceQA-TextVQA-ImageNet-GQA-VizWiz-Grounding-VQAv2-OCRVQA.json"
output_filepath = './playground/data/eval/gpt_eval/llava-fulltune-coin-ScienceQA-TextVQA-ImageNet-GQA-VizWiz-Grounding-VQAv2-OCRVQA-VQAv2_gt_pred_for_gpt_eval.jsonl'
jsonl_filepath = "./playground/data/eval/ImageNet/ImageNet_val.jsonl"
json_filepath = "./playground/data/eval/ImageNet/answers_upload/ImageNet_val/llava-fulltune-coin-ScienceQA-TextVQA-ImageNet-GQA-VizWiz-Grounding-VQAv2-OCRVQA.json"
output_filepath = './playground/data/eval/gpt_eval/llava-fulltune-coin-ScienceQA-TextVQA-ImageNet-GQA-VizWiz-Grounding-VQAv2-OCRVQA-ImageNet_gt_pred_for_gpt_eval.jsonl'
jsonl_filepath = "./playground/data/eval/VizWiz/VizWiz_val.jsonl"
json_filepath = "./playground/data/eval/VizWiz/answers_upload/VizWiz_val/llava-fulltune-coin-ScienceQA-TextVQA-ImageNet-GQA-VizWiz-Grounding-VQAv2-OCRVQA.json"
output_filepath = './playground/data/eval/gpt_eval/llava-fulltune-coin-ScienceQA-TextVQA-ImageNet-GQA-VizWiz-Grounding-VQAv2-OCRVQA-VizWiz_gt_pred_for_gpt_eval.jsonl'
jsonl_filepath = "./playground/data/eval/OCRVQA/OCRVQA_val.jsonl"
json_filepath = "./playground/data/eval/OCRVQA/answers_upload/OCRVQA_val/llava-fulltune-coin-ScienceQA-TextVQA-ImageNet-GQA-VizWiz-Grounding-VQAv2-OCRVQA.json"
output_filepath = './playground/data/eval/gpt_eval/llava-fulltune-coin-ScienceQA-TextVQA-ImageNet-GQA-VizWiz-Grounding-VQAv2-OCRVQA-OCRVQA_gt_pred_for_gpt_eval.jsonl'
jsonl_filepath = "./playground/data/eval/Grounding/Grounding_val.jsonl"
json_filepath = "./playground/data/eval/Grounding/answers_upload/Grounding_val/llava-fulltune-coin-ScienceQA-TextVQA-ImageNet-GQA-VizWiz-Grounding-VQAv2-OCRVQA.json"
output_filepath = './playground/data/eval/gpt_eval/llava-fulltune-coin-ScienceQA-TextVQA-ImageNet-GQA-VizWiz-Grounding-VQAv2-OCRVQA-Grounding_gt_pred_for_gpt_eval.jsonl'

jsonl_data = read_jsonl_file(jsonl_filepath)
json_data = read_json_file(json_filepath)

output_data = []
for entry in jsonl_data:
    question_id = entry['question_id']
    new_entry = {
        'id': question_id,
        'question': entry['text'].rstrip('\n'),
        'answer': entry['answer'],
        'pred': json_data.get(question_id, None)
    }
    output_data.append(new_entry)

write_jsonl_file(output_filepath, output_data)