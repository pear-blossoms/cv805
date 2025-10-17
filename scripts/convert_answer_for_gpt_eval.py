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

# jsonl_filepath = "./playground/data/eval/hfdata/iconqa_val.jsonl"
# json_filepath = "./playground/data/eval/hfdata/answers_upload/iconqa_val/Continual-LLaVA-rank1232-poolsize-8wtk-lrank12-chartqa-docvqa-iconqa-medicalqa.json"
# output_filepath = './playground/data/eval/gpt_eval/Continual-LLaVA-rank1232-poolsize-8wtk-lrank12-chartqa-docvqa-iconqa-medicalqa-iconqa_gt_pred_for_gpt_eval.jsonl'
# jsonl_filepath = "./playground/data/eval/docvqa/docvqa_val.jsonl"
# json_filepath = "./playground/data/eval/docvqa/answers_upload/docvqa_val/Continual-LLaVA-rank1232-poolsize-8wtk-lrank12-chartqa-docvqa-iconqa-medicalqa.json"
# output_filepath = './playground/data/eval/gpt_eval/Continual-LLaVA-rank1232-poolsize-8wtk-lrank12-chartqa-docvqa-iconqa-medicalqa-docvqa_gt_pred_for_gpt_eval.jsonl'
# jsonl_filepath = "./playground/data/eval/pathvqa/medicalqa_val.jsonl"
# json_filepath = "./playground/data/eval/pathvqa/answers_upload/medicalqa_val/Continual-LLaVA-rank1232-poolsize-8wtk-lrank12-chartqa-docvqa-iconqa-medicalqa.json"
# output_filepath = './playground/data/eval/gpt_eval/Continual-LLaVA-rank1232-poolsize-8wtk-lrank12-chartqa-docvqa-iconqa-medicalqa-medicalqa_gt_pred_for_gpt_eval.jsonl'
jsonl_filepath = "./playground/data/eval/chartqa/chartqa_val.jsonl"
json_filepath = "./playground/data/eval/chartqa/answers_upload/chartqa_val/Continual-LLaVA-rank1232-poolsize-8wtk-lrank12-chartqa-docvqa-iconqa-medicalqa.json"
output_filepath = './playground/data/eval/gpt_eval/Continual-LLaVA-rank1232-poolsize-8wtk-lrank12-chartqa-docvqa-iconqa-medicalqa-chartqa_gt_pred_for_gpt_eval.jsonl'
jsonl_data = read_jsonl_file(jsonl_filepath)
json_data = read_json_file(json_filepath)

output_data = []
for entry in jsonl_data:
    question_id = entry['question_id']
    new_entry = {
        'id': question_id,
        'question': entry['text'].rstrip('\n'),
        'answer': entry['answers'],
        'pred': json_data.get(question_id, None)
    }
    output_data.append(new_entry)

write_jsonl_file(output_filepath, output_data)