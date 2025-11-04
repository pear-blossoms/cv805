# 文件: llava/eval/eval_video_qa.py
# (已修改为按 domain 分类计算指标)

import json
import argparse
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

def load_jsonl(filename):
    """Loads a JSON Lines file into a list of dictionaries."""
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping invalid line in {filename}: {line.strip()} - Error: {e}")
    return data

def calculate_metrics(gts, res):
    """
    Calculates BLEU, METEOR, ROUGE-L, and CIDEr scores.
    Args:
        gts (dict): Ground truth dictionary {question_id: [answer_string]}
        res (dict): Results dictionary {question_id: [answer_string]}
    Returns:
        dict: Dictionary containing calculated scores.
    """
    # Set up scorers
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    final_scores = {}
    for scorer, method in scorers:
        # print(f'Computing {method} score...') # 减少重复打印
        # pycocoevalcap expects dict of lists for both gts and res
        # Ensure res format matches gts format (list of strings for answer)
        score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                final_scores[m] = sc
                # print(f"{m}: {sc:.4f}")
        else:
            final_scores[method] = score
            # print(f"{method}: {score:.4f}")

    # 打印分数
    for m, s in final_scores.items():
        print(f"{m}: {s:.4f}")

    return final_scores

def main(args):
    print("Loading ground truth data...")
    gt_data = load_jsonl(args.gt_file)
    print(f"Loaded {len(gt_data)} ground truth items.")

    print("Loading prediction data...")
    pred_data = load_jsonl(args.pred_path)
    print(f"Loaded {len(pred_data)} prediction items.")

    if not gt_data or not pred_data:
        print("Error: Ground truth or prediction data is empty.")
        return

    # --- Format data for pycocoevalcap ---
    # Ground Truths (gts): Dictionary {question_id: [list_of_reference_answers]}
    gts = {}
    # 存储 QID 到 domain 的映射
    qid_to_domain = {}
    
    for item in gt_data:
        qid = item.get('question_id')
        answer = item.get('gpt4_answer') # Assuming ground truth is in 'gpt4_answer'
        # 获取 domain 信息
        domain_info = item.get('domain')

        if qid is not None and answer is not None:
            # pycocoevalcap expects a list of reference strings
            gts[str(qid)] = [str(answer)] # Convert qid to string key, wrap answer in list
        else:
             print(f"Warning: Missing 'question_id' or 'gpt4_answer' in GT item: {item}")
             continue # 如果 qid 或 answer 为空，跳过此项

        # 解析并存储 domain
        if domain_info and isinstance(domain_info, dict):
            found_domain = None
            for domain_name, is_active in domain_info.items():
                if is_active:
                    found_domain = domain_name
                    break # 假设每个 qid 只有一个 Ture 的 domain
            if found_domain:
                qid_to_domain[str(qid)] = found_domain
            else:
                print(f"Warning: No active domain found for qid {qid} in {domain_info}")
        else:
            print(f"Warning: Missing or invalid 'domain' field for qid {qid}")


    # Results (res): Dictionary {question_id: [single_generated_answer]}
    res = {}
    # 自动检测 answer_key 的逻辑
    answer_key = 'text' # Default guess
    if pred_data:
        first_pred_keys = pred_data[0].keys()
        if 'text' in first_pred_keys:
            answer_key = 'text'
        elif 'answer' in first_pred_keys:
            answer_key = 'answer'
        elif 'pred' in first_pred_keys:
             answer_key = 'pred'
        elif 'pred_response' in first_pred_keys:
             answer_key = 'pred_response'
        else:
            print(f"Warning: Could not automatically determine answer key in prediction file. Using default '{answer_key}'. Keys found: {list(first_pred_keys)}")

    for item in pred_data:
        qid = item.get('question_id')
        answer = item.get(answer_key)
        if qid is not None and answer is not None:
            res[str(qid)] = [str(answer)]
        else:
             print(f"Warning: Missing 'question_id' or '{answer_key}' in prediction item: {item}")

    # Filter gts and res to only include common question_ids
    common_ids = set(gts.keys()) & set(res.keys())
    print(f"Found {len(common_ids)} common question IDs between GT and predictions.")

    if not common_ids:
        print("Error: No common question IDs found between ground truth and predictions. Cannot evaluate.")
        return

    filtered_gts = {qid: gts[qid] for qid in common_ids}
    filtered_res = {qid: res[qid] for qid in common_ids}

    # --- Tokenize (对所有数据一次性 Tokenize) ---
    print('Tokenizing...')
    tokenizer = PTBTokenizer() # (这仍然需要 Java)
    gts_formatted_for_tokenizer = {}
    for qid, answers in filtered_gts.items():
        gts_formatted_for_tokenizer[qid] = [{'caption': ans} for ans in answers]

    res_formatted_for_tokenizer = {}
    for qid, answers in filtered_res.items():
        res_formatted_for_tokenizer[qid] = [{'caption': ans} for ans in answers]

    # (这仍然需要 Java)
    gts_tokenized = tokenizer.tokenize(gts_formatted_for_tokenizer)
    res_tokenized = tokenizer.tokenize(res_formatted_for_tokenizer)

    # --- 按 Domain 分组 Tokenized 数据 ---
    print("Grouping tokenized results by domain...")
    domain_gts_tokenized = {}
    domain_res_tokenized = {}
    all_domains = set()

    for qid in common_ids:
        domain = qid_to_domain.get(qid) # 使用 .get 避免 qid 不在 map 中的错误
        if domain:
            all_domains.add(domain)
            if qid in gts_tokenized:
                domain_gts_tokenized.setdefault(domain, {})[qid] = gts_tokenized[qid]
            if qid in res_tokenized:
                domain_res_tokenized.setdefault(domain, {})[qid] = res_tokenized[qid]
        else:
            print(f"Warning: QID {qid} found in common_ids but has no domain info. It will only be in 'overall' metrics.")

    # --- Calculate Metrics (Overall + Per-Domain) ---
    
    all_scores = {} # 最终保存所有分数

    # 1. 计算 Overall 指标
    print(f"\n--- Computing Overall Metrics ({len(gts_tokenized)} samples) ---")
    overall_scores = calculate_metrics(gts_tokenized, res_tokenized)
    all_scores["overall"] = overall_scores

    # 2. 计算每个 Domain 的指标
    for domain in sorted(list(all_domains)):
        gts_domain = domain_gts_tokenized.get(domain, {})
        res_domain = domain_res_tokenized.get(domain, {})
        
        if not gts_domain or not res_domain:
            print(f"\nSkipping domain {domain}: No common data found after grouping.")
            continue
        
        print(f"\n--- Computing Metrics for Domain: {domain} ({len(gts_domain)} samples) ---")
        
        # 确保 gts 和 res 的 qid 匹配
        domain_common_ids = set(gts_domain.keys()) & set(res_domain.keys())
        gts_domain_final = {qid: gts_domain[qid] for qid in domain_common_ids}
        res_domain_final = {qid: res_domain[qid] for qid in domain_common_ids}
        
        if not gts_domain_final:
             print(f"Skipping domain {domain}: No overlapping QIDs found in domain split.")
             continue
             
        domain_scores = calculate_metrics(gts_domain_final, res_domain_final)
        all_scores[domain] = domain_scores

    # --- Save scores ---
    if args.output_json:
        print(f"\nSaving all scores (overall and per-domain) to {args.output_json}")
        with open(args.output_json, 'w') as f:
            json.dump(all_scores, f, indent=4) # 保存包含所有分数的嵌套字典

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VQA predictions using standard NLP metrics.")
    parser.add_argument('--pred-path', type=str, required=True,
                        help="Path to the merged prediction JSON Lines file (e.g., merge.jsonl).")
    parser.add_argument('--gt-file', type=str, required=True,
                        help="Path to the original evaluation JSON Lines file containing ground truth answers.")
    parser.add_argument('--output-json', type=str, default=None,
                        help="Optional path to save the calculated scores as a JSON file.")
    
    args = parser.parse_args()
    main(args)