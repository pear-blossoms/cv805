import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Any, Mapping, Union, List, Sequence
from transformers.tokenization_utils_base import BatchEncoding
from einops import rearrange
import math
import warnings
import json
from openai import OpenAI
import os
from copy import deepcopy
from filelock import FileLock
import numpy as np
from llava.model.language_model.rag_function import extract_entities_and_fetch_wiki
# from models.config import LlamaCLConfig


# class Config(object):
#     def __init__(self,
#                  pool_size=8,
#                  random_dropout = 0.25,
#                  weight_topk=2,
#                  groups = 6,
#                  hidden_size=4096,
#                  num_hidden_layers = 32,
#                  similarity_type = 'cosine',
#                  pool_train_keys = True,
#                  pool_train_weights = False) -> None:
#         self.pool_size =pool_size
#         self.random_dropout = random_dropout
#         self.weight_topk = weight_topk
#         self.groups = groups
#         self.hidden_size = hidden_size
#         self.num_hidden_layers = num_hidden_layers
#         self.similarity_type = similarity_type
#         self.pool_train_keys = pool_train_keys
#         self.pool_train_weights = pool_train_weights


def generate_orthogonal_matrix(rows, cols):
    tensor = torch.empty(rows, cols)
    indexes = list(range(0, rows, cols))
    if cols not in indexes:
        indexes.append(cols)
    for i in range(len(indexes) - 1):
        # import ipdb; ipdb.set_trace()
        # nn.init.orthogonal_(tensor[indexes[i]: indexes[i+1], :])
        tensor = tensor.to(torch.float32)
        nn.init.orthogonal_(tensor[indexes[i]: indexes[i+1], :])
        tensor = tensor.to(torch.bfloat16)
    return tensor

def find_most_similar_tensors_with_mask(query_tensor, tensor_list, mask=None, k=1):
    """
    找到与查询 tensor 余弦距离最近的 k 个 tensor 及其索引，仅考虑 mask 为 1 的索引
    
    参数:
    query_tensor (torch.Tensor): 查询的 768 维 tensor
    tensor_list (list of torch.Tensor): 待查询的 768 维 tensor 列表
    mask (torch.Tensor): 掩码张量，值为 1 的索引才会被考虑
    k (int): 返回的最近 tensor 的数量，默认为 1
    
    返回:
    best_indices (list of int): 前 k 个最相近 tensor 的索引列表
    best_similarities (list of float): 对应的余弦相似度列表
    """
    similarities = []

    for t in tensor_list:
        similarity = F.cosine_similarity(query_tensor, t, dim=0)
        similarities.append(similarity.item())
    
    similarities_tensor = torch.tensor(similarities, device=query_tensor.device)

    if mask is not None:
        similarities_tensor[mask == 0] = -100.0  

    best_indices = torch.topk(similarities_tensor, k=k, largest=True).indices.tolist()  
    best_similarities = similarities_tensor[best_indices].tolist() 
    
    return best_indices, best_similarities

class Retriever(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        assert config.similarity_type in ['cosine', 'softmax'], "The similarity calculation should be ['cosine', softmax]"
        self.similarity_type = config.similarity_type
        self.pool_size = config.pool_size
        self.weight_topk = config.weight_topk 
        self.hidden_size = config.hidden_size
        self.groups = config.groups
        self.random_dropout = config.random_dropout

        self.previous_weights = nn.ParameterList()  # Use nn.ParameterList() to store the parameters
        
        # Initialize weights for each task
        for i in range(1, 8):
            num_weights = 4 * (i)  # First task uses 4 weights, second uses 8, etc.
            weights = torch.normal(mean=1/(4*i), std=0, size=(num_weights,), device='cuda', requires_grad=True)
            self.previous_weights.append(nn.Parameter(weights))
        self.current_weight = nn.Parameter(
            torch.ones(16, device='cuda'), 
            requires_grad=True
        )

        self.pool_train_keys = config.pool_train_keys
        self.pool_train_weights = config.pool_train_weights
        self.num_hidden_layers = config.num_hidden_layers
        self.low_rank = config.low_rank

        if self.random_dropout is not None:
            assert self.random_dropout < 1 and self.random_dropout >=0, "random_dropout should be in [0, 1)"        

        # self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/nli-roberta-base-v2')
        # self.bert = AutoModel.from_pretrained('sentence-transformers/nli-roberta-base-v2')
        self.tokenizer = AutoTokenizer.from_pretrained('/vast/users/xiaodan/haokunlin/Continual_LLaVA/llava/output/nli-roberta-base-v2')
        self.bert = AutoModel.from_pretrained('/vast/users/xiaodan/haokunlin/Continual_LLaVA/llava/output/nli-roberta-base-v2')
        
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # self.model_hidden_size = self.bert.get_input_embeddings().weight.shape[-1]
        self.model_hidden_size = self.bert.get_input_embeddings().embedding_dim # 768
        assert self.model_hidden_size % self.groups == 0, "The vectordb hidden size: {} can not be divided envenly by the groups:{}".format(self.model_hidden_size, self.groups)
        self.key_hidden_size = self.model_hidden_size // self.groups

        # # weight offset 
        # low_rank_a = torch.zeros(self.pool_size, self.hidden_size * self.config.low_rank * self.num_hidden_layers)
        # low_rank_b = nn.init.normal_(torch.empty(self.pool_size,  self.hidden_size * self.config.low_rank * self.num_hidden_layers))
        # self.weight_offset = nn.parameter.Parameter(torch.stack([low_rank_a, low_rank_b], dim=-2), requires_grad=True)  # [pool_size, 2, channels*l*r]
        # self.retrieve_lora = nn.Parameter(torch.randn(2, 524288, device='cuda'), requires_grad=True)
        
        LORA_TOTAL_PARAMS = self.hidden_size * self.low_rank * self.num_hidden_layers
        low_rank_a = torch.zeros(self.pool_size, LORA_TOTAL_PARAMS)
        low_rank_b = nn.init.normal_(torch.empty(self.pool_size, LORA_TOTAL_PARAMS))
        # This will be [pool_size, 2, LORA_TOTAL_PARAMS]
        self.weight_offset_components = nn.parameter.Parameter(torch.stack([low_rank_a, low_rank_b], dim=-2), requires_grad=True)
        
        # RAG-based LoRA also needs to have the same component size
        self.retrieve_lora = nn.Parameter(torch.randn(2, LORA_TOTAL_PARAMS, device='cuda'), requires_grad=True)
     
        
        if self.pool_size > self.key_hidden_size:
            warnings.warn("The pool size is larger than the key_hidden_size, may cause the generate unstable keys")
        keys = [generate_orthogonal_matrix(self.pool_size, self.key_hidden_size) for _ in range(self.groups)]
        keys = torch.stack(keys, dim=0).unsqueeze(0)  # [1, groups, pool_size, key_hidden_size] [1, 6, 16, 128]
        self.keys = nn.parameter.Parameter(keys)

        self.last_keys = None  # Used for the centrifugal loss calculation

        if not config.pool_train_keys:
            self.freeze_keys()
        if not config.pool_train_weights:
            self.freeze_weights()

        self.embed_model = "text-embedding-ada-002"
        self.hyper_r = 64
        self.text_emb_dim = 1536
        self.hyper_U = nn.Sequential(
            nn.Linear(self.text_emb_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 2 * self.hyper_r),
        )

        self.hyper_V = nn.Parameter(torch.randn(self.hyper_r, LORA_TOTAL_PARAMS, device='cuda'), requires_grad=True)
        self.attn_dim = 64
        self.cross_query_proj = nn.Linear(LORA_TOTAL_PARAMS, self.attn_dim, bias=False)
        self.cross_key_proj   = nn.Linear(LORA_TOTAL_PARAMS, self.attn_dim, bias=False)
        self.cross_value_proj = nn.Linear(LORA_TOTAL_PARAMS, self.attn_dim, bias=False)
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.attn_dim, num_heads=8, batch_first=True)
        self.cross_out_proj = nn.Linear(self.attn_dim, LORA_TOTAL_PARAMS, bias=False)

        API_KEY  = ""
        API_BASE = ""   # 你的 endpoint

        self.client = OpenAI(api_key=API_KEY, base_url=API_BASE)

    
    def forward(
            self, 
            inputs, 
            imgpath=None, 
            pool_mask=None,
            return_topk_index=False,
            use_distance_weight=True,
            training_keys_only=False,
        ):
        LORA_TOTAL_PARAMS = self.hidden_size * self.low_rank * self.num_hidden_layers
        if not training_keys_only:
            B = len(inputs)
        else:
            B = inputs['input_ids'].shape[0]

        # if not training_keys_only:
        #     B = len(inputs)
        #     assert len(imgpath) == B, "inputs and imgpath must have same length"
        #     texts = []
        #     for q, im in zip(inputs, imgpath):
        #         try:
        #             res = extract_entities_and_fetch_wiki(q, im, top_k_per_entity=2, lang="en", model="gpt-4o-mini")
        #             if isinstance(res, list):
        #                 text = res[0] if len(res) > 0 else ""
        #             else:
        #                 text = res or ""
        #         except Exception:
        #             text = ""
        #         print('test:', text)
        #         texts.append(text)

        #     client = getattr(self, "client", None) or OpenAI()
        #     batch_size = getattr(self, "_emb_batch_size", 32)
        #     results = []  # will hold list[list[float]] length == B
        #     for i0 in range(0, len(texts), batch_size):
        #         batch_texts = texts[i0:i0+batch_size]
        #         # simple retry loop
        #         attempt = 0
        #         while True:
        #             try:
        #                 resp = client.embeddings.create(model=self.embed_model, input=batch_texts)
        #                 for item in resp.data:
        #                     results.append(item.embedding)
        #                 break
        #             except Exception as e:
        #                 attempt += 1
        #                 if attempt > 3:
        #                     # for failed batch, fallback to zero vectors for each entry to keep alignment
        #                     zero = [0.0] * self.text_emb_dim
        #                     for _ in batch_texts:
        #                         results.append(zero)
        #                     break
        #     emb_list = results
        #     emb_tensor = torch.tensor(emb_list, dtype=next(self.hyper_U.parameters()).dtype, device=self.hyper_V.device)
        #     B = emb_tensor.size(0)                                # batch size
        #     coeffs = self.hyper_U(emb_tensor)                     # -> [B, 2*r]
        #     coeffs = coeffs.view(B, 2, self.hyper_r)              # -> [B, 2, r]

        #     delta_lora = torch.einsum("bkr, rj -> bkj", coeffs, self.hyper_V)  # -> [B, 2, 802816]

        #     Q = self.cross_query_proj(delta_lora)   # -> [B, 2, attn_dim]

        #     K_base = self.cross_key_proj(self.retrieve_lora)    # -> [2, attn_dim]
        #     V_base = self.cross_value_proj(self.retrieve_lora)  # -> [2, attn_dim]

        #     K = K_base.unsqueeze(0).expand(B, -1, -1)            # -> [B, 2, attn_dim]
        #     V = V_base.unsqueeze(0).expand(B, -1, -1)            # -> [B, 2, attn_dim]

        #     attn_out, _ = self.cross_attn(Q, K, V)               # -> [B, 2, attn_dim]

        #     updated_lora = self.cross_out_proj(attn_out)         # -> [B, 2, 802816]
        # else:
        #     # This block runs only for `train_prompt_key.py`
        #     # Create a zero tensor placeholder for updated_lora
        #     # to avoid running the RAG pipeline.
        #     updated_lora = torch.zeros(B, 2, LORA_TOTAL_PARAMS, device=self.retrieve_lora.device, dtype=self.retrieve_lora.dtype)
        updated_lora = torch.zeros(B, 2, LORA_TOTAL_PARAMS, device=self.retrieve_lora.device, dtype=self.retrieve_lora.dtype)
        
        queries = torch.randn((4, 768),device='cuda') # This is a history design related to selection based method but now disabled
        # queries = self.encode(inputs)

        eval_mode = False
        # save_path = 'query_dir/'+'conversation_eval' + '_queries.json'
        # append_tensor_to_json(queries, save_path)
        if eval_mode:
            tmp_path = 'query_dir/'+'task_8poolsize_mean_vector.json'
            with open(tmp_path, 'r') as f:
                import json
                mean_vector = json.load(f)
                mean_vector = torch.tensor(mean_vector, dtype=torch.float32, device=queries.device)
            self.keys = nn.parameter.Parameter(mean_vector)
            bsz = queries.shape[0]
            queries = queries.view(bsz, -1)
            idx_vote, dis_weihgt = find_most_similar_tensors_with_mask(queries.view(-1), self.keys, mask=pool_mask,k=self.weight_topk)
            dis_weihgt = torch.tensor(dis_weihgt, device=queries.device)
            idx_vote = torch.tensor(idx_vote, device=queries.device)
            dis_weihgt = dis_weihgt / (dis_weihgt.sum() + 1e-9)
            outputs = dict()
            idx = idx_vote.unsqueeze(0).unsqueeze(0).expand(bsz, self.groups, -1)
        else:
            bsz = queries.shape[0]
            queries = rearrange(queries, "b (g c) -> b g c", g=self.groups)
            keys = self.keys.repeat(bsz, 1, 1, 1)
            outputs = dict()

            if self.similarity_type == 'cosine':
                queries = queries.unsqueeze(2).repeat(1, 1, self.pool_size, 1)
                sim = F.cosine_similarity(queries, keys, dim=-1)  # [bsz, groups, pool_size]
                
            else:
                queries = queries.unsqueeze(2).repeat(1, 1, self.pool_size, 1)
                sim = F.cosine_similarity(queries, keys, dim=-1) / 0.1  # [bsz, groups, pool_size]
                sim = torch.softmax(sim, dim=-1)

            idx_sim = sim.clone().detach()
            if self.training and self.random_dropout is not None:
                idx = torch.rand_like(idx_sim) <= self.random_dropout
                idx_sim.masked_fill_(idx, -100.)
            if pool_mask is not None:
                idx_sim[:, :, pool_mask == 0] = -100.
            
            if not use_distance_weight:
                _, idx = idx_sim.topk(self.weight_topk, dim=-1)  # [bsz, group, topk]
                idx_vote = rearrange(idx, "b g k -> g (b k)")
                base = (torch.arange(0, self.groups, device=idx_vote.device) * self.pool_size).view(-1, 1)
                idx_vote = (base + idx_vote).flatten()
                bin_count = torch.bincount(idx_vote, minlength=self.pool_size*self.groups).view(self.groups, self.pool_size)
                idx_vote = torch.topk(bin_count, k=self.weight_topk)[1]  # [groups, topk]
            else:
                idx_sim = torch.mean(idx_sim, dim=[0,1])
                dis_weihgt, idx_vote = idx_sim.topk(self.weight_topk, dim=-1) # [topk]
                dis_weihgt = dis_weihgt / (dis_weihgt.sum() + 1e-9)
                idx = idx_vote.unsqueeze(0).unsqueeze(0).expand(bsz, self.groups, -1)

        # weight_offset = self.weight_offset[idx_vote]
#=============================================================================
        def generate_dict(group_size, num_groups):
            result = {}
            total_keys = group_size * num_groups  
            for key in range(total_keys):
                result[key] = key // group_size

            return result

        # weight_offset = self.weight_offset.clone()  # clone to avoid in-place operation
        selected_offset_components = self.weight_offset_components[idx_vote]
        weight_offset_clone = selected_offset_components.clone() 
        l1_norm = torch.tensor(0.0, device=queries.device) # Initialize with a zero tensor
        idx_map = generate_dict(8, 8)

        for i, item in enumerate(idx_vote.tolist()):
            idx_prompt = idx_map[item] - 1
            if idx_prompt < 0:
                continue 
            else:

                prompt_weights = []  
                weight_offsets = []  
                
                for n in range(1, idx_prompt + 2): 
                    former_wo = self.weight_offset_components[(n - 1) * 4:(n) * 4, :]
                    weight_offsets.append(former_wo) 
                prompt_weights.append(self.previous_weights[idx_prompt].squeeze())
                current_weightoffset = selected_offset_components[i]
                weight_offsets.append(current_weightoffset.unsqueeze(0))
                prompt_weights.append(torch.ones(1, device='cuda', requires_grad=True))

                prompt_weights = [pw.flatten() for pw in prompt_weights]
                prompt_weights_tensor = torch.cat(prompt_weights)
                prompt_weights_normalized = prompt_weights_tensor / prompt_weights_tensor.sum(dim=0, keepdim=True)
                weight_offsets_normalized = torch.cat(weight_offsets)
                weighted_offset = torch.zeros_like(selected_offset_components[0])
                for k in range(prompt_weights_normalized.shape[0]):
                    weighted_offset += prompt_weights_normalized[k] * weight_offsets_normalized[k]

                weight_offset_clone[i] = weighted_offset

                n = len(weight_offsets)
                weight_offsets_normalized = torch.cat(weight_offsets) 
                if n > 1:

                    avg_weight_offset = weight_offsets_normalized[:-1].mean(dim=0)
                    last_weight_offset = weight_offsets_normalized[-1]
                    l1_norm = torch.norm(avg_weight_offset - last_weight_offset, p=1)
                else:
                    l1_norm = torch.tensor(0.0, device=weight_offsets_normalized.device)
#=============================================================================

        # weight_offset = weight_offset[idx_vote] #[4, 2, 802816]
        selected_offset_components = weight_offset_clone        # Average the per-sample dynamic LoRA across the batch to get a single batch-level dynamic LoRA
        updated_lora_batch_avg = torch.mean(updated_lora, dim=0) # Shape becomes [2, 1048576]
        # Use broadcasting to add the batch-averaged dynamic LoRA to each of the 4 selected static LoRAs
        # [4, 2, 1048576] + [2, 1048576] -> [4, 2, 1048576]
        final_offset_components = 1 * selected_offset_components + 0.01 * updated_lora_batch_avg

        low_rank_a = final_offset_components[..., 0,:].view(self.weight_topk, self.num_hidden_layers, self.low_rank, self.hidden_size)
        low_rank_b = final_offset_components[..., 1,:].view(self.weight_topk, self.num_hidden_layers, self.low_rank, self.hidden_size)

        weight_offset = torch.einsum("n l r x, n l r y -> n l x y", low_rank_a, low_rank_b)
        
        if not use_distance_weight:
            weight_offset = torch.mean(weight_offset, dim=0)
        else:
            weight_offset = (dis_weihgt[:, None, None, None] * weight_offset).sum(0)


        outputs['weight_offset'] = weight_offset

        if self.pool_train_keys:
            sim = torch.take_along_dim(sim, idx, dim=-1)
            loss = -sim.mean()
            outputs['key_loss'] = loss
            if self.last_keys is not None:
                outputs['centrifugal_loss'] = torch.einsum("b g x c, b g y c -> b g x y", F.normalize(self.keys, dim=-1), F.normalize(self.last_keys, dim=-1)).mean()

        if return_topk_index:
            outputs['topk_index'] = idx
        outputs['l1_norm'] = l1_norm
        self.l1_norm = l1_norm
        return outputs
    
    def encode(self, inps):
        with torch.no_grad():
            if isinstance(inps, str):
                inps = [inps]
            if isinstance(inps, Sequence):
                inps = self.tokenizer(inps, padding=True, truncation=True, return_tensors='pt')
            assert isinstance(inps, BatchEncoding), "The inputs of the encoder should be BatchEncoding."
            inps = inps.to(self.bert.device)
            embeddings = self.bert(**inps)
            embeddings = self.mean_pooling(embeddings, inps['attention_mask'])
        return embeddings
    
    def tokenize(self, sentences: Union[str, List[str]]):
        if isinstance(sentences, str):
            sentences = [sentences]
        return self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def freeze_keys(self):
        print("<=============== Freeze Keys =============>")
        self.keys.requires_grad = False
    
    def freeze_weights(self):
        print("<=============== Freeze weights =============>")
        self.vectordb.requires_grad = False
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def set_keys(self, values):
        assert values.shape == torch.Size([1, self.groups, self.pool_size, self.key_hidden_size]), "The shape of values: {} don't equal to {}".format(values.shape, [1, self.groups, self.pool_size, self.key_hidden_size])
        self.keys = nn.parameter.Parameter(values)
        if not self.pool_train_keys:
            self.freeze_keys()

    def set_last_keys(self, values):
        assert values.size(-1) == self.key_hidden_size and values.size(1) == self.groups, "The last keys shape: {} doesn't fit the retriever shape: {}".format(values.shape, self.keys.shape)
        self.last_keys = values