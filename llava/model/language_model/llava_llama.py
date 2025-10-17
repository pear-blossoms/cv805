#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union
from .retriever import Retriever

import torch
import torch.nn as nn


from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig
from .llama_continual import LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"
    def __init__(self, vocab_size=32000, hidden_size=4096, intermediate_size=11008, num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=None, hidden_act="silu", max_position_embeddings=2048, initializer_range=0.02, rms_norm_eps=0.000001, use_cache=True, pad_token_id=0, bos_token_id=1, eos_token_id=2, pretraining_tp=1, tie_word_embeddings=False, rope_scaling=None, rope_theta=10000.0, **kwargs):
        super().__init__(vocab_size=vocab_size, hidden_size=hidden_size, intermediate_size=intermediate_size, num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads, num_key_value_heads=num_key_value_heads, hidden_act=hidden_act, max_position_embeddings=max_position_embeddings, initializer_range=initializer_range, rms_norm_eps=rms_norm_eps, use_cache=use_cache, pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, pretraining_tp=pretraining_tp, tie_word_embeddings=tie_word_embeddings, rope_scaling=rope_scaling, rope_theta=rope_theta, **kwargs)
        self.model_name = '/vast/users/xiaodan/haokunlin/Continual_LLaVA/llava/output/llava-v1.5-7b'
        self.dataset_type = 'llava-med' # 'domain', 'capability', 'dataset'
        self.task = "" 
        self.preprare_retreival_version = 'firstq' # lastq, preqa, allqa
        self.retriever_state_dict = ''# False: know task id for training; True: not know task id for inference
        self.disable_task_id = True 

        self.pool_size = 32 
        self.weight_topk = 4 
        self.random_dropout = None
        self.low_rank = 8
        self.groups = 4
        self.similarity_type = 'cosine'
        self.pool_train_keys = True
        self.pool_train_weights = True

        self.dataset_type_map = {
            'domain': {
                'chartqa': [0, 8], 
                'docvqa': [8, 16], 
                'iconqa': [16, 24], 
                'medicalqa': [24, 32]
                },
            'newdomain': {
                'GeoChat_Instruct':[0, 8],
                "llava_med":[8, 16],
                "atom":[16, 24],
                "art":[24, 32],
                "astro":[32, 40],
                "agri":[40, 48],
                "chem":[48, 56],
                "climate":[56, 64]},
            'dataset': {
                'ScienceQA':[0, 8],
                "TextVQA":[8, 16],
                "ImageNet":[16, 24],
                "GQA":[24, 32],
                "VizWiz":[32, 40],
                "Grounding":[40, 48],
                "VQAv2":[48, 56],
                "OCRVQA":[56, 64]},
            'llava-med': {
                'CT':[0, 8],
                "CXR":[8, 16],
                "Histopathology":[16, 24],
                "MRI":[24, 32]}

        }
        self.task_pool_index_range = self.dataset_type_map[self.dataset_type]
        self.pool_size = 8*len(self.task_pool_index_range) # 8 is the pool size of each task
        config = LlamaConfig.from_pretrained(self.model_name)
        for k, v in config.__dict__.items():
            setattr(self, k, v)


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        # For Continual Learning Setting
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.retriever = Retriever(config)
        if config.retriever_state_dict:
            self.retriever.keys = nn.parameter.Parameter(torch.load(config.retriever_state_dict), requires_grad=False)
            print('retriever shape: ', self.retriever.keys.shape)

        # Initialize weights and apply final processing
        self.post_init()
        # config._attn_implementation = "flash_attention_2"

    def get_model(self):
        return self.model
    
    def prepare_raw_text_for_retrieval(self, raw_text, version='firstq'):
        # version: firstq, lastq, preqa, allqa
        if version == 'firstq': 
            res = [_item[0] for _item in raw_text]
        elif version == 'lastq':
            res = [_item[-2] for _item in raw_text]
        elif version == 'allqa':
            res = [" ".join(sub_raw_text) for sub_raw_text in raw_text]
        elif version == 'preqa':
            res = []
            for sublist in raw_text:
                combined_string = ' '.join(sublist[:-1])
                res.append(combined_string)
        return res

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        raw_text = None,
        imgpath = None,
        weight_offset = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        task = self.config.task
        pool_mask = None
        if self.training:
            pool_mask = self.get_task_mask(task)
        elif not self.config.disable_task_id:
            pool_mask = self.get_task_mask(task)

        if weight_offset is None:
            raw_text = self.prepare_raw_text_for_retrieval(raw_text, self.config.preprare_retreival_version)
            retriever_outputs = self.retriever(
                inputs=raw_text,
                imgpath = imgpath,
                pool_mask=pool_mask,
            )
            weight_offset = retriever_outputs['weight_offset'].to(torch.bfloat16)

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            weight_offset=weight_offset
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        raw_text: Optional[torch.Tensor] = None,
        task: Optional[torch.StringType] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        
        
        task = None # FIXME Support inference with or without given task
        
        pool_mask = None
        if task is not None and not self.config.disable_task_id:
            pool_mask = self.get_task_mask(task)
            print('pool_mask:', pool_mask)
        retriever_outputs = self.retriever(
            inputs=raw_text,
            pool_mask=pool_mask,
        )
        weight_offset = retriever_outputs['weight_offset']

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            weight_offset=weight_offset,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        weight_offset = kwargs.pop("weight_offset", None)
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, weight_offset=weight_offset, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        if weight_offset is not None:
            inputs['weight_offset'] = weight_offset
        return inputs
    
    # If know the task id when inference. Not use when in the inference withou task_id setting
    def get_task_mask(self, task):
        assert not self.config.disable_task_id, "Can not use task_id"
        mask = torch.zeros(self.config.pool_size, dtype=torch.int, device=self.retriever.keys.device)
        l_idx, r_idx = self.config.task_pool_index_range[task]
        mask[l_idx: r_idx] = 1
        return mask

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
