import torch

# ScienceQA_prompt_key = torch.load('./output/prompt-key/GeoChat_Instruct_prompt_key.pth')['keys'] # [1, 4, 32, 192]
# TextVQA_prompt_key = torch.load('./output/prompt-key/llava_med_prompt_key.pth')['keys']
# ImageNet_prompt_key = torch.load('./output/prompt-key/atom_prompt_key.pth')['keys']
# GQA_prompt_key = torch.load('./output/prompt-key/art_prompt_key.pth')['keys']
# VizWiz_prompt_key = torch.load('./output/prompt-key/astro_prompt_key.pth')['keys'] # [1, 4, 32, 192]
# Grounding_prompt_key = torch.load('./output/prompt-key/agri_prompt_key.pth')['keys']
# VQAv2_prompt_key = torch.load('./output/prompt-key/chem_prompt_key.pth')['keys']
# OCRVQA_prompt_key = torch.load('./output/prompt-key/climate_prompt_key.pth')['keys']
# print('keys loaded')
# # 'weight_offset', 'keys'

# # prompt_key = torch.cat(
# #     (chartqa_prompt_key[:, :, :8, :], 
# #      docvqa_prompt_key[:, :, 8:16, :], 
# #      iconqa_prompt_key[:, :, 16:24, :], 
# #      medicalqa_prompt_key[:, :, 24:, :]), dim=1)

# prompt_key = torch.cat(
#     (ScienceQA_prompt_key[:, :1, :, :], 
#      TextVQA_prompt_key[:, 1:2, :, :], 
#      ImageNet_prompt_key[:, 2:3, :, :], 
#      GQA_prompt_key[:, 3:4, :, :], 
#      VizWiz_prompt_key[:, 4:5, :, :], 
#      Grounding_prompt_key[:, 5:6, :, :], 
#      VQAv2_prompt_key[:, 6:7, :, :], 
#      OCRVQA_prompt_key[:, 7:8, :, :]), 
#      dim=1)

# print("prompt_key.shape: {}".format(prompt_key.shape))

# torch.save(prompt_key, 'output/prompt-key/newdomain_prompt_key.pth')

# # [1, 1, 1, 1, 1, 1, 1, 1, 0, ....]

# # [:, :, :8, :]
# # [:, :, 8:16, :]
# # [:, :, 16:24, :]
# # [:, :, 24:, :]

CT_prompt_key = torch.load('/vast/users/xiaodan/haokunlin/Continual_LLaVA/llava/output/prompt-key/CT.pth')['keys'] # [1, 4, 32, 192]
CXR_prompt_key = torch.load('/vast/users/xiaodan/haokunlin/Continual_LLaVA/llava/output/prompt-key/CXR.pth')['keys']
Histopathology_prompt_key = torch.load('/vast/users/xiaodan/haokunlin/Continual_LLaVA/llava/output/prompt-key/Histopathology.pth')['keys']
MRI_prompt_key = torch.load('/vast/users/xiaodan/haokunlin/Continual_LLaVA/llava/output/prompt-key/MRI.pth')['keys']
print('keys loaded')

prompt_key = torch.cat(
    (
        CT_prompt_key[:, :1, :, :], 
        CXR_prompt_key[:, 1:2, :, :], 
        Histopathology_prompt_key[:, 2:3, :, :], 
        MRI_prompt_key[:, 3:4, :, :], 
    ), 
    dim=1
)

print("prompt_key.shape: {}".format(prompt_key.shape))

torch.save(prompt_key, '/vast/users/xiaodan/haokunlin/Continual_LLaVA/llava/output/prompt-key/merged_prompt_key_o.pth')

# CT_prompt_key = torch.load('/vast/users/xiaodan/haokunlin/Continual_LLaVA/llava/output/prompt-key/CT.pth')['keys'] # [1, 4, 32, 192]
# CXR_prompt_key = torch.load('/vast/users/xiaodan/haokunlin/Continual_LLaVA/llava/output/prompt-key/CXR.pth')['keys']
# Histopathology_prompt_key = torch.load('/vast/users/xiaodan/haokunlin/Continual_LLaVA/llava/output/prompt-key/Histopathology.pth')['keys']
# print('keys loaded')

# prompt_key = torch.cat(
#     (
#         CT_prompt_key[:, :1, :, :], 
#         CXR_prompt_key[:, 1:2, :, :], 
#         Histopathology_prompt_key[:, 2:3, :, :], 
#     ), 
#     dim=1
# )

# print("prompt_key.shape: {}".format(prompt_key.shape))

# torch.save(prompt_key, '/vast/users/xiaodan/haokunlin/Continual_LLaVA/llava/output/prompt-key/merged_prompt_key.pth')