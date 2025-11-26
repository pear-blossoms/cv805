import torch
import os

base_path = 'bash_folder'
save_path = os.path.join(base_path, 'chest_abd_head.pth')

path1 = os.path.join(base_path, 'chest.pth')
path2 = os.path.join(base_path, 'abd.pth')
path3 = os.path.join(base_path, 'head.pth')
# path4 = os.path.join(base_path, 'CT.pth')

ckpt1 = torch.load(path1)
ckpt2 = torch.load(path2)
ckpt3 = torch.load(path3)
# ckpt4 = torch.load(path4)
print('all loaded')

merged_keys = torch.cat(
    (
        ckpt1['keys'][:, :, 0:8, :],   
        ckpt2['keys'][:, :, 8:16, :],   
        ckpt3['keys'][:, :, 16:24, :],
        # ckpt4['keys'][:, :, 24:32, :]   
    ), 
    dim=2
)
print(f"shape of keys after merging: {merged_keys.shape}, saved to {save_path}") 

# merged_weights = torch.cat(
#     (
#         ct_ckpt['weight_offset_components'][0:8, :, :],   
#         cxr_ckpt['weight_offset_components'][8:16, :, :],   
#         hist_ckpt['weight_offset_components'][16:24, :, :], 
#         mri_ckpt['weight_offset_components'][24:32, :, :] 
#     ), 
#     dim=0
# )
# print(f"shape of weights after merging: {merged_weights.shape}") 

# save_dir = os.path.dirname(save_path)
# if save_dir:
#     os.makedirs(save_dir, exist_ok=True)

torch.save({
    'keys': merged_keys,
    # 'weight_offset_components': merged_weights
}, save_path)

# print(f"saved: {save_path}")