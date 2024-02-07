# CLIP encodes the image once, hence each layer has one attention map
# LLaMA encodes the sentence multiple times, and with only queries for generation,
# hence the latter attention maps of the same layer
# have the shape [B, H, 1, KV_LEN]
# Hence we create once attention matrix for one layer by concating everything
# `rollout(...)` copied over from 
# https://github.com/sayakpaul/vit-explain/blob/4f92628ed4b5109f43febd2976f688e585baa44b/vit_rollout.py
# Thanks @sayakpaul!

import torch
import numpy as np
from torchvision.utils import save_image
import cv2
from torch.nn import functional as F



def calculate_modality_indices(inputs_emb_modalities, bsz=1, seq_len=None):
    # bsz, num_heads, q_len, kv_len = attn_weights.size()
    
    if seq_len == None:
        raise ValueError("`seq_len` should be a positive integer, found None.")
    
    image_attn_mask = torch.zeros(bsz, seq_len)
    video_attn_mask = torch.zeros(bsz, seq_len)
    text_attn_mask = torch.zeros(bsz, seq_len)
    
    mask_map = dict(
        text=text_attn_mask,
        image=image_attn_mask,
        video=video_attn_mask
    )
    
    modalities_buffer = inputs_emb_modalities
    # List[List[Dict['modality': num_tokens]]]
    
    for example_idx in range(len(modalities_buffer)):
        example_buffer = modalities_buffer[example_idx]
        running_tok_idx = 0
        for chunk_idx in range(len(example_buffer)):
            chunk_modality = list(example_buffer[chunk_idx].keys())[0]
            chunk_tokens = list(example_buffer[chunk_idx].values())[0]
            mask_map[chunk_modality][example_idx, running_tok_idx : running_tok_idx + chunk_tokens] = 1
            running_tok_idx += chunk_tokens
    
    
    return image_attn_mask, video_attn_mask, text_attn_mask

def combine_attention(layer_attn_list):
    final_attn = list()
    max_len = layer_attn_list[-1].shape[-1]
    print(max_len)
    
    for attn in layer_attn_list:
        curr_kv_len = attn.shape[-1]
        final_attn.append(F.pad(attn, (0, max_len - curr_kv_len, 0, 0, 0, 0, 0, 0)))
    
    return torch.cat(final_attn, dim=-2)



def combine_all_layers(attns):
    for key in attns.keys():
        if key.startswith("llama"):
            attns[key] = combine_attention(attns[key])
    return attns


def rollout(attentions, discard_ratio=0.8, head_fusion="min"):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)
    
    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0 , 1 :]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask    


if __name__ == "__main__":
    attns = torch.load("debug.pt")
    modalities = torch.load("debug_modalities.pt")
    combined_attn = combine_attention(attns["llama_attn_27"])
    img_mask, vid_mask, text_mask = calculate_modality_indices(modalities, seq_len=combined_attn.shape[-1])
    pooled_combined_attn = torch.mean(combined_attn, dim=1)
    # print(pooled_combined_attn.shape)
    pooled_combined_attn = pooled_combined_attn[..., img_mask[0].to(torch.bool), :][..., img_mask[0].to(torch.bool)]
    pooled_combined_attn = pooled_combined_attn.permute(1,2,0).cpu().numpy()
    
    print(pooled_combined_attn)
    # print(pooled_combined_attn.shape)
    #.permute(1,2,0).cpu().numpy()

    score_map = torch.tensor(cv2.applyColorMap(np.uint8(255 * pooled_combined_attn), cv2.COLORMAP_VIRIDIS) / 255.).to(torch.float32).permute(2, 0, 1)

    save_image(score_map, "pooled_combined_attn_layer_27.jpg")

    # for attn in attns["llama_attn_26"]:
    #     print(attn.shape)

    # for attn in attns["clip_attn_2"]:
    #     print(attn.shape)
    
    
    # print(len(attns["llama_attn_27"]))
    # print(len(attns["clip_attn_2"]))
    # raise ValueError()
