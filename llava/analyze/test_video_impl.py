import torch
from llava.constants import X_TOKEN_INDEX, DEFAULT_X_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_X_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers.models.llama.modeling_llama import SepEmb2LlamaAttention

# from llava.model.language_model.sepemb_boilerplate import SepEmb2LlamaAttention
# from torchview import draw_graph
import transformers
transformers.set_seed(42)

transformers.utils.logging.set_verbosity_error()

torch.set_warn_always(False)


def add_forward_hooks(model, cache, key_prefix=""):
    def get_llama_attn_maps(name):
        cache[key_prefix + name] = list()
        def hook(model, input, output):
            cache[key_prefix + name].append(output[1].detach())
        return hook

    def get_clip_attn_maps(name):
        cache[key_prefix + name] = list()
        def hook(model, input, output):            
            cache[key_prefix + name].append(output[1].detach())
        return hook
    
    all_hooks = list()
    
    # add hooks from llama LLM
    for block_idx in range(len(model.model.layers)):
        all_hooks.append(model.model.layers[block_idx].self_attn.register_forward_hook(
                get_llama_attn_maps(f"llama_attn_{block_idx}")
            )
        )   
    
    # add hooks from image tower
    for block_idx in range(len(model.model.video_tower.video_tower.encoder.layers)):
        all_hooks.append(model.model.video_tower.video_tower.encoder.layers[block_idx].self_attn.register_forward_hook(
                get_clip_attn_maps(f"clip_attn_{block_idx}")
            )
        )
    
    return model, all_hooks


def main():
    disable_torch_init()
    video = '/home/akane38/Video-LLaVA/llava/serve/examples/sample_demo_1.mp4'
    inp = 'Why is this video funny?'
    model_path = 'LanguageBind/Video-LLaVA-7B'
    device = 'cuda:7'
    load_4bit, load_8bit = True, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device)

    hook_cache = dict()
    model, all_hooks = add_forward_hooks(model, hook_cache)

    video_processor = processor['video']
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    
    video_tensor = video_processor(video, return_tensors='pt')['pixel_values']
    if type(video_tensor) is list:
        tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
    else:
        tensor = video_tensor.to(model.device, dtype=torch.float16)
    key = ['video']

    print(f"{roles[1]}: {inp}")
    inp = DEFAULT_X_TOKEN['VIDEO'] + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    print(f"{prompt=}")
    
    input_ids = tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX['VIDEO'], return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[tensor, key],
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            output_attentions=True)

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    print(outputs)
    
    torch.save(hook_cache, f"./video_sepemb_debug.pt")
    torch.save(SepEmb2LlamaAttention.inputs_emb_modalities, f"./video_sepemb_debug_modalities.pt")
    for hook in all_hooks:
        hook.remove()

if __name__ == '__main__':
    
    main()