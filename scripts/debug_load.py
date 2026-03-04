from diffusers import QwenImageEditPlusPipeline
import torch

pp = "/mmu-vcg/zhongweizhi/Pretrained_ckp/ckp_Qwen-Image-Edit-2509"
pipeline = QwenImageEditPlusPipeline.from_pretrained(pp, torch_dtype=torch.bfloat16)

dit_state = torch.load("/mmu-vcg/zb08/outputs/texture_edit/lora_merged_model_v2best/transformer/dit_merged.pth")
pipeline.transformer.load_state_dict(dit_state)
print("hh")