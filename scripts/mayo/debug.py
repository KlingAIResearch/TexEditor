# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
sys.path.append("/mmu-vcg/zb08/codes/UniWorld-main/UniWorld-V2")
sys.path.append("/mmu-vcg/zb08/codes/UniWorld-main/UniWorld-V2/flow_grpo")
from collections import defaultdict
import os
import datetime
from concurrent import futures
import time
import json
from absl import app, flags
import logging
from diffusers import QwenImageEditPlusPipeline
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionBlock, Qwen2_5_VLDecoderLayer
import numpy as np
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
# from flow_grpo.diffusers_patch.qwen_image_edit_pipeline_with_logprob import (
#     pipeline_with_logprob,
# )
from flow_grpo.diffusers_patch.qwen_image_edit_pipeline_with_zb import (
    pipeline_with_logprob,
)

from flow_grpo.diffusers_patch.train_dreambooth_lora_flux import encode_prompt
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
# import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper
from flow_grpo.fsdp2_utils import prepare_fsdp_model
from ml_collections import config_flags
from torch.cuda.amp import GradScaler, autocast as torch_autocast
import time
from diffusers import FluxKontextPipeline


from safetensors import safe_open
def load_state_dict(file_path, torch_dtype=None, device="cpu"):
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype, device=device)
    else:
        return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype, device=device)


def load_state_dict_from_safetensors(file_path, torch_dtype=None, device="cpu"):
    state_dict = {}
    with safe_open(file_path, framework="pt", device=str(device)) as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
            if torch_dtype is not None:
                state_dict[k] = state_dict[k].to(torch_dtype)
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None, device="cpu"):
    state_dict = torch.load(file_path, map_location=device, weights_only=True)
    if torch_dtype is not None:
        for i in state_dict:
            if isinstance(state_dict[i], torch.Tensor):
                state_dict[i] = state_dict[i].to(torch_dtype)
    return state_dict

if __name__ == "__main__":

    # --- Load pipeline and models ---
    ppp = "/mmu-vcg/zb08/models/black-forest-labs/FLUX.1-Kontext-dev"
    pipeline = FluxKontextPipeline.from_pretrained(ppp)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    ppp2 = '/mmu-vcg/zb08/outputs/mayo_flux_model/flux_merge_2_8/dit_merged.pth'
    temp2 = torch.load(ppp2)

    ppp3 = '/mmu-vcg/zb08/codes/DiffSynth-Studio-main/models/train/flux_mayo/epoch-4.safetensors'
    temp3 = load_state_dict(ppp3)


    print("jj")