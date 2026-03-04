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
# from flow_grpo.fsdp2_utils import prepare_fsdp_model
# from ml_collections import config_flags
# from torch.cuda.amp import GradScaler, autocast as torch_autocast
# import time

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
# config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

class PromptImageDataset(Dataset):
    def __init__(self, dataset, resolution=1024, split="train"):
        self.dataset = dataset
        self.resolution = resolution
        self.metadatas = []
        self.prompts = []
        self.file_path = os.path.join(dataset, f"{split}.jsonl")
        # with open(self.file_path, "r", encoding="utf-8") as f:
        #     self.metadatas = [json.loads(line) for line in f]
        #     self.prompts = [item["prompt"] for item in self.metadatas]
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                prompt = item.get("prompt", None)
                if not isinstance(prompt, str) or len(prompt.strip()) == 0:
                    print("Bad sample found, skipping:", item)
                    continue  # 🚫 丢掉坏样本
                self.metadatas.append(item)
                self.prompts.append(prompt)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        item = {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}
        # Assuming 'image' in metadata contains a path to the image file
        image_path = self.metadatas[idx]["edit_image"]
        item["prompt_with_image_path"] = f"{self.prompts[idx]}_{image_path}"
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        if w != h:
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            right = left + min_dim
            bottom = top + min_dim
            image = image.crop((left, top, right, bottom))
        image = image.resize(
            (self.resolution, self.resolution), Image.Resampling.LANCZOS
        )
        item["image"] = image
        return item

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        images = [example["image"] for example in examples]
        prompt_with_image_paths = [
            example["prompt_with_image_path"] for example in examples
        ]
        return prompts, metadatas, images, prompt_with_image_paths

if __name__ == "__main__":
    ddd = PromptImageDataset("/mmu-vcg/zb08/outputs/miccai/data/mayo")
    for i, item in enumerate(ddd):
        print(f"Sample {i}: {item['prompt']}")
        if i >= 5:  # 只打印前5个样本
            break
