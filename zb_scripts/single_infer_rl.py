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

from flow_grpo.diffusers_patch.qwen_image_edit_pipeline_with_zb import (
    pipeline_with_logprob,
)

from flow_grpo.diffusers_patch.train_dreambooth_lora_flux import encode_prompt
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
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


tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def cleanup_distributed():
    dist.destroy_process_group()


class PromptImageDataset(Dataset):
    def __init__(self, dataset, resolution=1024, split="train"):
        self.dataset = dataset
        self.resolution = resolution
        self.file_path = os.path.join(dataset, f"{split}.jsonl")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item["prompt"] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        item = {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}
        # Assuming 'image' in metadata contains a path to the image file
        image_path = self.metadatas[idx]["edit_image"]
        item["prompt_with_image_path"] = f"{image_path}"
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

def eval_fn(
    dst_folder,
    pipeline,
    test_dataloader,
    config,
    device,
    global_step,
    executor,
    mixed_precision_dtype,
):

    pipeline.transformer.eval()
    all_rewards = defaultdict(list)

    eval_loader = DataLoader(
        test_dataloader.dataset,
        batch_size=config.sample.test_batch_size,  # This is per-GPU batch size
        collate_fn=test_dataloader.collate_fn,
        num_workers=test_dataloader.num_workers,
    )

    all_ref_list = []
    all_out_list = []
    all_prompts = []
    all_prompt_with_image_paths = []

    for test_batch in tqdm(
        eval_loader,
        desc="Eval: ",
        position=0,
    ):


        prompts, prompt_metadata, ref_images, prompt_with_image_paths = test_batch

        with torch_autocast(
            enabled=(config.mixed_precision in ["fp16", "bf16"]),
            dtype=mixed_precision_dtype,
        ):

            # for debug_iter in range(4):
            with torch.no_grad():
                images, _, _, _, _, _, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt=prompts,
                    negative_prompt=[""] * len(prompts),
                    image=ref_images,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=1.0,
                    true_cfg_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    noise_level=config.sample.noise_level,
                    deterministic=True,
                    solver="dpm2",
                    # solver="flow",
                    max_area=config.resolution**2
                )

            all_ref_list.extend(ref_images)
            all_out_list.extend(images.cpu())
            all_prompts.extend(prompts)
            all_prompt_with_image_paths.extend(prompt_with_image_paths)

            # if len(all_prompts) > 10:
            #     break

    # if is_main_process(rank):


    # dst_
    orig_f = os.path.join(dst_folder, 'orig')
    os.makedirs(orig_f, exist_ok=True)
    edited_f =  os.path.join(dst_folder, 'edited')
    os.makedirs(edited_f, exist_ok=True)

    #ref_images 是pil格式，images是tensor格式

    for idx, item in enumerate(zip(all_prompts, all_ref_list, all_out_list)) :
        prompt, ref_image, edited_image = item

        img_name = os.path.basename(all_prompt_with_image_paths[idx])

        prompt_short = prompt[:80].replace(' ', '_').replace('/', '_')
        ref_image.save(os.path.join(orig_f, img_name))
        edited_img_pil = Image.fromarray(
            (edited_image.cpu().float().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        )
        edited_img_pil.save(os.path.join(edited_f, img_name))
        


def main(_):
    config = FLAGS.config

    local_rank = 0

    device = torch.device(f"cuda:{local_rank}")

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id


    # --- Load pipeline and models ---
    pipeline = QwenImageEditPlusPipeline.from_pretrained(config.pretrained.model, torch_dtype=torch.bfloat16)

    ### load sft lora base !!
    print("Loading LoRA SFT model ")
    dit_state = torch.load("/mmu-vcg/zb08/outputs/texture_edit/lora_merged_model_v2best/transformer/dit_merged.pth")
    pipeline.transformer.load_state_dict(dit_state)

    # from peft import PeftModel
    print("Loading LoRA RL model " + "**"*20)
    lora_path = '/mmu-vcg/zb08/codes/UniWorld-main/logs/nft/qwen_image_edit/zb_sft_base_bigbs_Coco_normal_kl_allwash/checkpoints+unique_id/checkpoint-92/lora'

    new_transformer = PeftModel.from_pretrained(
        pipeline.transformer,
        lora_path,          # e.g. checkpoint-xxx/lora
        is_trainable=False,  # 继续训练就 True；只推理 False
    )
    print("success Loading LoRA RL model " + "**"*20)

    pipeline.transformer = new_transformer

    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.transformer.requires_grad_(False)

    print("Loading eval dataset")

    test_dataset = PromptImageDataset(config.dataset, config.resolution, "test")

    banned_prompts_list = []

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,  # Per-GPU
        collate_fn=test_dataset.collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    
    executor = futures.ThreadPoolExecutor(max_workers=1)  # Async reward computation


    # --- Mixed Precision Setup ---
    mixed_precision_dtype = None
    if config.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16

    pipeline.to(device, dtype=mixed_precision_dtype)  # VAE usually fp32
    pipeline.vae.to(device, dtype=torch.float32)  # VAE usually fp32


    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    logger.info("***** Running training *****")

    pipeline.transformer.eval()
    print("Start eval before training")
    global_step = 0


    dst_folder = os.path.join('/mmu-vcg/zb08/outputs/texture_edit/sft_base', unique_id+f'_{global_step}_eval')
    os.makedirs('/mmu-vcg/zb08/outputs/texture_edit/sft_base', exist_ok=True)
    os.makedirs(dst_folder, exist_ok=True)

    eval_fn(
    dst_folder,
    pipeline,
    test_dataloader,
    config,
    device,
    global_step,
    executor,
    mixed_precision_dtype)




if __name__ == "__main__":
    app.run(main)
