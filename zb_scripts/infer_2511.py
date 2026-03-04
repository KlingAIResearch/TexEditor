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


def get_local_device():
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return torch.device(f"cuda:{local_rank}"),local_rank
    return torch.device("cpu"),local_rank

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def setup_distributed(rank, lock_rank, world_size):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(lock_rank)


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def set_seed(seed: int, rank: int = 0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


class DistributedKRepeatSampler(Sampler):
    def __init__(
        self, dataset, batch_size, k, num_replicas, rank, seed=0, banned_prompts=None
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k  # k means the number of images per prompt
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.total_samples = self.num_replicas * self.batch_size
        assert (
            self.total_samples % self.k == 0
        ), f"k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

        self.banned_prompts = banned_prompts if banned_prompts is not None else set()
        self.last_banned_prompts_len = len(self.banned_prompts)
        self.valid_indices_cache = None

    def get_valid_indices(self):
        start_time = time.time()
        if self.valid_indices_cache is None or self.last_banned_prompts_len != len(self.banned_prompts):
            self.valid_indices_cache = [
                i for i, prompt in enumerate(self.dataset.prompts)
                if prompt not in self.banned_prompts
            ]
            self.last_banned_prompts_len = len(self.banned_prompts)
        print("get valid indices time: ", time.time() - start_time)
        return self.valid_indices_cache

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            valid_indices = self.get_valid_indices()
            if len(valid_indices) < self.m:
                # TODO
                raise NotImplementedError()

            indices = torch.tensor(valid_indices)[
                torch.randperm(len(valid_indices), generator=g)[: self.m]
            ].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            shuffled_indices = torch.randperm(
                len(repeated_indices), generator=g
            ).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def gather_tensor_to_all(tensor, world_size):
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0).cpu()


def eval_fn(
    dst_folder,
    pipeline,
    test_dataloader,
    config,
    device,
    rank,
    world_size,
    global_step,
    reward_fn,
    executor,
    mixed_precision_dtype,
):

    device , local_rank = get_local_device()
    print("eval device:", device, local_rank)

    # if config.train.ema and ema is not None:
    #     ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    pipeline.transformer.eval()
    all_rewards = defaultdict(list)

    test_sampler = (
        DistributedSampler(
            test_dataloader.dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        if world_size > 1
        else None
    )
    eval_loader = DataLoader(
        test_dataloader.dataset,
        batch_size=config.sample.test_batch_size,  # This is per-GPU batch size
        sampler=test_sampler,
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
        disable=not is_main_process(rank),
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
        

    if world_size > 1:
        dist.barrier()


def main(_):
    config = FLAGS.config

    # --- Distributed Setup ---
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    setup_distributed(rank, local_rank, world_size)
    device = torch.device(f"cuda:{local_rank}")

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # # --- WandB Init (only on main process) ---
    # if is_main_process(rank):
    #     log_dir = os.path.join(config.logdir, config.run_name)
    #     os.makedirs(log_dir, exist_ok=True)
    #     wandb.init(
    #         project="flow-grpo",
    #         name=config.run_name,
    #         config=config.to_dict(),
    #         dir=log_dir,
    #     )
    logger.info(f"\n{config}")

    set_seed(config.seed, rank)  # Pass rank for different seeds per process

    # --- Mixed Precision Setup ---
    mixed_precision_dtype = None
    if config.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16

    enable_amp = mixed_precision_dtype is not None
    scaler = GradScaler(enabled=enable_amp)

    # --- Load pipeline and models ---
    model_pppp = '/mmu-vcg/zb08/CKPTS/qwen-edit_2511'
    pipeline = QwenImageEditPlusPipeline.from_pretrained(model_pppp, torch_dtype=torch.bfloat16)
    #     pipeline = QwenImageEditPlusPipeline.from_pretrained(
    #     MODEL_PATH, torch_dtype=torch.bfloat16
    # ).to(f"cuda:{gpu_id}")


    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.transformer.requires_grad_(True)
    # pipeline.transformer.requires_grad_(not config.use_lora)
    tokenizers = [pipeline.tokenizer]
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not is_main_process(rank),
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32

    pipeline.vae.to(device, dtype=torch.float32)  # VAE usually fp32

    #可弄可不弄 
    prepare_fsdp_model(
        pipeline.text_encoder,
        shard_conditions=[lambda n, m: isinstance(m, (Qwen2_5_VLVisionBlock, Qwen2_5_VLDecoderLayer))],
        cpu_offload=False,
        weight_dtype=text_encoder_dtype,
    )

    transformer = pipeline.transformer.to(device)


    transformer_ddp = DDP(
        transformer,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )


    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Datasets and Dataloaders ---
    # train_dataset = PromptImageDataset(
    #     config.dataset, config.resolution, "train"
    # )
    test_dataset = PromptImageDataset(config.dataset, config.resolution, "test")
    # print("Test dataset size:", len(test_dataset))


    # ban_str_p = '/mmu-vcg/zb08/outputs/texture_edit/aaa_utils/banned_prompts.json'
    # with open(ban_str_p, 'r', encoding='utf-8') as f:
    #     banned_prompts_list = json.load(f)
    banned_prompts_list = []

    test_sampler = (
        DistributedSampler(
            test_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        if world_size > 1
        else None
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,  # Per-GPU
        sampler=test_sampler,  # Use distributed sampler for eval
        collate_fn=test_dataset.collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    # --- Prompt Trackering ---
    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(
            config.sample.global_std,
            config.sample.ban_std_thres,
            config.sample.ban_mean_thres,
        )
    else:
        # assert False
        print("Warning: per-prompt stat tracking is disabled.")

    # add stat_tracker ban
    stat_tracker.banned_prompts.update(banned_prompts_list)
    ### 
    # print("Initial banned prompts len:", len(stat_tracker.banned_prompts))
    
    executor = futures.ThreadPoolExecutor(max_workers=1)  # Async reward computation

    # Train!
    samples_per_epoch = (
        config.sample.train_batch_size
        * world_size
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size * world_size * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")


    import zb_reward_test_sam

    # reward_fn = zb_reward_test_sam.multi_score(device, config.reward_fn)
    eval_reward_fn = zb_reward_test_sam.multi_score(device, config.reward_fn)

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
    rank,
    world_size,
    global_step,
    eval_reward_fn,
    executor,
    mixed_precision_dtype,)

    if is_main_process(rank):
        pass
        

    # wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    app.run(main)
