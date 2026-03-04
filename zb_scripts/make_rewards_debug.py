import os
import json

import sys
sys.path.append("/mmu-vcg/zb08/codes/UniWorld-main/UniWorld-V2")
sys.path.append("/mmu-vcg/zb08/codes/UniWorld-main/UniWorld-V2/flow_grpo")
from collections import defaultdict
import os
import datetime
from concurrent import futures
import time
from PIL import Image
import torch
import torchvision.transforms as transforms

import zb_reward_test_sam

from torch.utils.data import Dataset, DataLoader, Sampler

# local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device("cuda")

class PromptImageDataset(Dataset):
    def __init__(self, dataset, pred_img_p ,resolution=1024, split="train"):
        self.dataset = dataset
        self.resolution = resolution
        self.dst_imgp = pred_img_p
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
        item["prompt_with_image_path"] = f"{self.prompts[idx]}_{image_path}"
        image = Image.open(image_path).convert("RGB")
        pred_img_path = os.path.join(self.dst_imgp, os.path.basename(image_path))
        pred_img = Image.open(pred_img_path).convert("RGB")
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
        item['pred_image'] = pred_img
        return item

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        images = [example["image"] for example in examples]
        pred_images = [example["pred_image"] for example in examples]
        prompt_with_image_paths = [
            example["prompt_with_image_path"] for example in examples
        ]
        return prompts, metadatas, images, pred_images, prompt_with_image_paths

cccc = {"sam_iou": 1.0}
# cccc = {"sam_iou": 0.45, "zb_gemini": 0.4, "aesthetic": 0.15}
reward_fn = zb_reward_test_sam.multi_score(device, cccc)

# 其实感觉如果有一个 idx 就会好一些 

test_json_p = '/mmu-vcg/zb08/outputs/texture_edit/train_data/rl_data/maybe_v2_final_texture/test.jsonl'
# test_json =  
with open(test_json_p, 'r', encoding='utf-8') as f:
    test_json = f.readlines()
all_list = []
for item in test_json:
    data = json.loads(item)
    all_list.append(data)

transform = transforms.Compose(
    [
        transforms.ToTensor(),  # Convert to tensor
    ])

dst_img_folder_P = '/mmu-vcg/zb08/outputs/texture_edit/sft_base/2025.12.30_14_0_eval_sft_base_dpm18/edited'

test_dataset = PromptImageDataset("/mmu-vcg/zb08/outputs/texture_edit/train_data/rl_data/maybe_v2_final_texture",dst_img_folder_P, 1024, "test")

test_dataloader = DataLoader(
    test_dataset,
    batch_size=8,  # Per-GPU
    collate_fn=test_dataset.collate_fn,
    num_workers=4,
    pin_memory=True,
)

for batch in test_dataloader:
    prompts, prompt_metadata, images, pred_images, prompt_with_image_paths = batch
    # ref_images = []

    pred_images = torch.stack([transform(image) for image in pred_images])

    tt = reward_fn( pred_images, prompts, prompt_metadata, images, only_strict=True)

    # executor = futures.ThreadPoolExecutor(max_workers=1)  # Async reward computation

    # # 需要画出来mask图 
    # rewards_future = executor.submit(
    #     reward_fn,
    #     pred_images,
    #     prompts,
    #     prompt_metadata,
    #     images,
    #     only_strict=True,
    # )
    print("hh")

    # tt = rewards_future.result()
    print("jj")

# 对着 all_list 用之前的reward 评价函数 跑一遍结果就行了 