import asyncio
import time
from typing import List

from skimage.metrics import structural_similarity as ssim


import numpy as np
import cv2
from PIL import Image
import torch
from pydantic import BaseModel
import uvicorn
import os
import base64
import io
import random

import sys
sys.path.append("/mmu-vcg/zb08/codes/UniWorld-main/UniWorld-V2/flow_grpo")
sys.path.append("/mmu-vcg/zb08/codes/UniWorld-main/UniWorld-V2")
sys.path.append("/mmu-vcg/zb08/codes/sam3-main")
sys.path.append("/mmu-vcg/zb08/codes/UniWorld-main/UniWorld-V2/segment_anything")

MODEL_NAME="model.sauge_vitb"
import importlib
Model = importlib.import_module(MODEL_NAME)

from torchvision import transforms


import random

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamAutomaticMaskGeneratorForTest

import base64
import io
from PIL import Image

import os

# 这里计划写一个多进程的 打分judge 脚本  
# 打分我计划 分成2 部分  这里的编辑的 指令跟随程度和美观度的综合打分 、 用线条和 mask 做一致性的打分  或者再看看之前的文章 


import re
from typing import Optional, Dict


import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from PIL import Image
import torch
import numpy as np

import json

from tqdm import tqdm
# from openai import OpenAI

import random

import argparse

from tqdm import tqdm

def get_local_device():
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return torch.device(f"cuda:{local_rank}"),local_rank
    return torch.device("cpu"),local_rank

def post_process_edge(multi_outputs, alpha=0.4):
    percent_1 = alpha / 0.5
    percent_2 = 1 - percent_1
    any_output = percent_1 * multi_outputs[1] + percent_2 * multi_outputs[0]
    any_output = torch.clamp(any_output, 0, 1)

    png_numpy=torch.squeeze(any_output.detach()).cpu().numpy()
    result_png = Image.fromarray((png_numpy * 255).astype(np.uint8))

    return png_numpy, result_png

def parse_args():
    parser = argparse.ArgumentParser(description="Generate image editing commands using Qwen3-VL model.")

    # 新增参数
    # parser.add_argument('--gpu', type=int, default=0, help='指定运行的GPU设备ID (e.g., 0, 1, 2).')
    parser.add_argument('--num_splits', type=int, default=8, help='将数据集切分的总块数.')
    parser.add_argument('--split_index', type=int, default=5, help='当前程序处理的是第几块数据 (从 0 开始).')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    cur_id = args.split_index
    total_id = args.num_splits

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    dst_foldr_all = '/mmu-vcg/zb08/outputs/texture_edit/new_model_out/all_edges/score'
    os.makedirs(dst_foldr_all, exist_ok=True)

    # 线框监测
    ckpt_p = '/mmu-vcg/zb08/yihang/2026_works/SAUGE/model/sam_vit_b_01ec64.pth'
    sam = sam_model_registry['vit_b'](checkpoint=ckpt_p)
    for name, parameter in sam.named_parameters():
        parameter.requires_grad = False
    sam.cuda()
    mask_generator = SamAutomaticMaskGeneratorForTest(sam)

    model = Model.SAUGE(None, sam_generator=mask_generator, mode='eval').cuda()
    checkpoint = torch.load('/mmu-vcg/zb08/codes/UniWorld-main/UniWorld-V2/ckpt/bsds/sauge_vitb.pth')['state_dict']

    model.load_state_dict(checkpoint)
    model.eval()

    dst_dir_f = '/mmu-vcg/zb08/outputs/texture_edit/new_model_out/all_edges/texture'

    rl = '/mmu-vcg/zb08/outputs/texture_edit/sft_base/2026.01.05_14.31_0_eval_rl_more_36_test_20dpm_texture/edited'
    sft_base = '/mmu-vcg/zb08/outputs/texture_edit/sft_base/2026.01.04_11.52_0_eval_sftbase_dpm20_test_real/edited'
    origin = '/mmu-vcg/zb08/outputs/texture_edit/sft_base/2026.01.04_11.26_0_eval_origin_dpm20-tsts_real/edited'
    gemni25 = '/mmu-vcg/zb08/outputs/texture_edit/new_model_out/gemini-2.5/final_coco_texture'
    gemini3 = '/mmu-vcg/zb08/outputs/texture_edit/new_model_out/gemini-3/final_coco_texture'
    qwen_11 = '/mmu-vcg/zb08/outputs/texture_edit/sft_base/2026.01.05_16.15_0_eval_2511_base_texture_dpm20/edited'
    # 先弄texture
    # 这个感觉 texture 和 属性还是分开写吧  
    source_imgs_f = '/mmu-vcg/zb08/yihang/enhance_coco_data_12_25/new_data/new_data_texture'
    img_folder_list = [ gemini3, gemni25, sft_base, rl, origin, qwen_11 ]
    img_folder_dict = { "gemini3": gemini3, "gemini2.5": gemni25, "sft_base": sft_base, "rl": rl, "origin": origin, "qwen11": qwen_11 }

    ddd, rank = get_local_device()

    results = []

    for k in img_folder_dict:
        print("Processing folder: ", k)

        curres_dict = {}
        cur_p = img_folder_dict[k]

        dst_folder = os.path.join(dst_dir_f, k)
        os.makedirs(dst_folder, exist_ok=True)

        dst_s_folder = os.path.join(dst_dir_f, 'source')
        os.makedirs(dst_s_folder, exist_ok=True)

        names = os.listdir(cur_p)

        # 数据分片 
        n = len(names)
        chunk_size = (n + total_id - 1) // total_id  # 向上取整
        sub_names = names[cur_id*chunk_size : (cur_id+1)*chunk_size]

        for cur_name in tqdm(sub_names) :
            src_p = os.path.join(source_imgs_f, cur_name)
            dst_p = os.path.join(cur_p, cur_name)

            if (not os.path.exists(dst_p)) and os.path.exists(src_p):
                print("no exists ", dst_p)
                continue
            
            img1 = Image.open(dst_p).convert("RGB") # 修改后的
            img2 = Image.open(src_p).convert("RGB") # 原始的

            img_t = transforms.ToTensor()(img1).unsqueeze(0).to(ddd)
            with torch.no_grad():
                multi_outputs, outputs, _, _= model(img_t)
            img_edge, img_edge_pil = post_process_edge(multi_outputs)

            ref_img_t = transforms.ToTensor()(img2).unsqueeze(0).to(ddd)
            with torch.no_grad():
                multi_outputs2, outputs, _, _= model(ref_img_t)
            ref_img_edge, ref_img_edge_pil = post_process_edge(multi_outputs2)

            try:
                score = ssim(
                img_edge,
                ref_img_edge,
                data_range=1.0
                )

                dst_cur_edge_p = os.path.join(dst_folder, cur_name)
                img_edge_pil.save(dst_cur_edge_p)

                dst_s_edge_p = os.path.join(dst_s_folder, cur_name)
                if not os.path.exists(dst_s_edge_p):
                    ref_img_edge_pil.save(dst_s_edge_p)

                curres_dict[cur_name] = score
            except Exception as e:

                print(img_edge.shape, ref_img_edge.shape)

                curres_dict[cur_name] = 0
                print("Error at ", cur_name, e)
                continue
        
        results.append({k: curres_dict})

    dst_p = os.path.join(dst_foldr_all, f"final_part{cur_id}.json")

    # === 4. 保存结果 ===
    with open(dst_p, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Finished! Saved to {dst_p}")
