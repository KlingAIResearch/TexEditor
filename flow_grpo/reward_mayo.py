from PIL import Image
import io
import os
import numpy as np
import torch
from collections import defaultdict
import random

from google import genai
from google.genai import types

import math
from typing import List, Tuple

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/mmu-vcg/zb08/mmu-ketu-eff898205d01.json'

import sys
sys.path.append("/mmu-vcg/zb08/codes/UniWorld-main/UniWorld-V2/flow_grpo")
sys.path.append("/mmu-vcg/zb08/codes/UniWorld-main/UniWorld-V2")
sys.path.append("/mmu-vcg/zb08/codes/sam3-main")
from zb_reward_helper import *

import re

import torchvision.transforms as transforms
from typing import Optional, Dict

from skimage.metrics import structural_similarity as ssim

import random

import base64
import io
from PIL import Image

import requests

import lpips


URL0 = "http://10.82.121.94:9004/reward" # aiplatform 
URL1 = "http://10.82.121.94:9002/reward" # aiplatform 
URL2 = "http://10.82.121.94:9003/reward" # aiplatform 
URL3 = "http://10.82.121.94:9005/reward" # aiplatform 
ULR_list = [URL0, URL1, URL2, URL3]

def lpips_from_pil_lists(
    preds,
    targets,
    net = "vgg",        # "alex" | "vgg" | "squeeze"
    resize: bool = False,
    device: str = None,
) -> Tuple[float, List[float]]:
    """
    Compute LPIPS between two lists of PIL Images.

    Args:
        preds:    List of predicted PIL images
        targets: List of target (GT) PIL images
        net:      LPIPS backbone: "alex" (default), "vgg", "squeeze"
        resize:   If True, resize preds to match targets
        device:   "cuda" or "cpu" (auto-detect if None)

    Returns:
        mean_lpips: average LPIPS score (lower is better)
        lpips_list: per-image LPIPS values
    """
    assert len(preds) == len(targets), "Image list lengths must match"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    loss_fn = lpips.LPIPS(net=net).to(device)
    loss_fn.eval()

    lpips_list = []

    with torch.no_grad():
        for pred, gt in zip(preds, targets):
            # --- ensure RGB ---
            pred = pred.convert("RGB")
            gt = gt.convert("RGB")

            # --- optional resize ---
            if resize and pred.size != gt.size:
                pred = pred.resize(gt.size, Image.BICUBIC)

            # --- PIL -> tensor, range [-1, 1] ---
            pred_t = torch.from_numpy(
                np.asarray(pred, dtype=np.float32)
            ).permute(2, 0, 1) / 127.5 - 1.0

            gt_t = torch.from_numpy(
                np.asarray(gt, dtype=np.float32)
            ).permute(2, 0, 1) / 127.5 - 1.0

            pred_t = pred_t.unsqueeze(0).to(device)
            gt_t = gt_t.unsqueeze(0).to(device)

            # --- LPIPS ---
            d = loss_fn(pred_t, gt_t)
            lpips_list.append(float(d.item()))

    mean_lpips = float(np.mean(lpips_list))
    return mean_lpips, lpips_list


def pil2b64(img: Image.Image, format="PNG", quality=100) -> str:
    """
    PIL.Image -> base64 string (no header)
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    buffer = io.BytesIO()
    img.save(buffer, format=format, quality=quality)
    buffer.seek(0)

    img_bytes = buffer.read()
    b64_str = base64.b64encode(img_bytes).decode("utf-8")
    return b64_str

def b64_to_pil(b64_str: str) -> Image.Image:
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_bytes))
    return img.convert("RGB")

def get_local_device():
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return torch.device(f"cuda:{local_rank}"),local_rank
    return torch.device("cpu"),local_rank

def parse_eval_output(text: str) -> Dict[str, Optional[str]]:
    """
    Robustly parse <think> justification and <answer> score from model output.

    Returns:
        {
            "reason": str or None,
            "score": float or None,
            "raw": original text
        }
    """
    result = {
        "reason": None,
        "score": None,
        "raw": text
    }

    if not text or not isinstance(text, str):
        return result

    # -------- 1. 解析 think --------
    think_pattern = re.compile(
        r"<\s*think\s*>(.*?)<\s*/\s*think\s*>",
        re.IGNORECASE | re.DOTALL
    )
    think_match = think_pattern.search(text)
    if think_match:
        reason = think_match.group(1).strip()
        # 压缩多余空白
        reason = re.sub(r"\s+", " ", reason)
        result["reason"] = reason if reason else None

    # -------- 2. 解析 answer --------
    answer_pattern = re.compile(
        r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>",
        re.IGNORECASE | re.DOTALL
    )
    answer_match = answer_pattern.search(text)

    score_text = None
    if answer_match:
        score_text = answer_match.group(1).strip()

    # -------- 3. 从 answer 中提取数值 --------
    if score_text:
        # 支持：8 / 8.5 / 8/10 / score: 8
        num_match = re.search(
            r"(\d+(\.\d+)?)",
            score_text
        )
        if num_match:
            score = float(num_match.group(1))
            if 0 <= score <= 10:
                result["score"] = score

    # -------- 4. 兜底：全文找分数 --------
    if result["score"] is None:

        fallback_match = re.search(
            r"(?:<score>|score[:=]?)?\s*(\d+(?:\.\d+)?)\s*(?:</score>|/10)?",
            text,
            re.IGNORECASE
        )
        if fallback_match:
            score = float(fallback_match.group(1))
            if 0 <= score <= 10:
                result["score"] = score

    # 这里分数是float
    return result


def aesthetic_score(device):
    from flow_grpo.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        scores = scores.cpu()

        # new_list = []
        # scores = (scores - lo) / (hi - lo)
        normalized_scores = (scores- 3) / (7 - 3)  # Normalize to [0, 1]
        # 转换为数值
        new_list = [ ii.item() for ii in  normalized_scores]

        return new_list, {}

    return _fn



import torch
import cv2
import numpy as np


def post_process_edge(multi_outputs,alpha=0.4):
    percent_1 = alpha / 0.5
    percent_2 = 1 - percent_1
    any_output = percent_1 * multi_outputs[1] + percent_2 * multi_outputs[0]
    any_output = torch.clamp(any_output, 0, 1)

    png_numpy=torch.squeeze(any_output.detach()).cpu().numpy()
    # result_png = Image.fromarray((result * 255).astype(np.uint8))

    return png_numpy

def psnr_from_pil_lists(
    preds: List[Image.Image],
    targets: List[Image.Image],
    resize: bool = False,
) -> Tuple[float, List[float]]:
    """
    Compute PSNR between two lists of PIL Images.

    Args:
        preds:    List of predicted PIL images
        targets: List of target (GT) PIL images
        resize:  If True, resize preds to match targets' size

    Returns:
        mean_psnr: average PSNR over all image pairs
        psnr_list: per-image PSNR values
    """
    assert len(preds) == len(targets), "Image list lengths must match"

    psnr_list = []

    for pred, gt in zip(preds, targets):
        # --- ensure RGB ---
        pred = pred.convert("RGB")
        gt = gt.convert("RGB")

        # --- optional resize ---
        if resize and pred.size != gt.size:
            pred = pred.resize(gt.size, Image.BICUBIC)

        # --- to numpy, float32 ---
        pred_np = np.asarray(pred, dtype=np.float32)
        gt_np = np.asarray(gt, dtype=np.float32)

        # --- MSE ---
        mse = np.mean((pred_np - gt_np) ** 2)

        if mse == 0:
            psnr = float("inf")
        else:
            psnr = 20 * math.log10(255.0) - 10 * math.log10(mse)

        psnr_list.append(psnr)

    mean_psnr = float(np.mean(psnr_list))
    return mean_psnr, psnr_list


# def norm_psnr(lll):
#     # 越高越好 0-1
#     new_list = []
#     for item in lll:
#         if item <35:
#             item = (item - 18) / (35-18)
#         else:
#             item = 1.0
#         new_list.append(item)
#     return new_list

import math

def norm_psnr(psnr_list, ref=20.0, scale=0.2):
    """
    ref:   PSNR reference point (30dB 常用)
    scale: 控制增长速度
    """
    rewards = []
    for p in psnr_list:
        r = 1.0 - math.exp(-scale * max(p - ref, 0))
        rewards.append(r)
    return rewards


# def norm_lpips(lll):
#     # 越低越好 0-1
#     new_list = []
#     for item in lll:
#         if item > 0.01:
#             item = (item - 0.01) / (0.20 - 0.01)
#         else:
#             item = 1.0
#         new_list.append(item)
#     return new_list
#     # pass

def norm_lpips(lpips_list, scale=10.0):
    return [math.exp(-scale * d) for d in lpips_list]

def pixel_loss (device=None):
    import time
    import random
    import re
    from io import BytesIO
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import torch
    from PIL import Image


    # =========================
    # Threaded batch execution
    # =========================

    def evaluate_psnr_threaded(
        gt_imgs,
        images,
        prompts,
        metadata,
        max_workers=1,
    ):
        results = [None] * len(prompts)

        m_psnr,psnr_list = psnr_from_pil_lists(gt_imgs, images)
        m_lpips, lpips_list = lpips_from_pil_lists(gt_imgs, images)

        psnr_list = norm_psnr(psnr_list)
        lpips_list = norm_lpips(lpips_list)

        return psnr_list, lpips_list, m_psnr, m_lpips

    # =========================
    # External callable fn
    # =========================

    def fn(ref_images, images, prompts, metadata):

        # device, rank = get_local_device()
        # print("Using device for SAM IOU:", device)
        # ---- Tensor -> PIL ----
        net = "vgg"


        # breakpoint()
        if isinstance(images, torch.Tensor):
            images = (images * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)

        if isinstance(ref_images, torch.Tensor):
            ref_images = (ref_images * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            ref_images = ref_images.transpose(0, 2, 3, 1)

        # 这里要拿到GT imgs  并变成PIL 
        gt_imgs = [  Image.open(item['image']) for item in metadata ]

        # 都给转为Numpy 

        images = [Image.fromarray(img) for img in images]
        # ref_images = [Image.fromarray(img).resize((512, 512)) for img in ref_images] # 这一行有时候不需要 

        # ---- threaded reward ----
        psnr_list, lpips_list, m_psnr, m_lpips = evaluate_psnr_threaded(
            # ref_images,
            gt_imgs,
            images,
            prompts,
            metadata,
            max_workers=2,  # ⭐ 并发阈值
        )

        print("psnr:", m_psnr, "lpips:", m_lpips)
        scores = []

        for item in zip(psnr_list, lpips_list):            
            score = item
            m_s = 0.8*item[0] + 0.2*item[1]  # 最终得分是加权

            scores.append(m_s if m_s is not None else 0.0)

        return scores

    return fn



def dummy():
    def _fn(images, prompts, metadata):
        return [random.random() for _ in range(len(images))], {}
    return _fn
    

def multi_score(old_device, score_dict):
    device, rank  = get_local_device()

    score_functions = {
        # "zb_gemini": gemini_score_api,
        "pixel_loss" : pixel_loss,
        "aesthetic": aesthetic_score,
        # "gemini_aes": gemini_aes_api,
        "dummy": dummy
    }
    score_fns = {}
    for score_name, weight in score_dict.items():
        score_fns[score_name] = (
            score_functions[score_name](device)
            if "device" in score_functions[score_name].__code__.co_varnames
            else score_functions[score_name]()
        )

    # only_strict is only for geneval. During training, only the strict reward is needed, and non-strict rewards don't need to be computed, reducing reward calculation time.
    def _fn(images, prompts, metadata, ref_images=None, only_strict=True):
        total_scores = []
        score_details = {}

        device, rank  = get_local_device()
        # print_list = []

        for score_name, weight in score_dict.items():
            if score_name == "geneval":
                scores, rewards, strict_rewards, group_rewards, group_strict_rewards = score_fns[score_name](
                    images, prompts, metadata, only_strict
                )
                score_details["accuracy"] = rewards
                score_details["strict_accuracy"] = strict_rewards
                for key, value in group_strict_rewards.items():
                    score_details[f"{key}_strict_accuracy"] = value
                for key, value in group_rewards.items():
                    score_details[f"{key}_accuracy"] = value
            elif score_name.startswith("mllm_") or score_name == "zb_gemini" or score_name == "gemini_aes":
                scores, rewards = score_fns[score_name](ref_images, images, prompts, metadata) # 这里是有  ref_images 的 
            elif score_name == "sam_iou" or score_name == "pixel_loss":
                scores = score_fns[score_name](ref_images, images, prompts, metadata) # 这里是有  ref_images 的 
            else:
                scores, rewards = score_fns[score_name](images, prompts, metadata)
            score_details[score_name] = scores
            try:
                weighted_scores = [weight * score for score in scores]
            except:
                breakpoint()

            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]
            # print_list.append(scores)

        # if rank == 0:
        #     ll = len(prompts)
        #     for iii in range(ll):

        #         sam_s = score_details['sam_iou'][iii]
        #         # gem_s = score_details['zb_gemini'][iii]
        #         if 'gemini_aes' in score_details:
        #             aes_s = score_details['gemini_aes'][iii]
        #             print(f"Sample {iii}: Aesthetic: {aes_s:.4f}, SAM IOU: {sam_s:.4f}, Gemini: {gem_s:.4f}, prompt: {prompts[iii][:60]}...")
        #         else:
        #             # print(f"Sample {iii}: SAM IOU: {sam_s:.4f}, Gemini: {gem_s:.4f}, prompt: {prompts[iii][:60]}...")
        #             print(f"Sample {iii}: Gemini: {gem_s:.4f}, prompt: {prompts[iii][:80]}...")
        #             # print(f"Sample {iii}: Gemini: {gem_s:.4f}, prompt: {prompts[iii][:80]}...")

        # print("sam iou scores:", score_details['sam_iou'][:5])

        score_details["avg"] = total_scores
        return score_details, {}

    return _fn


def main():
    

    '''
    在测试阶段。这里其实输入的都是tensor。
    但在训练时候 ref-imgs 好像是pil
    
    '''

    ppp = '/mmu-vcg/zb08/codes/DiffSynth-Studio-main/data/example_image_dataset/zb_mayo.json'
    # 读取json 
    import json
    data = json.load(open(ppp, "r", encoding="utf-8"))
    data_cur = data[:5] # 只取前50个做测试

    ref_imgs_paths = [ item['edit_image'] for item in data_cur ]
    image_paths = [ item['image'] for item in data_cur ]
    prompts = [ item['prompt'] for item in data_cur ]
    metadata = [ item for item in data_cur ]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert to tensor
        ]
    )

    images = torch.stack([transform(Image.open(image_path).convert("RGB")) for image_path in image_paths])

    ref_imgs = [ Image.open(image_path).convert("RGB") for image_path in ref_imgs_paths] 

    # metadata = {}  # Example metadata

    # score_dict = {"sam_iou": 1.0}
    score_dict = {"pixel_loss": 1.0}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring_fn = multi_score(device, score_dict)
    # Get the scores
    scores, _ = scoring_fn(images, prompts, metadata, ref_imgs)
    # Print the scores
    print("Scores:", scores)


    # 晚上在这里 把其他loss 一起调试好 ！！！

if __name__ == "__main__":
    main()
