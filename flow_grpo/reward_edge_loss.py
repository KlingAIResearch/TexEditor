from PIL import Image
import io
import os
import numpy as np
import torch
from collections import defaultdict
import random

from google import genai
from google.genai import types

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


URL0 = "http://10.82.121.94:9004/reward" # aiplatform 
URL1 = "http://10.82.121.94:9002/reward" # aiplatform 
URL2 = "http://10.82.121.94:9003/reward" # aiplatform 
URL3 = "http://10.82.121.94:9005/reward" # aiplatform 
ULR_list = [URL0, URL1, URL2, URL3]


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


def gemini_score_api(device=None):
    import time
    import random
    import re
    from io import BytesIO
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import torch
    from PIL import Image

    # =========================
    # Utils
    # =========================

    def pil_to_jpeg_bytes(pil_img):
        buf = BytesIO()
        pil_img.save(buf, format="JPEG")
        return buf.getvalue()

    def extract_scores(text_outputs):
        scores = []
        pattern = r"<answer>\s*([0-9]+(?:\.[0-9]+)?)\s*</answer>"
        for text in text_outputs:
            if not text:
                scores.append(0.0)
                continue
            m = re.search(pattern, text)
            if m:
                try:
                    scores.append(float(m.group(1)))
                except ValueError:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return scores

    # =========================
    # Gemini blocking call
    # =========================

    def evaluate_image_blocking(
        ref_img_bytes,
        img_bytes,
        instruction,
        client,
        max_retry=4,
    ):

        system_prompt = f"""
        You are given an editing instruction, a source image, and an edited image.

        Your task is to strictly evaluate whether the texture editing instruction is followed.
        Assume the instruction is NOT correctly followed unless there is clear visual evidence.

        Scoring criteria:

        9-10:
        Near-perfect texture editing. Texture matches the instruction precisely,
        with consistent spatial distribution, preserved fine details,
        and no artifacts, leakage, structural change, or added objects.

        7-8:
        Instruction largely followed. Texture change is correct in intent,
        but shows minor issues such as slight inconsistency, mild over-smoothing,
        or small local artifacts. No new objects or structural changes.

        4-6:
        Partial or ambiguous compliance. Noticeable texture errors, loss of detail,
        imprecise alignment with the instruction, or visible side effects.
        No hard constraint violation, but quality is clearly degraded.

        1-3:
        Major failure. Severe texture mismatch, instruction mostly ignored,
        or clear structural or semantic errors.
        May include unintended structural changes but no fully distinct new objects.

        0-2 (Hard Violation Override):
        Any unintended new object or semantic entity is introduced.
        This is considered a severe violation regardless of texture quality.

        First, provide a concise justification in no more than 25 words.
        Then, assign a score from 0 to 10.

        Wrap the justification with <think></think> and the score with <answer></answer>.
        """ 

        s_list = []
        first = None
        for attempt in range(max_retry):
            try:
                response = client.models.generate_content(
                    # model="gemini-2.5-flash",
                    model="gemini-3-flash-preview",
                    contents=[
                        "origin image:",
                        types.Part.from_bytes(
                            data=ref_img_bytes,
                            mime_type="image/jpeg",
                        ),
                        "edited image:",
                        types.Part.from_bytes(
                            data=img_bytes,
                            mime_type="image/jpeg",
                        ),
                        "editing instruction:",
                        instruction,
                        system_prompt,
                    ],
                )

                # s , cot = parse_eval_output(response.text)
                rrr = parse_eval_output(response.text)
                s = rrr['score']
                if s is not None:
                    if len(s_list) == 0:
                        first = response.text
                    s_list.append(s)
                else:
                    print(f"⚠️ Gemini output parsing failed, retrying... Attempt {attempt + 1}")
                    print("Gemini output:", response.text)
                    print(response)
                    print(response.candidates)
                    sleep_time = 1.5 * attempt
                    time.sleep(sleep_time)
                    continue
                
                if len(s_list) >=2:
                    cur_std = np.std(s_list) 

                    if cur_std < 0.5:
                        break
                    if len(s_list)>=4:
                        break
                
            except Exception as e:
                print(f"❌ Gemini failed after {max_retry} tries: {e}")
                return 4.0 , "bad response"

        # 返回平均分
        avg_s = sum(s_list) / len(s_list) if len(s_list)>0 else None

        device, rank  = get_local_device()
        if rank ==0:
            print(f"instruction: {instruction}, Gemini score: {s_list}")
        avg_s = avg_s-0.2 * len(s_list) if len(s_list)>=2 else avg_s #  如果 波动比较大 就稍微调低一点 因为这种情况下的高分往往都是看偏了
        return avg_s, first

    # =========================
    # Threaded batch execution
    # =========================

    def evaluate_batch_threaded(
        ref_imgs_bytes,
        imgs_bytes,
        prompts,
        client,
        max_workers=4,
    ):
        results = [None] * len(prompts)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    evaluate_image_blocking,
                    ref_imgs_bytes[i],
                    imgs_bytes[i],
                    prompts[i],
                    client,
                ): i
                for i in range(len(prompts))
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"❌ Unexpected thread error: {e}")
                    results[idx] = None

        return results

    # =========================
    # External callable fn
    # =========================

    def fn(ref_images, images, prompts, metadata):
        # ---- Tensor -> PIL ----


        if isinstance(images, torch.Tensor):
            images = (images * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)

        if isinstance(ref_images, torch.Tensor):
            ref_images = (ref_images * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            ref_images = ref_images.transpose(0, 2, 3, 1)

        # 都给转为Numpy 


        images = [Image.fromarray(img) for img in images]
        # images = [Image.fromarray(img).resize((512, 512)) for img in images]
        # ref_images = [Image.fromarray(img).resize((512, 512)) for img in ref_images] # 这一行有时候不需要 

        imgs_bytes = [pil_to_jpeg_bytes(img) for img in images]
        ref_imgs_bytes = [pil_to_jpeg_bytes(img) for img in ref_images]

        # ---- Gemini client ----
        PROJECT_ID = "mmu-ketu"
        LOCATION = "global"

        client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=LOCATION,
        )

        # ---- threaded reward ----
        scores_list = evaluate_batch_threaded(
            ref_imgs_bytes,
            imgs_bytes,
            prompts,
            client,
            max_workers=2,  # ⭐ 并发阈值
        )

        scores = []
        # reasons = []
        text_outputs = []
        for item in scores_list:
            
            if item is None:
                all_out_item = "bad response"
                score = None
                scores.append(score if score is not None else 4.0)
                text_outputs.append(all_out_item)
            else:
                score, all_out_item = item

                scores.append(score if score is not None else 4.0)
                text_outputs.append(all_out_item)

        scores = [s / 10 for s in scores]  # normalize to [0,1]

        return scores, text_outputs

    return fn


def gemini_aes_api(device=None):
    import time
    import random
    import re
    from io import BytesIO
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import torch
    from PIL import Image
    # import google.generativeai as genai
    # from google.generativeai import types

    # =========================
    # Utils
    # =========================

    def pil_to_jpeg_bytes(pil_img):
        buf = BytesIO()
        pil_img.save(buf, format="JPEG")
        return buf.getvalue()

    def extract_scores(text_outputs):
        scores = []
        pattern = r"<answer>\s*([0-9]+(?:\.[0-9]+)?)\s*</answer>"
        for text in text_outputs:
            if not text:
                scores.append(0.0)
                continue
            m = re.search(pattern, text)
            if m:
                try:
                    scores.append(float(m.group(1)))
                except ValueError:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return scores

    # =========================
    # Gemini blocking call
    # =========================

    def evaluate_image_blocking(
        ref_img_bytes,
        img_bytes,
        instruction,
        client,
        max_retry=6,
    ):

        system_prompt = f"""
        You will be given a source image and an edited image.
        Your task is to evaluate the consistency of edited image with the source image, epsecially the shape and deatil.
        First, provide a concise justification in no more than 25 words.
        Then, assign a score from 0 to 10 reflecting overall aesthetics and consistency.
        Wrap the justification with <think></think> and the score with <answer></answer>.
        """ 

        s_list = []
        first = None
        for attempt in range(max_retry):
            try:
                response = client.models.generate_content(
                    # model="gemini-2.5-flash",
                    model="gemini-3-flash-preview",
                    contents=[
                        "origin image:",
                        types.Part.from_bytes(
                            data=ref_img_bytes,
                            mime_type="image/jpeg",
                        ),
                        "edited image:",
                        types.Part.from_bytes(
                            data=img_bytes,
                            mime_type="image/jpeg",
                        ),
                        "editing instruction:",
                        instruction,
                        system_prompt,
                    ],
                )

                # s , cot = parse_eval_output(response.text)
                rrr = parse_eval_output(response.text)
                s = rrr['score']
                if s is not None:
                    if len(s_list) == 0:
                        first = response.text
                    s_list.append(s)
                else:
                    print(f"⚠️ Gemini output parsing failed, retrying... Attempt {attempt + 1}")
                    print("Gemini output:", response.text)
                    print(response)
                    print(response.candidates)
                    sleep_time = 2 * attempt + random.uniform(0.5, 1.5) + 1
                    time.sleep(sleep_time)
                    continue
                
                if len(s_list) >=2:
                    cur_std = np.std(s_list) 

                    if cur_std < 0.5:
                        break
                    if len(s_list)>=3:
                        break
                
            except Exception as e:
                print(f"❌ Gemini failed after {max_retry} tries: {e}")
                if attempt == max_retry - 1:
                    return ""

                sleep_time = 2 * attempt + random.uniform(0.5, 1.5) + 1
                time.sleep(sleep_time)
        # 返回平均分
        avg_s = sum(s_list) / len(s_list) if len(s_list)>0 else None

        device, rank  = get_local_device()
        if rank ==0:
            print(f"instruction: {instruction}, aes score: {s_list}")
        avg_s = avg_s-0.2 * len(s_list) if len(s_list)>=2 else avg_s #  如果 波动比较大 就稍微调低一点 因为这种情况下的高分往往都是看偏了
        return avg_s, first

    # =========================
    # Threaded batch execution
    # =========================

    def evaluate_batch_threaded(
        ref_imgs_bytes,
        imgs_bytes,
        prompts,
        client,
        max_workers=4,
    ):
        results = [None] * len(prompts)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    evaluate_image_blocking,
                    ref_imgs_bytes[i],
                    imgs_bytes[i],
                    prompts[i],
                    client,
                ): i
                for i in range(len(prompts))
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"❌ Unexpected thread error: {e}")
                    results[idx] = None

        return results

    # =========================
    # External callable fn
    # =========================

    def fn(ref_images, images, prompts, metadata):
        # ---- Tensor -> PIL ----


        if isinstance(images, torch.Tensor):
            images = (images * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)

        if isinstance(ref_images, torch.Tensor):
            ref_images = (ref_images * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            ref_images = ref_images.transpose(0, 2, 3, 1)

        # 都给转为Numpy 


        images = [Image.fromarray(img).resize((512, 512)) for img in images]
        # ref_images = [Image.fromarray(img).resize((512, 512)) for img in ref_images] # 这一行有时候不需要 

        imgs_bytes = [pil_to_jpeg_bytes(img) for img in images]
        ref_imgs_bytes = [pil_to_jpeg_bytes(img) for img in ref_images]

        # ---- Gemini client ----
        PROJECT_ID = "mmu-ketu"
        LOCATION = "global"

        client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=LOCATION,
        )

        # ---- threaded reward ----
        scores_list = evaluate_batch_threaded(
            ref_imgs_bytes,
            imgs_bytes,
            prompts,
            client,
            max_workers=2,  # ⭐ 并发阈值
        )

        scores = []
        # reasons = []
        text_outputs = []
        for item in scores_list:
            
            if item is None:
                all_out_item = ""
                score = 0.0
            else:
                score, all_out_item = item

                scores.append(score if score is not None else 4.0)
                text_outputs.append(all_out_item)

        scores = [s / 10 for s in scores]  # normalize to [0,1]

        return scores, text_outputs

    return fn


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



def shape_loss (device=None):
    import time
    import random
    import re
    from io import BytesIO
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import torch
    from PIL import Image

    # =========================
    # sam call
    # =========================

    def evaluate_sam_blocking(
        idx,
        ref_img,
        img,
        instruction,
        meta_o,
    ):

        ddd, rank =  get_local_device()

        # 这里确认了一下 输入都是pil img  okk

        kind = meta_o['kind']

        b64_img_ref = pil2b64(ref_img)
        b64_img_cur = pil2b64(img)

        # url 选择 
        cur_id = idx % 4
        URL = ULR_list[cur_id]

        # 发送请求
        r = requests.post(
        URL,
        json={
            "samples": [
                {
                    "image1_b64": b64_img_cur,
                    "image2_b64": b64_img_ref,
                    "prompt": instruction,
                    "kind": kind
                }
            ]
        },
        timeout=60
        )
        reward = r.json()

        cur_dict = reward['results'][0]
        img_cur = b64_to_pil(cur_dict['image_edge1_b64'])
        img_ref = b64_to_pil(cur_dict['image_edge2_b64'])
        s = cur_dict['iou']

        # 这里写一个加工处理逻辑 算了 转移到 损失函数吧


        return s, img_cur , img_ref


    # =========================
    # Threaded batch execution
    # =========================

    def evaluate_sam_threaded(
        ref_imgs,
        images,
        prompts,
        metadata,
        max_workers=1,
    ):
        results = [None] * len(prompts)

        # 这里写一个 发送与重传的 try except
        for i in range(len(prompts)):
            for idx in range(2):
                try:
                    s, cur_img, ref_img = evaluate_sam_blocking(i, ref_imgs[i],images[i],prompts[i],metadata[i])
                    results[i] = s
                    break
                except Exception as e:
                    print(f"❌ Unexpected thread error: {e}")
        

        return results

    # =========================
    # External callable fn
    # =========================

    def fn(ref_images, images, prompts, metadata):

        # device, rank = get_local_device()
        # print("Using device for SAM IOU:", device)
        # ---- Tensor -> PIL ----

        # breakpoint()
        if isinstance(images, torch.Tensor):
            images = (images * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)

        if isinstance(ref_images, torch.Tensor):
            ref_images = (ref_images * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            ref_images = ref_images.transpose(0, 2, 3, 1)

        # 都给转为Numpy 

        images = [Image.fromarray(img).resize((1024, 1024)) for img in images]
        # ref_images = [Image.fromarray(img).resize((512, 512)) for img in ref_images] # 这一行有时候不需要 

        # ---- threaded reward ----
        scores_list = evaluate_sam_threaded(
            ref_images,
            images,
            prompts,
            metadata,
            max_workers=1,  # ⭐ 并发阈值
        )

        scores = []

        for item in scores_list:
            
            score = item

            scores.append(score if score is not None else 0.0)

        return scores

    return fn



def dummy():
    def _fn(images, prompts, metadata):
        return [random.random() for _ in range(len(images))], {}
    return _fn
    

def multi_score(old_device, score_dict):
    device, rank  = get_local_device()

    score_functions = {
        "zb_gemini": gemini_score_api,
        "sam_iou" : shape_loss,
        "aesthetic": aesthetic_score,
        "gemini_aes": gemini_aes_api,
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
            elif score_name == "sam_iou":
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

    ppp = '/mmu-vcg/zb08/outputs/texture_edit/train_data/new_coco_4_train/merged_no_wash.json'
    # 读取json 
    import json
    data = json.load(open(ppp, "r", encoding="utf-8"))
    data_cur = data[:5] # 只取前50个做测试

    ref_imgs_paths = [ item['edit_image'] for item in data_cur ]
    image_paths = [ item['image'] for item in data_cur ]
    prompts = [ item['prompt'] for item in data_cur ]
    metadata = [ {"object":item['object']} for item in data_cur ]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert to tensor
        ]
    )

    images = torch.stack([transform(Image.open(image_path).convert("RGB")) for image_path in image_paths])

    ref_imgs = [ Image.open(image_path).convert("RGB") for image_path in ref_imgs_paths] 

    # metadata = {}  # Example metadata

    # score_dict = {"sam_iou": 1.0}
    score_dict = {"sam_iou": 0.45, "zb_gemini": 0.45, "gemini_aes":0.1}
    # score_dict = {"sam_iou": 0.45, "zb_gemini": 0.4, "aesthetic": 0.15}
    # Initialize the multi_score function with a device and score_dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring_fn = multi_score(device, score_dict)
    # Get the scores
    scores, _ = scoring_fn(images, prompts, metadata, ref_imgs)
    # Print the scores
    print("Scores:", scores)


    # 晚上在这里 把其他loss 一起调试好 ！！！

if __name__ == "__main__":
    main()
