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
from typing import Optional, Dict



from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

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
        system_prompt = (
            "You are given an editing instruction, a source image, and an edited image.\n"
            "Your task is to evaluate how well the editing instruction has been followed.\n"
            "First, provide a concise justification in no more than 25 words.\n"
            "Then, assign a score from 0 to 5 reflecting the completion quality.\n"
            "Wrap the justification with <think></think> and the score with <answer></answer>."
        )

        s_list = []
        first = None
        for attempt in range(max_retry):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    # model="gemini-3-flash-preview",
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
                    sleep_time = 4 ** attempt + random.uniform(0.5, 1.5) + 10
                    time.sleep(sleep_time)
                    continue
                
                if len(s_list) >=2:
                    cur_std = np.std(s_list) 
                    # print(f"Gemini scores so far: {s_list}, std: {cur_std}")
                    if cur_std < 0.5:
                        break
                    if len(s_list)>=4:
                        break
                
            except Exception as e:
                print(f"❌ Gemini failed after {max_retry} tries: {e}")
                if attempt == max_retry - 1:
                    return ""

                sleep_time = 4 ** attempt + random.uniform(0.5, 1.5) + 10
                time.sleep(sleep_time)
        # 返回平均分
        avg_s = sum(s_list) / len(s_list) if len(s_list)>0 else None

        device, rank  = get_local_device()
        if rank ==0:
            print(f"instruction: {instruction}, Gemini score: {s_list}")
        avg_s = avg_s-0.1 * len(s_list) if len(s_list)>=1 else avg_s #  如果 波动比较大 就稍微调低一点 因为这种情况下的高分往往都是看偏了
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
                # print("Gemini Output:", all_out_item)
                scores.append(score if score is not None else 0.0)
                text_outputs.append(all_out_item)

        scores = [s / 5 for s in scores]  # normalize to [0,1]

        return scores, text_outputs

    return fn


def alpha_plus( s_img, mask, default_color=(255,0,0)):
    """ 
    s_img: PIL Image
    mask: numpy array HxW , bool
    default_color: tuple of 3 int
    """
    import cv2
    import numpy as np
    from PIL import Image

    s_img_np = np.array(s_img).copy()
    overlay = s_img_np.copy()

    color = np.array(default_color, dtype=np.uint8)

    overlay[mask] = color

    alpha = 0.5
    cv2.addWeighted(overlay, alpha, s_img_np, 1 - alpha, 0, s_img_np)

    return s_img_np



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
        ref_img,
        img,
        instruction,
        meta_o,
        processor
    ):

        ddd, rank =  get_local_device()
        # if rank == 2:
        #     breakpoint()

        with torch.no_grad():
            cur_object = meta_o.get("object", "object")
            state1 = processor.set_image(ref_img)
            infer1 = processor.set_text_prompt(state=state1, prompt=cur_object)
            mask1 = merge_masks(infer1.get("masks"))

            state2 = processor.set_image(img)
            infer2 = processor.set_text_prompt(state=state2, prompt=cur_object)
            mask2 = merge_masks(infer2.get("masks"))

        m_iou = mask_iou(mask1, mask2)

        if m_iou == 0:
            print("Zero IOU for instruction:", instruction)

        if m_iou > 0.9:
            normalized_diff = 1.0
        elif m_iou < 0.4:
            normalized_diff = 0.0
        else:
            abs_diff = m_iou - 0.4
            normalized_diff = abs_diff / 0.5

        # 可视化保存。可以注释掉
        vis_dir = '/mmu-vcg/zb08/outputs/texture_edit/aa_debug/aaa_sam_reward_vis_rllora'
        os.makedirs(vis_dir, exist_ok=True)
        mask_vis = visualize_mask_overlap(mask1, mask2, ref_img)

        mask11 = mask_to_bool_numpy(mask1, ref_img.size[::-1])
        mask22 = mask_to_bool_numpy(mask2, ref_img.size[::-1])

        mask_ref = alpha_plus(ref_img, mask11, default_color=(255,0,0))
        mask_img = alpha_plus(img, mask22, default_color=(0,0,255))

        combined = np.concatenate([ref_img, mask_ref, img, mask_img, mask_vis], axis=1)
        save_path = os.path.join(
            vis_dir,
            instruction + "_sam_iou_" + str(round(m_iou,4)) + ".png"
        )
        cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        # pass

        return normalized_diff

    # =========================
    # Threaded batch execution
    # =========================

    def evaluate_sam_threaded(
        ref_imgs,
        images,
        prompts,
        metadata,
        processor,
        max_workers=1,
    ):
        results = [None] * len(prompts)

        for i in range(len(prompts)):
            try:
                res = evaluate_sam_blocking(ref_imgs[i],images[i],prompts[i],metadata[i],processor)
                results[i] = res
            except Exception as e:
                print(f"❌ Unexpected thread error: {e}")
                results[i] = 0

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

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # model = build_sam3_image_model(device=device, checkpoint_path="/mmu-vcg/zb08/CKPTS/sam3/sam3.pt").to(device)
        model = build_sam3_image_model(checkpoint_path="/mmu-vcg/zb08/CKPTS/sam3/sam3.pt")
        # for p in model.parameters():
        #     assert p.device == device
        
        processor = Sam3Processor(model)
        # processor = Sam3Processor(model, device = device)


        # ---- threaded reward ----
        scores_list = evaluate_sam_threaded(
            ref_images,
            images,
            prompts,
            metadata,
            processor,
            max_workers=1,  # ⭐ 并发阈值
        )

        scores = []

        for item in scores_list:
            
            score = item
            # print("Gemini Output:", all_out_item)
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
            elif score_name.startswith("mllm_") or score_name == "zb_gemini":
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
        #         aes_s = score_details['aesthetic'][iii]
        #         sam_s = score_details['sam_iou'][iii]
        #         gem_s = score_details['zb_gemini'][iii]
        #         print(f"Sample {iii}: Aesthetic: {aes_s:.4f}, SAM IOU: {sam_s:.4f}, Gemini: {gem_s:.4f}, prompt: {prompts[iii][:60]}...")
        #     # breakpoint()
        print("sam iou scores:", score_details['sam_iou'][:5])

        score_details["avg"] = total_scores
        return score_details, {}

    return _fn


def main():
    import torchvision.transforms as transforms

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
    # ref_imgs = torch.stack([transform(Image.open(image_path).convert("RGB")) for image_path in ref_imgs_paths])
    ref_imgs = [ Image.open(image_path).convert("RGB") for image_path in ref_imgs_paths] 

    # metadata = {}  # Example metadata

    score_dict = {"sam_iou": 0.45, "zb_gemini": 0.4, "aesthetic": 0.15}
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
