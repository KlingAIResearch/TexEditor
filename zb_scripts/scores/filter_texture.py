from google import genai
from google.genai import types
import os


import math 
from PIL import Image
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/mmu-vcg/zb08/mmu-ketu-eff898205d01.json'


# 这里计划写一个多进程的 打分judge 脚本  
# 打分我计划 分成2 部分  这里的编辑的 指令跟随程度和美观度的综合打分 、 用线条和 mask 做一致性的打分  或者再看看之前的文章 


import re
from typing import Optional, Dict

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
            r"(score|rating)?\s*[:=]?\s*(\d+(\.\d+)?)\s*/?\s*10?",
            text,
            re.IGNORECASE
        )
        if fallback_match:
            score = float(fallback_match.group(2))
            if 0 <= score <= 10:
                result["score"] = score

    return result


import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from PIL import Image
import torch
import numpy as np

import json
from pycocotools.coco import COCO
from tqdm import tqdm
# from openai import OpenAI

# from modelscope import AutoModelForCausalLM, AutoTokenizer
# from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

import random

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate image editing commands using Qwen3-VL model.")

    # 新增参数
    # parser.add_argument('--gpu', type=int, default=0, help='指定运行的GPU设备ID (e.g., 0, 1, 2).')
    parser.add_argument('--num_splits', type=int, default=8, help='将数据集切分的总块数.')
    parser.add_argument('--split_index', type=int, default=3, help='当前程序处理的是第几块数据 (从 0 开始).')
    parser.add_argument('--dst_foldr', type=str, default="", help='')
    parser.add_argument('--dst_img_p', type=str, default="", help='')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    results = []

    # dst_foldr = args.dst_foldr
    # dst_img_p = args.dst_img_p
    dst_img_p = '/mmu-vcg/zb08/outputs/texture_edit/sft_base/2026.01.07_19.30_0_eval_train_more_e2/edited'
    dst_foldr = '/mmu-vcg/zb08/outputs/texture_edit/final_old_scores/tain_more_e2'
    # dst_foldr = '/mmu-vcg/zb08/outputs/texture_edit/final_old_scores/tain_texture_e1'
    # dst_foldr = '/mmu-vcg/zb08/outputs/texture_edit/final_old_scores/qwen2511_dpm20_texture_500_g2'
    # dst_foldr = '/mmu-vcg/zb08/outputs/texture_edit/final_old_scores/gemini3_dpm20_texture_500_g2'
    # dst_foldr = '/mmu-vcg/zb08/outputs/texture_edit/final_old_scores/sft_dpm20_texture_new_500_g3'
    # dst_foldr = '/mmu-vcg/zb08/outputs/texture_edit/final_old_scores/origin_dpm20_texture_new'

    os.makedirs(dst_foldr, exist_ok=True)

    # 输入图像与指令都是一样的
    # in_p = '/mmu-vcg/zb08/outputs/texture_edit/train_data/rl_data/maybe_v2_final_texture/train.jsonl'
    in_p = '/mmu-vcg/zb08/outputs/texture_edit/train_data/rl_data/maybe_v2_final_more/train.jsonl'

    # dst_img_p = '/mmu-vcg/zb08/outputs/texture_edit/sft_base/2026.01.04_13.15_0_eval_sftbase_dpm20_more-test_real'
    # dst_img_p = '/mmu-vcg/zb08/outputs/texture_edit/sft_base/2026.01.07_14.38_0_eval_train_texture_e2/edited'
    # dst_img_p = '/mmu-vcg/zb08/outputs/texture_edit/new_model_out/gemini-2.5/final_coco_texture'
    # dst_img_p = '/mmu-vcg/zb08/outputs/texture_edit/sft_base/2026.01.05_13.38_0_eval_rl_more_36_test_20dmp_more/edited'
    # dst_img_p = '/mmu-vcg/zb08/outputs/texture_edit/sft_base/2026.01.04_13.15_0_eval_sftbase_dpm20_more-test_real/edited'
    # dst_img_p = '/mmu-vcg/zb08/outputs/texture_edit/sft_base/2026.01.04_13.58_0_eva_origin_dpm_20_more_test/edited'
    # dst_img_p = '/mmu-vcg/zb08/outputs/texture_edit/sft_base/2026.01.04_11.52_0_eval_sftbase_dpm20_test_real/edited'

    with open(in_p, "r", encoding="utf-8") as f:
        in_data = [json.loads(line) for line in f]
    # in_data = json.load(open(in_p, "r", encoding="utf-8"))
    # in_data = in_data[:200]
    #对in_data 分片

    cur_id = args.split_index
    total_id = args.num_splits


    PROJECT_ID = "mmu-ketu"
    LOCATION = "global"

    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    # 数据分片 
    n = len(in_data)
    chunk_size = (n + total_id - 1) // total_id  # 向上取整
    sub_data = in_data[cur_id*chunk_size : (cur_id+1)*chunk_size]


    # === 2. 遍历每张图像 ===
    for item in tqdm(sub_data):  # 

        # print("jj")
        # img_id_name = item['image_id']
        img_path = item['edit_image']
        img_name = os.path.basename(img_path)

        d_img_path = os.path.join(dst_img_p, img_name)

        if (not os.path.exists(img_path)) or (not os.path.exists(d_img_path)):
            print(f"❌ Image not found: {d_img_path}")
            continue


        ins = item['prompt']
        # cat = item['category'] 

        with open(img_path, 'rb') as f:
            simg_bytes = f.read()
        with open(d_img_path, 'rb') as f:
            dimg_bytes = f.read()


        # 补充原图 和 目标图 


        # === 3. 构造提示 ===
        prompt = f"""
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

        cur_results = []
        for i in range(3):
            try:

                response = client.models.generate_content(
                # model='gemini-3-flash-preview',
                model='gemini-2.5-flash',
                contents=[
                    'origin image:',
                    types.Part.from_bytes(
                    data=simg_bytes,
                    mime_type='image/jpeg',
                    ),
                    'edited image:',
                    types.Part.from_bytes(
                    data=dimg_bytes,
                    mime_type='image/jpeg',
                    ),
                    'editing instruction:',
                    ins,
                    prompt
                ]
                )

                print(response.text)
                o1 = parse_eval_output(response.text)
                
                if o1["score"] is None:
                    print(f"❌ Failed to parse score, defaulting to 0.0")
                    cur_results.append(0.0)
                else:
                    cur_results.append(o1["score"])

                if len(cur_results) >=2:
                    tts = sum(cur_results) / len(cur_results)
                    break
                
            except Exception as e:
                print(f"❌ Error during instruction follow-up evaluation (attempt {i+1}/3): {e}")
                sleep_time = random.uniform(1.0, 2.0)
                import time
                time.sleep(sleep_time)
                # cur_results.append(0.0)
                tts = 0.0
                break

        results.append({
            "origin_p": img_path,
            "edited_p": d_img_path,
            "instruction": ins,
            "scores": tts,
        })


        sleep_time = random.uniform(1.0, 2.0)
        import time
        time.sleep(sleep_time)

    dst_p = os.path.join(dst_foldr, f"final_part{cur_id}.json")

    # === 4. 保存结果 ===
    with open(dst_p, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Finished! Saved to {dst_p}")
