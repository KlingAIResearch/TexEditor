import sys
sys.stdout.reconfigure(line_buffering=True)
import os
import json
import argparse
import re
from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.box_ops import box_xywh_to_cxcywh


# ========================= Utility Functions =========================

def normalize_bbox_cxcywh(box_cxcywh, img_width, img_height):
    """Normalize (cx, cy, w, h) to [0, 1]."""
    b = box_cxcywh.clone().float()
    if b.dim() == 1:
        b = b.view(1, 4)
    b[..., 0] /= float(img_width)
    b[..., 1] /= float(img_height)
    b[..., 2] /= float(img_width)
    b[..., 3] /= float(img_height)
    return b


def qwen_generate(model, processor, messages, max_new_tokens=128):
    """Qwen3-VL inference: input messages, return generated text."""
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, generated_ids)]
    text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return text


# ========================= Algorithm Steps =========================

def compute_change_mask(img_s, img_t):
    """
    Compute image difference D = |Ie - I|, convert to grayscale and binarize to get change mask Md.
    Returns diff_mask (numpy uint8) and the bounding box (x1,y1,x2,y2) of the changed region,
    or None if no change is detected.
    """
    diff = cv2.absdiff(img_s, img_t)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    _, diff_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    ys, xs = np.where(diff_mask > 0)
    if len(xs) == 0:
        return None, None

    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())
    return diff_mask, (x1, y1, x2, y2)


def sam3_segment(sam_processor, pil_t, bbox, img_width, img_height):
    """
    Feed the changed-region bbox as a positive box prompt to SAM3 and obtain the precise segmentation mask Ms.
    Returns mask (numpy), or None on failure.
    """
    x1, y1, x2, y2 = bbox
    box_xywh = torch.tensor([[x1, y1, x2 - x1, y2 - y1]], dtype=torch.float32)
    box_cxcywh = box_xywh_to_cxcywh(box_xywh)
    norm_box = normalize_bbox_cxcywh(box_cxcywh, img_width, img_height).flatten().tolist()

    inference_state = sam_processor.set_image(pil_t)
    sam_processor.reset_all_prompts(inference_state)
    inference_state = sam_processor.add_geometric_prompt(
        state=inference_state, box=norm_box, label=True
    )

    try:
        mask = inference_state["masks"][0][0]
    except (KeyError, IndexError):
        return None

    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    return mask


def generate_highlight_image(pil_t, mask, alpha=0.4):
    """
    Generate a red-highlighted image Ih on the edited image Ie using the segmentation mask Ms.
    """
    img = np.array(pil_t)
    overlay = img.copy()
    overlay[mask > 0] = [255, 0, 0]
    highlight = (img * (1 - alpha) + overlay * alpha).astype(np.uint8)
    return Image.fromarray(highlight)


def identify_region(qwen_model, qwen_proc, highlight_pil):
    """
    Step 1: 输入高亮图 Ih 到 Qwen3-VL，提取目标物体描述 d。
    """
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": (
                    "Identify the red-highlighted region in the image. "
                    "Return only the object or surface name. No explanation."
                ),
            }],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Identify the red-highlighted region."},
                {"type": "image", "image": highlight_pil},
            ],
        },
    ]
    return qwen_generate(qwen_model, qwen_proc, messages, max_new_tokens=128)


def refine_instruction(qwen_model, qwen_proc, past_pil, original_instruction, region_name):
    """
    Step 2: 输入 (I, P0, d) 到 Qwen3-VL，生成精炼指令 P（≤25 词）。
    """
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": (
                    "You rewrite image-editing instructions. "
                    "Use the original instruction and the provided target region name. "
                    "Revise the instruction concisely (max 25 words). "
                    "Keep meaning consistent with the original intent. "
                    "Do not add new content. "
                    "Output only the revised instruction."
                ),
            }],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Original image:"},
                {"type": "image", "image": past_pil},
                {"type": "text", "text": "Original instruction:"},
                {"type": "text", "text": original_instruction},
                {"type": "text", "text": "Target region name:"},
                {"type": "text", "text": region_name},
                {"type": "text", "text": "Revise the instruction accordingly."},
            ],
        },
    ]
    return qwen_generate(qwen_model, qwen_proc, messages, max_new_tokens=128)


def evaluate_appearance(qwen_model, qwen_proc, past_pil, new_pil):
    """
    Step 3: Input (I, Ie) to Qwen3-VL and evaluate appearance change plausibility score s in [0, 10].
    A higher score indicates a more visible and plausible texture change.
    """
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": (
                    "You are an image quality evaluator for texture editing. "
                    "Compare the two images: the first is the original, the second is the edited version. "
                    "Rate the texture change on a scale of 0-10: "
                    "0 means no visible change or implausible appearance; "
                    "10 means clearly visible, realistic and plausible texture change. "
                    "Output ONLY a single integer score, nothing else."
                ),
            }],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Original image:"},
                {"type": "image", "image": past_pil},
                {"type": "text", "text": "Edited image:"},
                {"type": "image", "image": new_pil},
                {"type": "text", "text": "Rate the texture change quality (0-10):"},
            ],
        },
    ]
    raw = qwen_generate(qwen_model, qwen_proc, messages, max_new_tokens=16)
    # Extract the first integer from the output as the score
    match = re.search(r"\d+", raw)
    if match:
        return int(match.group())
    return 0  # Treat parsing failure as unqualified


# ========================= Argument Parsing =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified pipeline: SAM3 highlight + Qwen3-VL instruction refinement + quality filtering."
    )
    parser.add_argument("--gpu", type=int, nargs='+', default=[0],
                        help="GPU device ID(s). Pass multiple IDs to auto-parallelize, e.g. --gpu 0 1 2 3.")
    parser.add_argument("--json_file", type=str, required=True, help="Path to the input JSON data file.")
    parser.add_argument("--dest_dir", type=str, required=True, help="Output result directory.")
    parser.add_argument("--qwen_model", type=str,
                        default="/mmu-vcg/zhongweizhi/Pretrained_ckp/Qwen3-VL-32B-Instruct",
                        help="Path to the Qwen3-VL model.")
    parser.add_argument("--score_threshold", type=int, default=3,
                        help="Appearance evaluation score threshold theta; samples below this score will be discarded.")
    # Internal args used by child subprocesses (not intended for direct user use)
    parser.add_argument("--num_splits", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--split_index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint_path", type=str, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


# ========================= Main Pipeline =========================

def run_pipeline(args, gpu_id, num_splits, split_index, output_json_path, _checkpoint_path):
    """Run the full pipeline on a single GPU for the assigned data split."""
    # Always use cuda:0 since CUDA_VISIBLE_DEVICES is set to restrict to one GPU
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"[GPU {gpu_id}] Using device: {device}")

    # ---------- Load SAM3 model ----------
    print(f"[GPU {gpu_id}] Loading SAM3 model...")
    print(_checkpoint_path)
    sam_model = build_sam3_image_model(checkpoint_path=_checkpoint_path).to(device)
    sam_proc = Sam3Processor(sam_model, confidence_threshold=0.5)

    # ---------- Load Qwen3-VL model ----------
    print(f"[GPU {gpu_id}] Loading Qwen3-VL model...")
    qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.qwen_model,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": device_str},
    )
    qwen_proc = AutoProcessor.from_pretrained(args.qwen_model)

    # ---------- Read data & split ----------
    with open(args.json_file, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    splits = np.array_split(all_data, num_splits)
    current_data = splits[split_index].tolist()

    print(f"[GPU {gpu_id}] Split {split_index + 1}/{num_splits}: "
          f"processing {len(current_data)}/{len(all_data)} samples. "
          f"Score threshold = {args.score_threshold}")

    # ---------- Process each sample ----------
    accepted_samples = []
    discarded_count = 0

    for i, item in enumerate(current_data):
        s_img_path = item["s_img"]
        t_img_path = item["t_img"]
        original_instruction = item.get("instruction", "")
        idx = item.get("idx", i)

        print(f"[GPU {gpu_id}][{i + 1}/{len(current_data)}] Processing sample idx={idx} ...")

        # Read original image I and edited image Ie
        img_s_bgr = cv2.imread(s_img_path)
        img_t_bgr = cv2.imread(t_img_path)
        if img_s_bgr is None or img_t_bgr is None:
            print(f"  Skipped: unable to read image s_img={s_img_path} or t_img={t_img_path}")
            continue

        img_s = img_s_bgr[:, :, ::-1]  # BGR -> RGB
        img_t = img_t_bgr[:, :, ::-1]
        height, width = img_t.shape[:2]

        past_pil = Image.fromarray(img_s)
        new_pil = Image.fromarray(img_t)

        # ---- Step 1: Compute image difference D = |Ie - I|, binarize to get change mask Md ----
        diff_mask, bbox = compute_change_mask(img_s, img_t)
        if diff_mask is None:
            print(f"  Skipped: no changed region detected.")
            continue

        # ---- Step 2: Feed Ie and Md to SAM3 for segmentation, obtain precise mask Ms ----
        mask = sam3_segment(sam_proc, new_pil, bbox, width, height)
        if mask is None:
            print(f"  Skipped: SAM3 segmentation failed.")
            continue

        # ---- Step 3: Generate red-highlighted image Ih on Ie using mask Ms ----
        highlight_pil = generate_highlight_image(new_pil, mask)

        # ---- Step 4: Input Ih to Qwen3-VL, extract target object description d ----
        region_name = identify_region(qwen_model, qwen_proc, highlight_pil)
        print(f"  Target region: {region_name}")

        # ---- Step 5: Input (I, P0, d) to Qwen3-VL, generate refined instruction P ----
        revised_instruction = refine_instruction(
            qwen_model, qwen_proc, past_pil, original_instruction, region_name
        )
        print(f"  Refined instruction: {revised_instruction}")

        # ---- Step 6: Input (I, Ie) to Qwen3-VL, evaluate appearance difference score s ----
        score = evaluate_appearance(qwen_model, qwen_proc, past_pil, new_pil)
        print(f"  Appearance score: {score}/10")

        # ---- Step 7: Filter by threshold theta ----
        if score < args.score_threshold:
            print(f"  Discarded: score {score} < threshold {args.score_threshold}")
            discarded_count += 1
            continue

        # Accept sample
        accepted_samples.append({
            "idx": idx,
            "s_img": s_img_path,
            "t_img": t_img_path,
            "object": region_name,
            "past_instruction": original_instruction,
            "instruction": revised_instruction,
            "score": score,
        })
        print(f"  Accepted (total {len(accepted_samples)})")

    # ---------- Save results ----------
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(accepted_samples, f, ensure_ascii=False, indent=2)

    print(f"[GPU {gpu_id}] Done. Accepted: {len(accepted_samples)}, Discarded: {discarded_count}. "
          f"Saved to: {output_json_path}")


def main():
    args = parse_args()
    gpu_ids = args.gpu
    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # ---- Child subprocess mode (invoked internally) ----
    if args.num_splits is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        partial_path = dest_dir / f"partial_{args.split_index}.json"
        run_pipeline(args, gpu_ids[0], args.num_splits, args.split_index, partial_path, args.checkpoint_path)
        return

    # ---- Single GPU: run directly ----
    if len(gpu_ids) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        run_pipeline(args, gpu_ids[0], 1, 0, dest_dir / "result.json")
        return

    # ---- Multi-GPU: spawn one subprocess per GPU, then merge ----
    import subprocess
    num_splits = len(gpu_ids)
    print(f"Launching {num_splits} subprocesses for GPUs: {gpu_ids}")

    procs = []
    for i, gpu_id in enumerate(gpu_ids):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--gpu", str(gpu_id),
            "--num_splits", str(num_splits),
            "--split_index", str(i),
            "--json_file", args.json_file,
            "--dest_dir", args.dest_dir,
            "--qwen_model", args.qwen_model,
            "--score_threshold", str(args.score_threshold),
            "--checkpoint_path", args.checkpoint_path
        ]
        procs.append(subprocess.Popen(cmd, env=env))

    for p in procs:
        p.wait()

    # ---- Merge partial results into one final JSON ----
    merged = []
    for i in range(num_splits):
        partial = dest_dir / f"partial_{i}.json"
        if partial.exists():
            with open(partial, encoding="utf-8") as f:
                merged.extend(json.load(f))
            partial.unlink()

    output_json_path = dest_dir / "result.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\n========== All GPUs Done ==========")
    print(f"Total accepted: {len(merged)}")
    print(f"Results saved to: {output_json_path}")


if __name__ == "__main__":
    main()
