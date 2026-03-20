import sys
sys.stdout.reconfigure(line_buffering=True)
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import os
import json
import torch
import re
import subprocess
from PIL import Image
import argparse
import numpy as np


# 1. Roughness-related
roughness_words = ["rough", "grainy", "gritty", "textured", "coarse", "scratched", "worn", "dusty", "matte", "smooth", "polished", "glossy", "shiny", "reflective", "sleek"]
roughness_nouns = ["roughness", "texture", "scratches", "dust", "matte surface", "smooth surface", "gloss", "shine", "reflection", "polished surface"]
roughness = roughness_words+roughness_nouns

# 2. Metalness-related
metalness_words = ["metallic", "metal-like", "chrome", "steel", "copper", "silver", "gold", "shiny metal", "non-metal", "plastic", "ceramic", "rubber", "wood", "stone"]
metalness_nouns = ["metal", "chrome surface", "steel surface", "copper surface", "silver finish", "gold finish", "plastic surface", "ceramic surface", "rubber surface", "wood surface"]
metalness = metalness_words+metalness_nouns

# 3. Alpha-related
alpha_words = ["transparent", "translucent", "semi-transparent", "clear", "frosted", "opaque", "solid", "matte"]
alpha_nouns = ["transparency", "translucency", "clear glass", "frosted glass", "opacity", "solid surface", "matte finish"]
alpha = alpha_words+alpha_nouns

danci_list = [alpha, metalness, roughness]


# --- 1. Argument parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Generate image editing commands using Qwen3-VL model.")

    # Launcher mode: specify --gpus to auto-spawn parallel subprocesses, no external bash script needed
    parser.add_argument('--gpus', type=int, nargs='+', default=None, help='Launcher mode: GPU IDs to use (e.g. --gpus 0 1 2 3). When set, the script spawns one subprocess per GPU.')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for log files in launcher mode.')

    parser.add_argument('--gpu', type=int, default=0, help='Subprocess mode: GPU device ID for the current process.')
    parser.add_argument('--num_splits', type=int, default=1, help='Total number of data splits.')
    parser.add_argument('--split_index', type=int, default=0, help='Index of the split this process handles (0-based).')

    parser.add_argument('--model_path', type=str, default='/mmu-vcg/zhongweizhi/Pretrained_ckp/Qwen3-VL-32B-Instruct', help='Path to pretrained model.')
    # base_dir points to the output root; structure: {uuid}/{group_strategy}/
    parser.add_argument('--base_dir', type=str, default='/mmu-vcg/zb08/yihang/git_texblender/output', help='Input data root directory. Expected structure: {uuid}/{group_strategy}/.')
    parser.add_argument('--dest_dir', type=str, default='./infer_results', help='Output directory for results.')

    return parser.parse_args()


# --- Launcher mode: auto-spawn parallel subprocesses ---
def launch_parallel(args):
    gpu_list = args.gpus
    num_instances = len(gpu_list)
    os.makedirs(args.log_dir, exist_ok=True)

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    script_path = os.path.abspath(__file__)
    procs = []
    for split_index, gpu_id in enumerate(gpu_list):
        log_path = os.path.join(args.log_dir, f"infer_all_{timestamp}_log_{split_index}.out")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = [
            sys.executable, script_path,
            "--gpu", "0",
            "--num_splits", str(num_instances),
            "--split_index", str(split_index),
            "--model_path", args.model_path,
            "--base_dir", args.base_dir,
            "--dest_dir", args.dest_dir,
        ]
        print(f"Launching worker {split_index} (CUDA_VISIBLE_DEVICES={gpu_id}), log: {log_path}")
        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=log_f)
        procs.append(proc)

    print(f"\nAll {num_instances} workers launched. Waiting for completion...")
    for p in procs:
        p.wait()
    print("✅ All workers finished.")

    # Merge all split results into a single JSON
    merged = []
    split_files = []
    for split_index in range(num_instances):
        split_path = os.path.join(args.dest_dir, f"split_{split_index}_of_{num_instances}.json")
        if os.path.exists(split_path):
            with open(split_path, "r", encoding="utf-8") as f:
                merged.extend(json.load(f))
            split_files.append(split_path)
    merged_path = os.path.join(args.dest_dir, "all.json")
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    for sp in split_files:
        os.remove(sp)
    print(f"✅ {len(merged)} entries merged into: {merged_path}")


# --- 2. Main logic ---
def main():
    args = parse_args()

    # Launcher mode: spawn subprocesses when --gpus is provided
    if args.gpus:
        launch_parallel(args)
        return

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device} (GPU ID: {args.gpu})")

    base_dir = args.base_dir
    dest_dir = args.dest_dir
    os.makedirs(dest_dir, exist_ok=True)

    # --- 3. Load model ---
    pp = args.model_path
    print(f"⏳ Loading model: {pp}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        pp,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
        device_map={"": f"cuda:{args.gpu}"},
    )
    processor = AutoProcessor.from_pretrained(pp)
    print("✅ Model loaded.")

    # --- 4. Collect all leaf folders (format: {uuid}/{group_strategy}/) ---
    all_leaf_folders = []
    for uuid_name in os.listdir(base_dir):
        uuid_path = os.path.join(base_dir, uuid_name)
        if not os.path.isdir(uuid_path):
            continue
        for group_strategy in os.listdir(uuid_path):
            leaf_path = os.path.join(uuid_path, group_strategy)
            if os.path.isdir(leaf_path):
                all_leaf_folders.append(leaf_path)
    all_leaf_folders = sorted(all_leaf_folders)

    # --- 5. Data splitting ---
    total_folders = len(all_leaf_folders)
    splits = np.array_split(all_leaf_folders, args.num_splits)

    if args.split_index >= len(splits) or args.split_index < 0:
        print(f"❌ Error: split_index ({args.split_index}) out of range (0 to {args.num_splits - 1}).")
        return

    current_folders = splits[args.split_index].tolist()

    print(f"\n--- Split info ---")
    print(f"Total leaf folders: {total_folders}")
    print(f"Num splits: {args.num_splits}, current index: {args.split_index}")
    print(f"Processing {len(current_folders)} folders in this batch.")
    print("------------------\n")

    all_list = []
    attributes = ["alpha", "metalness", "roughness"]

    # --- 6. Iterate over current batch ---
    for i, folder_path in enumerate(current_folders):
        print(f"[{i+1}/{len(current_folders)}] Processing: {folder_path}")

        past_path = os.path.join(folder_path, "past.png")
        if not os.path.exists(past_path):
            print(f"   Skip: {past_path} not found.")
            continue

        try:
            past_image = Image.open(past_path).convert("RGB")
        except Exception as e:
            print(f"   Error: cannot load {past_path}. {e}")
            continue

        # ===== 6a. Attribute changes (alpha / metalness / roughness) =====
        for idx, attr in enumerate(attributes):
            cur_ciku = danci_list[idx]

            attr_folder = os.path.join(folder_path, attr)
            attr_new_path = os.path.join(attr_folder, "render.png")
            json_p = os.path.join(attr_folder, "variation_info.json")

            if not (os.path.isdir(attr_folder) and os.path.exists(attr_new_path) and os.path.exists(json_p)):
                continue

            try:
                attr_new_image = Image.open(attr_new_path).convert("RGB")
                with open(json_p, "r", encoding="utf-8") as f:
                    variation_info = json.load(f)
                    direction = variation_info['changes'][0]['direction']
                    object_name = variation_info['changes'][0]['object']
                    object_name = object_name.split('.')[0]
            except Exception as e:
                print(f"   Error: cannot load {attr_new_path} or {json_p}. {e}")
                continue

            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Act as a professional image editing command generator. Output only executable editing steps. Do not include any image content descriptions or comparative explanations. Generate corresponding editing commands based on the original image, edited image and supplementary information I provide. Prioritize concise and natural imperative sentences."
                            )
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Original image:"},
                        {"type": "image", "image": past_image},
                        {"type": "text", "text": "New image:"},
                        {"type": "image", "image": attr_new_image},
                        {"type": "text", "text": f"The object to be edited is approximately {object_name}, but please refer to the image and use the name that human use and better for giving the correspoding postion description. Output the corresponding concise editing instructions without any extra explanation. You can use the item in the following word list to help you describe the texture change: {', '.join(cur_ciku)}. The change direction is {direction}.If the difference is too minor, just output bad pair."}
                    ]
                }
            ]

            try:
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(model.device)

                generated_ids = model.generate(**inputs, max_new_tokens=128)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip()

                print(f"   -> {attr}: {output_text}")
                cur_dict = {}
                cur_dict['s_img'] = past_path
                cur_dict['t_img'] = attr_new_path
                cur_dict['instruction'] = output_text
                all_list.append(cur_dict)

            except Exception as e:
                print(f"   ❌ Generation failed for {attr}: {e}")
                continue

        # ===== 6b. Texture changes (sample_*.png) =====
        cur_names = os.listdir(folder_path)
        texture_names = [s for s in cur_names if s.startswith("sample") and s.endswith(".png")]

        for file_name_texture in texture_names:
            tt_img_p = os.path.join(folder_path, file_name_texture)
            tt_json_p = os.path.join(folder_path, file_name_texture.replace(".png", ".json"))

            if not os.path.exists(tt_json_p):
                print(f"   Skip texture {file_name_texture}: missing JSON.")
                continue

            try:
                tt_img = Image.open(tt_img_p).convert("RGB")
                with open(tt_json_p, "r", encoding="utf-8") as f:
                    tt_info = json.load(f)

                tt_name = re.sub(r" +", " ",
                    re.sub(r"(?<![A-Za-z])[A-Za-z](?![A-Za-z])", "",
                           re.sub(r"\d+", "", tt_info['texture_metadata']['category']))
                ).strip()
                tt_des = re.sub(r" +", " ",
                    re.sub(r"(?<![A-Za-z])[A-Za-z](?![A-Za-z])", "",
                           re.sub(r"\d+", "", tt_info['texture_metadata']['name']))
                ).strip()
            except Exception as e:
                print(f"   Error: cannot load texture file {file_name_texture}. {e}")
                continue

            print("Processing:", tt_info['texture_metadata']['category']+"  ", tt_name+"  ", tt_des+"  ")

            texture_messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Act as a professional image editing command generator. Output only executable editing steps. Do not include any image content descriptions or comparative explanations. Generate corresponding editing texture commands based on the original image, edited image and supplementary information I provide. Prioritize concise and natural imperative sentences."
                            )
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Original image:"},
                        {"type": "image", "image": past_image},
                        {"type": "text", "text": "Edited image:"},
                        {"type": "image", "image": tt_img},
                        {"type": "text", "text": "Output the corresponding edit prompt. The target texture description is " + tt_des + "." + f" The final material editing instruction must include a word meaningly same as {tt_name}. If the visual quality of the edited image is too bad and the texture change is not obvious or hard to describe, just output bad pair."}
                    ]
                }
            ]

            try:
                inputs = processor.apply_chat_template(
                    texture_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(model.device)

                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=128)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                texture_output = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip()

                print("Texture Edit Instruction:", texture_output)
            except Exception as e:
                print(f"   ❌ Generation failed for texture {file_name_texture}: {e}")
                continue

            revise_messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Act as a professional image editing command reviser. Your task is revising the editing instruction I provide based on the original image and edited image. If the visual quality of the edited image is too bad and the texture change is not obvious or hard to describe, just output bad pair. And make the final instruction concise and no more than 25 word."
                            )
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Original image:"},
                        {"type": "image", "image": past_image},
                        {"type": "text", "text": "Edited image:"},
                        {"type": "image", "image": tt_img},
                        {"type": "text", "text": "Edited insturction:"},
                        {"type": "text", "text": texture_output},
                    ]
                }
            ]

            try:
                inputs = processor.apply_chat_template(
                    revise_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(model.device)

                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=128)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                revised_out = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip()

                print("revised_out:", revised_out)
                print("___________________________________")
            except Exception as e:
                print(f"   ❌ Revision failed for texture {file_name_texture}: {e}")
                continue

            cur_dict = {}
            cur_dict['s_img'] = past_path
            cur_dict['t_img'] = tt_img_p
            cur_dict['instruction'] = revised_out
            all_list.append(cur_dict)

    # --- 7. Save results ---
    output_path = os.path.join(dest_dir, f"split_{args.split_index}_of_{args.num_splits}.json")

    print(f"\n--- Saving results ---")
    print(f"{len(all_list)} entries, saving to: {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_list, f, ensure_ascii=False, indent=2)

    print("✅ Done.")


if __name__ == "__main__":
    main()