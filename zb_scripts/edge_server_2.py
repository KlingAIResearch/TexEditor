import asyncio
import time
from typing import List

from skimage.metrics import structural_similarity as ssim


import numpy as np
import cv2
from PIL import Image
import torch
from fastapi import FastAPI
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

# =====================================================
# Utils
# =====================================================

def b64_to_pil(b64_str: str) -> Image.Image:
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_bytes))
    return img.convert("RGB")

def _is_empty_mask(mask):
    return mask is None or not hasattr(mask, "shape") or mask.shape[0] == 0


def mask_to_bool_numpy(mask, ref_shape=None):
    if _is_empty_mask(mask):
        if ref_shape is None:
            return np.zeros((0, 0), dtype=bool)
        return np.zeros(ref_shape, dtype=bool)

    arr = mask[0, 0].detach().cpu().numpy().astype(bool)
    if ref_shape is not None and arr.shape != ref_shape:
        h, w = ref_shape
        arr = cv2.resize(
            arr.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return arr


# =====================================================
# Model Init (GLOBAL, READ-ONLY)
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

# =====================================================
# FastAPI App
# =====================================================

app = FastAPI()

# request_queue item:
# (samples: List[RewardSample], future)
request_queue: asyncio.Queue = asyncio.Queue()


# =====================================================
# Request / Response Schema
# =====================================================

class RewardSample(BaseModel):
    image1_b64: str
    image2_b64: str
    prompt: str
    kind: str


class RewardRequest(BaseModel):
    samples: List[RewardSample]


class RewardResult(BaseModel):
    iou: float
    image_edge1_b64: str = ""
    image_edge2_b64: str = ""


class RewardResponse(BaseModel):
    results: List[RewardResult]

# 版本更新 我觉得 还是需要把 图也传回来 ！！！！ 2026.1.3 


# =====================================================
# HTTP Endpoint
# =====================================================

@app.post("/reward", response_model=RewardResponse)
async def reward(req: RewardRequest):
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    await request_queue.put((req.samples, fut))
    return await fut


# =====================================================
# Serial GPU Worker
# =====================================================

def process_texture_s(s):
    if s < 0.7:
        s = -0.2
    else:
        if s > 0.9 :
            s = 1
        else:
            s = (s - 0.7) / 0.2
    return s

def process_more_s(s):
    if s < 0.8:
        s = -0.2
    else:
        if s > 0.95 :
            s = 1
        else:
            s = (s - 0.8) / 0.15
    return s

async def serial_worker():
    print("[RewardServer] Serial SAM3 worker started")
    cur_device = torch.cuda.current_device()
    print(
        f"[RewardServer] current_device={cur_device}, "
        f"name={torch.cuda.get_device_name(cur_device)}"
    )


    # IMPORTANT: processor must be local & private
    # processor = Sam3Processor(MODEL)
    ddd, rank = get_local_device()

    while True:
        samples, fut = await request_queue.get()
        results: List[RewardResult] = []

        try:
            with torch.no_grad():
                for sample in samples:


                    img1 = b64_to_pil(sample.image1_b64)
                    img2 = b64_to_pil(sample.image2_b64)
                    prompt = sample.prompt
                    cur_kind = sample.kind

                    img_t = transforms.ToTensor()(img1).unsqueeze(0).to(ddd)
                    with torch.no_grad():
                        multi_outputs, outputs, _, _= model(img_t)
                    img_edge, img_edge_pil = post_process_edge(multi_outputs)

                    ref_img_t = transforms.ToTensor()(img2).unsqueeze(0).to(ddd)
                    with torch.no_grad():
                        multi_outputs2, outputs, _, _= model(ref_img_t)
                    ref_img_edge, ref_img_edge_pil = post_process_edge(multi_outputs2)

                    score = ssim(
                    img_edge,
                    ref_img_edge,
                    data_range=1.0
                    )

                    # 这里写一个 加工处理逻辑!!!
                    if cur_kind == 'texture':
                            score = process_texture_s(score)
                    else:
                        if cur_kind == 'more':
                            score = process_more_s(score)
                        else:
                            score = score

                    edge_img1_b64 = pil2b64(img_edge_pil, format="PNG", quality=100)
                    edge_img2_b64 = pil2b64(ref_img_edge_pil, format="PNG", quality=100)

                    print(f"[RewardServer] Computed edge ssim : {score:.4f}")
                    results.append(RewardResult(iou=score, image_edge1_b64=edge_img1_b64, image_edge2_b64=edge_img2_b64))
                    # print(edge_img1_b64)

            if not fut.done():
                fut.set_result(RewardResponse(results=results))

        except Exception as e:
            if not fut.done():
                fut.set_exception(e)


# =====================================================
# Startup
# =====================================================

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(serial_worker())


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    import  argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9002, help='local rank for DistributedDataParallel')
    args = parser.parse_args()

    uvicorn.run(
        "edge_server_2:app",
        host="0.0.0.0",
        port=args.port,
        workers=1,   # 必须是 1
    )
