import json
import os
import argparse
import numpy as np
import cv2
from PIL import Image
import torch


def mask_to_bool_numpy(mask, ref_shape=None):
    if _is_empty_mask(mask):
        if ref_shape is None:
            return np.zeros((0, 0), dtype=bool)
        return np.zeros(ref_shape, dtype=bool)
    try:
        arr = mask[0, 0].cpu().numpy().astype(bool)
        if ref_shape is not None and arr.shape != ref_shape:
            h, w = ref_shape
            arr = cv2.resize(
                arr.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        return arr
    except Exception:
        if ref_shape is None:
            return np.zeros((0, 0), dtype=bool)
        return np.zeros(ref_shape, dtype=bool)

def mask_iou(mask1, mask2):
    ref_shape = None
    ref_shape = (mask1.shape[2], mask1.shape[3])

    m1 = mask_to_bool_numpy(mask1, ref_shape)
    m2 = mask_to_bool_numpy(mask2, ref_shape)

    if m1.size == 0 and m2.size == 0:
        return 0.0

    if m1.size == 0:
        m1 = np.zeros_like(m2)
    if m2.size == 0:
        m2 = np.zeros_like(m1)

    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return 0.0 if union == 0 else inter / union

import numpy as np
import cv2

def edge_iou_tolerant(edge1, edge2, thresh=64, radius=1, eps=1e-6):
    """
    Edge IoU with spatial tolerance

    edge1, edge2: (H, W), uint8, [0,255]
    radius: 容忍半径（像素）
    """
    assert edge1.shape == edge2.shape

    # 1. 二值化
    mask1 = (edge1 >= thresh).astype(np.uint8)
    mask2 = (edge2 >= thresh).astype(np.uint8)

    # 2. 膨胀核
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius + 1, 2 * radius + 1)
    )

    mask1_d = cv2.dilate(mask1, kernel)
    mask2_d = cv2.dilate(mask2, kernel)

    # 3. 容忍交集
    intersection = np.logical_and(mask1, mask2_d).sum() + \
                   np.logical_and(mask2, mask1_d).sum()

    # 4. 并集
    union = mask1.sum() + mask2.sum()

    return intersection / (union + eps)


def edge_iou(edge1, edge2, thresh=128, eps=1e-6):
    """
    edge1, edge2: (H, W), uint8, [0,255]
    thresh: 二值化阈值
    """
    assert edge1.shape == edge2.shape

    # 1. 二值化
    mask1 = edge1 >= thresh
    mask2 = edge2 >= thresh

    # 2. 交并集
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    # 3. IoU
    iou = intersection / (union + eps)
    return iou

def mask_iou_simple(mask1, mask2):
    # 假定输入是numpy数组
    ref_shape = None
    ref_shape = (mask1.shape[0], mask1.shape[1])

    if mask1.size == 0 and mask2.size == 0:
        return 0.0

    m1 = mask1.astype(bool)
    m2 = mask2.astype(bool)

    if m1.size == 0:
        m1 = np.zeros_like(m2)
    if m2.size == 0:
        m2 = np.zeros_like(m1)

    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return 0.0 if union == 0 else inter / union
    
def _is_empty_mask(mask):
    if mask is None:
        return True
    try:
        return mask.shape[0] == 0 or mask.numel() == 0
    except Exception:
        return True


def merge_masks(mask):
    if mask is None:
        return None
    try:
        if mask.shape[0] == 0:
            return None
        merged = mask.bool().any(dim=0, keepdim=True)
        return merged.to(mask.dtype)
    except Exception:
        return None

def visualize_mask_overlap(mask1, mask2, ref_img, alpha=0.5):
    img = np.array(ref_img).copy()
    h, w = img.shape[:2]

    m1 = mask_to_bool_numpy(mask1, (h, w))
    m2 = mask_to_bool_numpy(mask2, (h, w))

    only1 = m1 & (~m2)
    only2 = m2 & (~m1)
    inter = m1 & m2

    overlay = np.zeros_like(img)
    overlay[only1] = (255, 0, 0)      # red
    overlay[only2] = (0, 0, 255)      # blue
    overlay[inter] = (255, 0, 255)    # purple

    vis = img.copy()
    mask_any = only1 | only2 | inter
    vis[mask_any] = (
        img[mask_any] * (1 - alpha) + overlay[mask_any] * alpha
    ).astype(np.uint8)

    return vis