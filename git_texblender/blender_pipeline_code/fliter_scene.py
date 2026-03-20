import json
import random
from typing import Any, Dict, List, Tuple
import os
import sys


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)
    print(f"保存：{path}")


def build_furniture_map(furniture_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    m = {}
    for f in furniture_list:
        for key_field in ("uid", "jid", "aid"):
            val = f.get(key_field)
            if not val:
                continue
            if isinstance(val, list):
                for vv in val:
                    s = str(vv)
                    m[s] = f
                    if "/" in s:
                        m[s.split("/")[-1]] = f
            else:
                s = str(val)
                m[s] = f
                if "/" in s:
                    m[s.split("/")[-1]] = f
    return m


def find_instance_nodes(obj: Any, parent=None, parent_key=None) -> List[Tuple[List, int, Dict]]:
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            found.extend(find_instance_nodes(v, obj, k))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, dict) and "ref" in item and "pos" in item:
                found.append((obj, i, item))
            else:
                found.extend(find_instance_nodes(item, obj, i))
    return found


def compute_aabb_from_instance(inst: Dict, furn: Dict) -> Dict[str, float]:
    pos = inst.get("pos")
    scale = inst.get("scale", [1, 1, 1])
    size = furn.get("size")
    if pos is None or size is None:
        return None

    sx, sy, sz = (scale + [1, 1, 1])[:3]
    w, d, h = (size + [0, 0, 0])[:3]

    w_s = w * sx
    d_s = d * sz
    h_s = h * sy

    x, y, z = pos

    return {
        "xmin": x - w_s / 2.0,
        "xmax": x + w_s / 2.0,
        "ymin": y - h_s / 2.0,
        "ymax": y + h_s / 2.0,
        "zmin": z - d_s / 2.0,
        "zmax": z + d_s / 2.0,
    }


def aabb_overlap(a: Dict[str, float], b: Dict[str, float]) -> bool:
    return not (
        a["xmax"] < b["xmin"] or a["xmin"] > b["xmax"] or
        a["ymax"] < b["ymin"] or a["ymin"] > b["ymax"] or
        a["zmax"] < b["zmin"] or a["zmin"] > b["zmax"]
    )


# --------------------
#  修复版的 clean：按实例对象删除，而不是按 ref
# --------------------
def clean_scene_random_remove(input_path: str, output_path: str = None):

    scene = load_json(input_path)

    furniture_list = scene.get("furniture", [])
    furn_map = build_furniture_map(furniture_list)

    nodes = find_instance_nodes(scene)

    boxes = []
    inst_nodes = []  # (parent, idx, inst)
    for parent, idx, inst in nodes:
        ref = inst.get("ref")

        furn = furn_map.get(str(ref))
        if furn is None and isinstance(ref, str) and "/" in ref:
            furn = furn_map.get(ref.split("/")[-1])
        if furn is None:
            continue

        aabb = compute_aabb_from_instance(inst, furn)
        if aabb is None:
            continue

        boxes.append(aabb)
        inst_nodes.append((parent, idx, inst))
    # 改为按实例对象 id 删除
    removed_inst_ids = set()

    for i in range(len(inst_nodes)):
        if id(inst_nodes[i][2]) in removed_inst_ids:
            continue
        for j in range(i + 1, len(inst_nodes)):
            if id(inst_nodes[j][2]) in removed_inst_ids:
                continue
            if aabb_overlap(boxes[i], boxes[j]):
                victim = random.choice([i, j])
                removed_inst_ids.add(id(inst_nodes[victim][2]))

    # 过滤掉被标记删除的实例
    def filter_lists(obj):
        if isinstance(obj, dict):
            return {k: filter_lists(v) for k, v in obj.items()}
        if isinstance(obj, list):
            new_list = []
            for item in obj:
                if isinstance(item, dict) and "ref" in item and "pos" in item:
                    # 如果这个实例被删除则跳过
                    if id(item) in removed_inst_ids:
                        continue
                new_list.append(filter_lists(item))
            return new_list
        return obj

    new_scene = filter_lists(scene)

    if output_path:
        save_json(new_scene, output_path)