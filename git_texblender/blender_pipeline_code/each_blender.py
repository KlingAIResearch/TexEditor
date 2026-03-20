import blenderproc as bproc
import argparse
import os
import warnings
import re
import random
import bpy
import copy
import numpy as np
import h5py
import imageio
import json
from pathlib import Path
warnings.filterwarnings("ignore")
def parse_args():
    parser = argparse.ArgumentParser(description="Texture editing with BlenderProc.")
    
    # 保持原有的路径参数
    parser.add_argument('--room_json_path', type=str, default='c787a5b2-b0ee-4710-b5fe-81b588b05388.json', help='room json file path.')
    parser.add_argument('--max_attempts_per_room', type=int, default=10, help='max attempts to load a furniture group per room.')
    parser.add_argument('--camera_sampling_attempts', type=int, default=1000, help='max attempts to sample a valid camera pose per photo.')
    parser.add_argument('--output_base_path', type=str, default='./output', help='base output path.')
    parser.add_argument('--furniture_base_path', type=str, default="", help='base path for 3D-FUTURE models.')
    parser.add_argument('--texture_base_path', type=str, default="", help='base path for 3D-FRONT textures.')
    parser.add_argument('--new_texture_base_path', type=str, default="", help='base path for new textures.')
    parser.add_argument('--mapping_file', type=str, default="", help='path for texture mapping file.')
    parser.add_argument('--samples', type=int, default=1024, help='number of samples per pixel.')
    parser.add_argument('--light_bounces', type=int, default=8, help='number of light bounces.')
    parser.add_argument('--group_strategy', type=str, choices=['obj', 'obj+texture'], help='strategy to select furniture group.')
    return parser.parse_args()

def check_name(name):
    if not name:
        return False
    for category_name in ["front", "back", "floor", "wall", "ceiling", "baseboard",
                          "slabbottom", "window", "door", "slabtop", "stairs", "pocket",
                          "slabside", "slabbottom", "cabinet", "others"]:
        if category_name in name.lower():
            return False
    return True

def model_key_from_name(name: str) -> str:
    if not name:
        return ""
    n = name.lower()
    if "/" in n:
        return n.split("/")[0].strip()
    n = re.sub(r'[\._-]\d+$', '', n)
    n = n.strip()
    tokens = n.split()
    if len(tokens) >= 2:
        if not tokens[0].isdigit() and not tokens[1].isdigit():
            return (tokens[0] + " " + tokens[1]).strip()
    return tokens[0] if tokens else ""


def change_texture(objs, texture_path):
    for obj in objs:
        if obj.is_hidden():
            continue
        print(texture_path)
        for i, mat in enumerate(obj.get_materials()):
            bpy_mat = mat.blender_obj
            if bpy_mat.users > 1:
                bpy_mat = bpy_mat.copy()
                obj.blender_obj.data.materials[i] = bpy_mat
            nodes = bpy_mat.node_tree.nodes
            links = bpy_mat.node_tree.links
            principled_bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
            if principled_bsdf is None:
                continue
            texture_lis = set(os.listdir(texture_path))
            if "basecolor.png" in texture_lis:
                bproc.material.add_base_color(nodes, links, os.path.join(texture_path, "basecolor.png"), principled_bsdf)
            if "metallic.png" in texture_lis:
                bproc.material.add_metal(nodes, links, os.path.join(texture_path, "metallic.png"), principled_bsdf)
            if "normal.png" in texture_lis:
                bproc.material.add_normal(nodes, links, os.path.join(texture_path, "normal.png"), principled_bsdf, False)
            if "roughness.png" in texture_lis:
                bproc.material.add_roughness(nodes, links, os.path.join(texture_path, "roughness.png"), principled_bsdf)
            if "specular.png" in texture_lis:
                bproc.material.add_specular(nodes, links, os.path.join(texture_path, "specular.png"), principled_bsdf)
            if "opacity.png" in texture_lis:
                bproc.material.add_alpha(nodes, links, os.path.join(texture_path, "opacity.png"), principled_bsdf)

def main_batch():
    args = parse_args()
    output_path = os.path.join(args.output_base_path, os.path.basename(args.room_json_path), args.group_strategy)
    os.makedirs(output_path, exist_ok=True)
    mapping_file = bproc.utility.resolve_resource(args.mapping_file)
    mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)
    bproc.init()
    bproc.renderer.set_max_amount_of_samples(args.samples)
    # 光线反弹设置
    bproc.renderer.set_light_bounces(
        diffuse_bounces=args.light_bounces,
        glossy_bounces=args.light_bounces,
        max_bounces=args.light_bounces,
        transmission_bounces=args.light_bounces,
        transparent_max_bounces=args.light_bounces
    )
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_segmentation_output(map_by=["category_id"])
    bproc.camera.set_resolution(args.samples, args.samples)

    loaded_objects = bproc.loader.load_front3d(
        json_path=args.room_json_path,
        future_model_path=args.furniture_base_path,
        front_3D_texture_path=args.texture_base_path,
        label_mapping=mapping,
    )

    light_zpz = random.uniform(0.6, 1.0)

    for mat in bpy.data.materials:
        if mat.node_tree:
            for node in mat.node_tree.nodes:
                if node.type == "EMISSION":
                    node.inputs["Strength"].default_value *= light_zpz

    mesh_objects = [o for o in loaded_objects if isinstance(o, bproc.types.MeshObject)]
    bvh_tree = bproc.object.create_bvh_tree_multi_objects(mesh_objects)

    special_objects = [obj for obj in mesh_objects if check_name(obj.get_name())]

    if not special_objects:
        print(f"❌ 房间 {args.room_json_path} 没有可替换家具，跳过。")
        return
    groups = {}
    for obj in special_objects:
        mats = obj.get_materials()
        if args.group_strategy == "obj" and mats:
            key = model_key_from_name(obj.get_name())
        else:
            key = model_key_from_name(obj.get_name() + obj.get_materials()[0].get_name().lower())
        groups.setdefault(key, []).append(obj)

    attempt = 0
    success = False
    groups_keys = list(groups.keys())
    random.shuffle(groups_keys)
    while attempt < args.max_attempts_per_room and not success:
        attempt += 1
        selected_key = groups_keys[attempt % len(groups_keys)]
        furniture_group = groups[selected_key]
        idx = random.randrange(len(furniture_group))
        lucky_obj = furniture_group[idx]
        print(f"🎯 尝试 {attempt}/{args.max_attempts_per_room} ：{selected_key} ({len(furniture_group)} 个对象)")

        bb_world = lucky_obj.get_bound_box(local_coords=False)
        object_location = np.mean(bb_world, axis=0)
        object_size = np.max(np.max(bb_world, axis=0) - np.min(bb_world, axis=0))
        radius_min = object_size * 2.0
        radius_max = object_size * 4.0
        proximity_checks = {
            "min": radius_min,  # 放宽最小距离
            "avg": {"min": radius_min * 0.8, "max": radius_max * 1.2},  # 放宽平均距离范围
            "no_background": True  # 不允许背景出现
        }
        cam_counter = 0
        for _ in range(args.camera_sampling_attempts):
            camera_location = bproc.sampler.shell(
                center=object_location,
                radius_min=radius_min,
                radius_max=radius_max,
                elevation_min=0,
                elevation_max=30
            )
            toward_direction = (object_location + np.random.uniform(0, 1, size=3) * object_size * 0.5) - camera_location
            rotation_matrix = bproc.camera.rotation_from_forward_vec(toward_direction, inplane_rot=0)
            cam2world_matrix = bproc.math.build_transformation_mat(camera_location, rotation_matrix)

            obstacle_ok = bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, proximity_checks, bvh_tree)

            if obstacle_ok:
                success = True
                bproc.camera.add_camera_pose(cam2world_matrix)
                cam_counter += 1
                break
        if cam_counter == 0:
            print(f"⚠️ 尝试 {attempt} 未找到可见视角，换下一个物体组。")
            continue
        # 渲染初始图象
        data = bproc.renderer.render()
        bproc.writer.write_hdf5(output_path, data, append_to_existing_output=True)
        with h5py.File(os.path.join(output_path, "0.hdf5"), "r") as f:
            imageio.imwrite(os.path.join(output_path, "past.png"), np.array(f["colors"]).astype(np.uint8))
        for kind in ["roughness", "metalness", "alpha"]:
            variation_info = {"property": kind, "changes": []}
            original_values = []
            # === 第一步：计算平均值 ===
            values = []
            for i, mem in enumerate(furniture_group):
                materials = mem.get_materials()
                if not materials:
                    continue
                mat = materials[0]
                bpy_mat = mat.blender_obj
                nodes = bpy_mat.node_tree.nodes
                principled = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
                if principled is None:
                    continue
                if kind == "roughness":
                    slot = principled.inputs.get("Roughness")
                elif kind == "metalness":
                    slot = principled.inputs.get("Metallic")
                else:
                    slot = principled.inputs.get("Alpha")
                values.append(float(slot.default_value))
            avg_value = np.mean(values) if values else 0.5
            # === 第二步：根据平均值决定方向 ===
            if avg_value < 0.3:
                direction = "increase"
            elif avg_value > 0.7:
                direction = "decrease"
            else:
                direction = random.choice(["increase", "decrease"])

            print(f"\n=== 正在调整 {kind} ({'整体变大' if direction == 'increase' else '整体变小'})，平均值={avg_value:.2f} ===")
            # === 第三步：按统一方向修改 ===
            for i, mem in enumerate(furniture_group):
                materials = mem.get_materials()
                if not materials:
                    continue
                mat = materials[0]
                bpy_mat = mat.blender_obj
                nodes = bpy_mat.node_tree.nodes
                principled = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
                if principled is None:
                    continue
                # 获取输入槽
                if kind == "roughness":
                    description = "更粗糙" if direction == "increase" else "更光滑"
                    slot = principled.inputs.get("Roughness")
                elif kind == "metalness":
                    description = "更金属" if direction == "increase" else "更非金属"
                    slot = principled.inputs.get("Metallic")
                else:
                    description = "更不透明" if direction == "increase" else "更透明"
                    slot = principled.inputs.get("Alpha")
                old_value = values[i]
                if kind == "roughness":
                    new_value = min(old_value + 0.5, 0.99) if direction == "increase" else max(old_value - 0.5, 0.01)
                elif kind == "metalness":
                    new_value = min(old_value + 0.5, 0.99) if direction == "increase" else max(old_value - 0.5, 0.01)
                else:
                    new_value = min(old_value + 0.5, 0.99) if direction == "increase" else max(old_value - 0.5, 0.01)
                # 保存原始值
                original_values.append((slot, old_value))

                # 应用修改
                slot.default_value = new_value
                variation_info["changes"].append({
                    "object": mem.get_name(),
                    "old_value": old_value,
                    "new_value": float(new_value),
                    "direction": direction,
                    "description": description
                })
                print(f"  ➤ {kind}: {old_value:.2f} → {new_value:.2f} ({direction}, {description})")
            data = bproc.renderer.render()
            kind_dir = os.path.join(output_path, kind)
            bproc.writer.write_hdf5(kind_dir, data, append_to_existing_output=True)

            with h5py.File(os.path.join(kind_dir, "0.hdf5"), "r") as f:
                imageio.imwrite(os.path.join(kind_dir, "render.png"),
                                np.array(f["colors"]).astype(np.uint8))

            with open(os.path.join(kind_dir, "variation_info.json"), "w", encoding="utf-8") as f:
                json.dump(variation_info, f, ensure_ascii=False, indent=2)
            for slot, old_value in original_values:
                slot.default_value = old_value
            print(f"✅ 已恢复 {kind} 原始值\n")
        texture_lis = [
            d for d in os.listdir(args.new_texture_base_path)
            if os.path.isdir(os.path.join(args.new_texture_base_path, d))
        ]
        three_texture_lis = random.sample(texture_lis, min(3, len(texture_lis)))
        for i, texture_path in enumerate(three_texture_lis):
            full_texture_path = os.path.join(args.new_texture_base_path, texture_path)
            change_texture(furniture_group, full_texture_path)
            texture_name = os.path.basename(texture_path)
            data = bproc.renderer.render()
            bproc.writer.write_hdf5(output_path, data, append_to_existing_output=True)
            with h5py.File(os.path.join(output_path, f"{1 + i}.hdf5"), "r") as f:
                imageio.imwrite(os.path.join(output_path, f"{texture_name}.png"),
                                np.array(f["colors"]).astype(np.uint8))
            metadata_path = os.path.join(full_texture_path, "metadata.json")
            metadata_info = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as mf:
                    metadata_info = json.load(mf)
            useful_fields = [
                "name", "category", "description", "license", "link", "method",
                "source", "tags", "height_factor", "height_mean", "maps",
                "version_date"
            ]
            metadata_summary = {k: metadata_info.get(k) for k in useful_fields if k in metadata_info}
            # 整合信息
            info = {
                "object_name": lucky_obj.get_name(),
                "texture_path": full_texture_path,
                "group_size": len(furniture_group),
                "group_key": selected_key,
                "texture_metadata": metadata_summary
            }
            with open(os.path.join(output_path, f"{texture_name}.json"), "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
    if not success:
        print(f"❌ 房间 {os.path.basename(args.room_json_path)}跳过。")
main_batch()