import argparse
import os
import subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from fliter_scene import clean_scene_random_remove

def parse_args():
    parser = argparse.ArgumentParser(description="Texture editing with BlenderProc.")
    parser.add_argument('--front_path', type=str, default='', help='room json file path.')
    parser.add_argument('--max_attempts_per_room', type=int, default=10, help='max attempts to load a furniture group per room.')
    parser.add_argument('--camera_sampling_attempts', type=int, default=1000, help='max attempts to sample a valid camera pose per photo.')
    parser.add_argument('--output_path', type=str, default='./output', help='base output path.')
    parser.add_argument('--texture_path', type=str, default="", help='base path for new textures.')
    parser.add_argument('--samples', type=int, default=1024, help='number of samples per pixel.')
    parser.add_argument('--light_bounces', type=int, default=100, help='number of light bounces.')
    parser.add_argument('--blenderproc_path', type=str, default="./Blenderproc", help='path to BlenderProc executable.')
    parser.add_argument('--blender_path', type=str, default="", help='path to BlenderProc executable.')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='GPU IDs, e.g. --gpus 0 1 2 3')
    return parser.parse_args()
def main():
    args = parse_args()
    front_room_path = os.path.join(args.front_path, "3D-FRONT")
    front_furniture_path = os.path.join(args.front_path, "3D-FUTURE-model")
    front_texture_path = os.path.join(args.front_path, "3D-FRONT-texture")
    front_mapping_path = os.path.join(args.blenderproc_path, "blenderproc/resources/front_3D/3D_front_mapping.csv")
    room_lis = sorted(os.listdir(front_room_path))
    os.makedirs(os.path.join(args.front_path, "flitered_3dfront"), exist_ok=True)

    splits = np.array_split(room_lis, len(args.gpus))

    def run_rooms(rooms, gpu_id):
        for room_name in rooms:
            room_json_path = os.path.join(front_room_path, room_name)
            filtered_room_json_path = os.path.join(args.front_path, "flitered_3dfront", room_name)
            clean_scene_random_remove(room_json_path, filtered_room_json_path)
            for group_strategy in ["obj", "obj+texture"]:
                cmd = (
                    f"CUDA_VISIBLE_DEVICES={gpu_id} blenderproc run "
                    f"blender_pipeline_code/each_blender.py "
                    f"--custom-blender-path {args.blender_path} "
                    f"--room_json_path {filtered_room_json_path} "
                    f"--output_base_path {args.output_path} "
                    f"--new_texture_base_path {args.texture_path} "
                    f"--group_strategy {group_strategy} "
                    f"--mapping_file {front_mapping_path} "
                    f"--texture_base_path {front_texture_path} "
                    f"--furniture_base_path {front_furniture_path} "
                    f"--max_attempts_per_room {args.max_attempts_per_room} "
                    f"--camera_sampling_attempts {args.camera_sampling_attempts} "
                    f"--samples {args.samples} "
                    f"--light_bounces {args.light_bounces}"
                )
                subprocess.run(cmd, shell=True)

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        for i, gpu_id in enumerate(args.gpus):
            executor.submit(run_rooms, splits[i].tolist(), gpu_id)


if __name__ == "__main__":
    main()