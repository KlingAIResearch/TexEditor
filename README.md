
<h2 align="center"> 
  <a href="https://arxiv.org/abs/2603.18488">
TexEditor: Structure-Preserving Text-Driven Texture Editing
  </a>
</h2>


## 📣 News

**[2026/05/01]**: **TexEditor** has been accepted in ICML 2026 [https://icml.cc/virtual/2026/poster/61988].

**[2026/03/21]**: We release **TexEditor**, a dedicated texture editing model based on Qwen-Image-Edit-2509.


## Data and model

You can find RL training data and TexBench here:
[https://www.modelscope.cn/datasets/zhaohhh2000/TexBench]


You can find SFT merged Dit, RL lora and Struture server model (SAUGE) here:
[https://www.modelscope.cn/models/zhaohhh2000/TexEditor]

More detail about SAUGE:
[https://github.com/Star-xing1/SAUGE]


## Env
We recommend to consturct envionment for model training and blender rendering respectively.

### Model training

For SFT, we use DiffSynth-Studio for quick lora learing.[https://github.com/modelscope/DiffSynth-Studio]

For RL learning env:

```
conda env create -f environment_rl.yml
```

Download the SAUGE (wireframe detector) model weights from our released model directory in advance and place them in a directory of your choice.

### Blender

```
conda create -n yh_tex python=3.11
pip install blenderproc
git clone https://github.com/DLR-RM/BlenderProc
cd BlenderProc
pip install -e .
pip install transformers
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install accelerate
pip install flash-attn --no-build-isolation
pip install opencv-python
git clone https://github.com/facebookresearch/sam3
cd sam3
pip install -e .
pip install pycocotools
```



## 🗝️ Train
For the SFT stage of model training, we do not provide detailed descriptions; please refer to the LoRA training section of Qwen-Edit in DiffSynth-Studio. We now focus on the training pipeline of the RL stage.


### Deploy Reward Server

For Instruction score：
```
You need to modify the Gemini API configuration in the gemini_score_api function of UniWorld-V2/flow_grpo/reward_edge_loss.py and enter your own account credentials.
```

For Structure score：
```
First， check below model path：
ckpt_p = '/mmu-vcg/zb08/yihang/2026_works/SAUGE/model/sam_vit_b_01ec64.pth'
sam = sam_model_registry['vit_b'](checkpoint=ckpt_p)

model = Model.SAUGE(None, sam_generator=mask_generator, mode='eval').cuda()
checkpoint = torch.load('UniWorld-V2/ckpt/bsds/sauge_vitb.pth')['state_dict']

Then launch UniWorld-V2/zb_scripts/edge_server_2.py on different ports to distribute requests for computing structural scores across different GPUs.
```



### RL Data 

Directory structure:

```
- dataset-dir
  - images/
     - new_data_more
     - new_data_texture
  - train.jsonl
  - val.jsonl
```

Data format
```
{"prompt": "Make the person's white shirt slightly translucent, like frosted glass, preserving its original color and shape.", "edit_image": "xxx/new_data_more/000000103504.jpg", "kind": "more"}
...
```

### Configure Training

See `config/texture_rl.py' for available configurations.

### Run Training


Modify the server address settings for the structural loss in flow_grpo/reward_edge_loss.py, making sure the IP addresses and ports are matched correctly.
```
URL0 = "http://10.82.121.94:9004/reward" # aiplatform 
URL1 = "http://10.82.121.94:9002/reward" # aiplatform 
URL2 = "http://10.82.121.94:9003/reward" # aiplatform 
URL3 = "http://10.82.121.94:9005/reward" # aiplatform 
ULR_list = [URL0, URL1, URL2, URL3]
```

Training script：

```shell

torchrun --nproc_per_node=8 \
    scripts/train_main.py --config config/texture_rl.py:qwen_mllm_reward
```

## Inference

Before Inference，you need to set input data path and model path

Check model path
```
    ### load sft lora base !!
    print("Loading LoRA SFT model ")
    dit_state = torch.load("SFT_merged_ckpt_path")
    pipeline.transformer.load_state_dict(dit_state)

    # from peft import PeftModel
    print("Loading LoRA RL model " + "**"*20)
    lora_path = 'RL_lora_path'
```

Infer script
```
python zb_scripts/single_infer_rl.py  --config config/infer_nft.py:qwen_mllm_reward
```

## Evaluation

Before using the scoring script, ensure that the input and output paths are properly configured.
```
dst_foldr = 'score_save_path'

os.makedirs(dst_foldr, exist_ok=True)

# 输入数据json
in_p = '/mmu-vcg/zb08/outputs/texture_edit/train_data/rl_data/maybe_v2_final_more/test_clean_v2.jsonl'

# 输入的编辑结果图像folder path
dst_img_p = 'xxx'
```

Scripts paths
```
zb_scripts/open_scores/make_more.py # attribute task / instruction

zb_scripts/open_scores/make_texture.py # texture task / instruction

zb_scripts/open_scores/make_score_edge.py # structure score

zb_scripts/open_scores/make_score_gemini.sh # multiprocess bash for instruction score

zb_scripts/open_scores/make_score_edge.sh # multiprocess bash for instruction score

```


## Make Blender Data


### Data Pipeline Overview

```
Blender Rendering  →  Instruction Generation  →  Data Cleaning
```

---

### Stage 1 — Blender Pipeline

**Goal**: Render 3D-FRONT scenes with Blender to produce image pairs (original texture / replaced texture).

#### Required Datasets

- [3D-FRONT](https://tianchi.aliyun.com/dataset/65347) — 3D furnished indoor scene dataset. Access requires an application.
- [MatSynth](https://huggingface.co/datasets/gvecchio/MatSynth) — Material texture dataset.

#### Usage

```bash
python blender_pipeline_code/run_blender.py \
  --front_path /path/to/3D-FRONT \
  --output_path ./output \
  --texture_path /path/to/MatSynth \
  --blenderproc_path /path/to/BlenderProc \
  --blender_path /path/to/blender-4.2.1-linux-x64 \
  --gpus 2 3 4
```

#### Arguments

| Argument | Description |
|---|---|
| `--front_path` | Root directory of the 3D-FRONT dataset |
| `--output_path` | Output directory for rendered image pairs |
| `--texture_path` | Root directory of the MatSynth dataset |
| `--blenderproc_path` | Path to the BlenderProc package |
| `--blender_path` | Path to the Blender executable directory |
| `--gpus` | GPU IDs to use. One task is launched per GPU in parallel. |

---

### Stage 2 — Instruction Pipeline

**Goal**: Automatically generate texture-editing instructions for each image pair produced in Stage 1. Output is saved as `all.json`.

#### Usage

```bash
python blender_pipeline_code/instruction.py \
  --gpus 2 3 4 \
  --base_dir ./output \
  --dest_dir ./results \
  --log_dir ./logs
```

#### Arguments

| Argument | Description |
|---|---|
| `--gpus` | GPU IDs to use |
| `--base_dir` | Image pair directory (output from Stage 1) |
| `--dest_dir` | Output directory for `all.json` |
| `--log_dir` | Log output directory (optional) |

---

### Stage 3 — Wash Pipeline

**Goal**: Filter and clean the generated instructions using SAM3 and Qwen-VL. Output is saved as `result.json`.

#### Usage

```bash
python blender_pipeline_code/pipeline.py \
  --gpu 2 3 \
  --json_file ./results/all.json \
  --dest_dir ./results \
  --qwen_model /path/to/Qwen3-VL-32B-Instruct \
  --checkpoint_path /path/to/sam3.pt
```

#### Arguments

| Argument | Description |
|---|---|
| `--gpu` | GPU IDs to use |
| `--json_file` | Path to the JSON file from Stage 2 |
| `--dest_dir` | Output directory for `result.json` |
| `--qwen_model` | Path to the Qwen-VL model weights |
| `--checkpoint_path` | Path to the SAM3 checkpoint. If omitted, downloads automatically from HuggingFace. |


<!-- 

## 👍 Acknowledgement

- [DiffusionNFT](https://github.com/NVlabs/DiffusionNFT)
- [UniWorld-V1](https://github.com/PKU-YuanGroup/UniWorld-V1) -->



<!-- ## ✏️ Citation

```
@article{li2025uniworldv2,
    title={Uniworld-V2: Reinforce Image Editing with Diffusion Negative-aware Finetuning and MLLM Implicit Feedback},
    author={Li, Zongjian and Liu, Zheyuan and Zhang, Qihui and Lin, Bin and Yuan, Shenghai and Yan, Zhiyuan and Ye, Yang and Yu, Wangbo and Niu, Yuwei and Yuan, Li},
    journal={arXiv preprint arXiv:2506.03147},
    year={2025}
}

``` -->
