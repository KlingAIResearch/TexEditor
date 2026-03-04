from typing import Any, Dict, List, Optional, Union
import torch
import numpy as np
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import (
    retrieve_timesteps,
)

from diffusers.utils.torch_utils import randn_tensor

from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    calculate_shift,
    calculate_dimensions,
)

from diffusers.image_processor import PipelineImageInput
from flow_grpo.diffusers_patch.solver import run_sampling


CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024

def _get_qwen_prompt_embeds(
    self,
    prompt: Union[str, List[str]] = None,
    image: Optional[List[torch.Tensor]] = None, # 预期为与 prompt 长度一致的 list
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    max_seq_len: int = 1024,
):
    device = device or self._execution_device
    dtype = dtype or self.text_encoder.dtype
    
    if isinstance(prompt, str):
        prompt = [prompt]
    
    # 修改：不再使用 Picture 1, Picture 2 这种累加模式
    # 每个样本都是独立的 "图像 + Prompt" 结构
    img_prompt_template = "<|vision_start|><|image_pad|><|vision_end|>"
    template = self.prompt_template_encode
    drop_idx = self.prompt_template_encode_start_idx
    
    # 构造 batch 化的 prompt 文本
    # 确保每个 prompt 只关联自己索引下的那张图
    txt = [template.format(img_prompt_template + p) for p in prompt]
    
    # processor 传入 images 列表时，会自动按 batch 处理
    model_inputs = self.processor(
        text=txt,
        images=image,
        padding=True,
        return_tensors="pt",
    ).to(device)
    
    outputs = self.text_encoder(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        pixel_values=model_inputs.pixel_values,
        image_grid_thw=model_inputs.image_grid_thw,
        output_hidden_states=True,
    )
    
    hidden_states = outputs.hidden_states[-1]
    split_hidden_states = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
    split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
    
    attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
    
    prompt_embeds = torch.stack([
        torch.cat([
            u[:max_seq_len] if u.size(0) > max_seq_len else u,
            u.new_zeros(max(0, max_seq_len - u.size(0)), u.size(1))
        ])
        for u in split_hidden_states
    ])
    
    encoder_attention_mask = torch.stack([
        torch.cat([
            u[:max_seq_len] if u.size(0) > max_seq_len else u,
            u.new_zeros(max(0, max_seq_len - u.size(0)))
        ])
        for u in attn_mask_list
    ])
    
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    return prompt_embeds, encoder_attention_mask

def encode_prompt(
    self,
    prompt: Union[str, List[str]],
    image: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    num_images_per_prompt: int = 1,
    prompt_embeds: Optional[torch.Tensor] = None,
    prompt_embeds_mask: Optional[torch.Tensor] = None,
    max_sequence_length: int = 1024,
):
    device = device or self._execution_device
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]
    if prompt_embeds is None:
        prompt_embeds, prompt_embeds_mask = _get_qwen_prompt_embeds(self, prompt, image, device, max_seq_len=max_sequence_length)
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
    prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)
    return prompt_embeds, prompt_embeds_mask

def zb_prepare_latents(
    self,
    images,
    batch_size,
    num_images_per_prompt,
    num_channels_latents,
    height,
    width,
    dtype,
    device,
    generator,
    latents=None,
):
    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (self.vae_scale_factor * 2))
    width = 2 * (int(width) // (self.vae_scale_factor * 2))

    shape = (batch_size, 1, num_channels_latents, height, width)

    image_latents = None
    if images is not None:
        if not isinstance(images, list):
            images = [images]
        all_image_latents = []
        for image in images:
            image = image.to(device=device, dtype=dtype)
            
            if image.shape[1] != self.latent_channels:
                image_latents = self._encode_vae_image(image=image, generator=generator)
            else:
                image_latents = image

            # if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            #     # expand init_latents for batch_size
            #     additional_image_per_prompt = batch_size // image_latents.shape[0]
            #     image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
            # elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            #     raise ValueError(
            #         f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            #     )
            # else:
            #     image_latents = torch.cat([image_latents], dim=0)

            if num_images_per_prompt:
                image_latents = torch.cat([image_latents] * num_images_per_prompt, dim=0)

            
            image_latent_height, image_latent_width = image_latents.shape[3:]
            image_latents = self._pack_latents(
                image_latents, 1, num_channels_latents, image_latent_height, image_latent_width
            )
            all_image_latents.append(image_latents)

        # image_latents = all_image_latents
        # image_latents = torch.cat(all_image_latents, dim=1)
        image_latents = torch.cat(all_image_latents, dim=0)


    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )
    if latents is None:
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
    else:
        latents = latents.to(device=device, dtype=dtype)

    return latents, image_latents

@torch.no_grad()
def pipeline_with_logprob(
    self,
    image: Optional[List[PipelineImageInput]] = None, # 明确输入为 List
    prompt: Union[str, List[str]] = None,
    negative_prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    true_cfg_scale: float = 4.0,
    guidance_scale: Optional[float] = None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    prompt_embeds_mask: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "pil",
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 256,
    noise_level: float = 0.7,
    deterministic: bool = False,
    max_area: Optional[int] = None,
    solver: str = "flow",
):
    max_area = VAE_IMAGE_SIZE if max_area is None else max_area
    device = self._execution_device
    
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
        prompt = [prompt]
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # 修改：基于 batch 中的第一张图计算输出尺寸，或根据需要统一 resize
    reference_image = image[0] if isinstance(image, list) else image
    calculated_width, calculated_height = calculate_dimensions(VAE_IMAGE_SIZE, reference_image.size[0] / reference_image.size[1])
    height = height or calculated_height
    width = width or calculated_width
    
    multiple_of = self.vae_scale_factor * 2
    width = width // multiple_of * multiple_of
    height = height // multiple_of * multiple_of

    self.check_inputs(prompt, height, width, negative_prompt=negative_prompt, 
                      prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                      prompt_embeds_mask=prompt_embeds_mask, negative_prompt_embeds_mask=negative_prompt_embeds_mask,
                      max_sequence_length=max_sequence_length)

    # 3. 图像预处理 (1:1 对应)
    condition_images = []
    vae_images = []
    vae_image_sizes = []
    
    # 确保 image 是列表且长度与 batch_size 一致
    input_images = image if isinstance(image, list) else [image]
    
    for img in input_images:
        img_w, img_h = img.size
        # 计算该样本对应的条件图尺寸
        c_w, c_h = calculate_dimensions(CONDITION_IMAGE_SIZE, img_w / img_h)
        # 计算该样本对应的 VAE 隐空间参考图尺寸
        v_w, v_h = calculate_dimensions(max_area, img_w / img_h)
        
        condition_images.append(self.image_processor.resize(img, c_h, c_w))
        vae_images.append(self.image_processor.preprocess(img, v_h, v_w).unsqueeze(2))
        vae_image_sizes.append((v_w, v_h))

    has_neg_prompt = negative_prompt is not None or (negative_prompt_embeds is not None)
    do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

    # 4. 编码 Prompt (调用修改后的 _get_qwen_prompt_embeds)
    prompt_embeds, prompt_embeds_mask = encode_prompt(
        self,
        image=condition_images, 
        prompt=prompt,
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )
    
    if do_true_cfg:
        negative_prompt_embeds, negative_prompt_embeds_mask = encode_prompt(
            self,
            image=condition_images,
            prompt=negative_prompt,
            prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=negative_prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

    # 5. 准备 Latents
    num_channels_latents = self.transformer.config.in_channels // 4
    # 破案了 我应该按照batchsize 排列 image latents
    latents, image_latents = zb_prepare_latents(
        self,
        vae_images,
        batch_size ,
        num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # breakpoint()
    if image_latents is not None:
        channel_dim = image_latents.shape[-1]
        image_latents = t2 = image_latents.view(batch_size * num_images_per_prompt, -1, channel_dim)

    # 4. 构造 img_shapes (每样本只包含 1个目标 + 1个参考图)
    s = self.vae_scale_factor * 2
    img_shapes = []
    for i in range(batch_size):
        v_w, v_h = vae_image_sizes[i]
        img_shapes.append([
            (1, height // s, width // s), 
            (1, v_h // s, v_w // s)
        ])

    # 准备 sigmas 和 timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    if getattr(self.scheduler.config, "use_flow_sigmas", False):
        sigmas = None
        
    mu = calculate_shift(
        latents.shape[1],
        self.scheduler.config.get("base_image_seq_len", 256),
        self.scheduler.config.get("max_image_seq_len", 4096),
        self.scheduler.config.get("base_shift", 0.5),
        self.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)
    
    # Guidance 处理
    guidance = None
    if self.transformer.config.guidance_embeds and guidance_scale is not None:
        guidance = torch.full([latents.shape[0]], guidance_scale, device=device, dtype=torch.float32)

    txt_seq_lens = [max_sequence_length] * (batch_size * num_images_per_prompt)

    
    # 7. 采样函数
    # def v_pred_fn(z, sigma):
    #     # 这里的 image_latents 现在是 [Batch, Seq, Channel]
    #     # z 也是 [Batch, Seq, Channel]，两者在 dim=1 拼接
    #     latent_model_input = torch.cat([z, image_latents], dim=1) if image_latents is not None else z

    #     t_steps = torch.full([latent_model_input.shape[0]], sigma, device=z.device, dtype=torch.float32)
        
    #     noise_pred = self.transformer(
    #         hidden_states=latent_model_input,
    #         timestep=t_steps,
    #         guidance=guidance,
    #         encoder_hidden_states_mask=prompt_embeds_mask,
    #         encoder_hidden_states=prompt_embeds,
    #         img_shapes=img_shapes, # 每个 batch 独立的形状信息
    #         txt_seq_lens=txt_seq_lens,
    #         return_dict=False,
    #     )[0]
    #     noise_pred = noise_pred[:, : z.size(1)]
        
    #     if do_true_cfg:
    #         neg_pred = self.transformer(
    #             hidden_states=latent_model_input,
    #             timestep=t_steps,
    #             guidance=guidance,
    #             encoder_hidden_states_mask=negative_prompt_embeds_mask,
    #             encoder_hidden_states=negative_prompt_embeds,
    #             img_shapes=img_shapes,
    #             txt_seq_lens=txt_seq_lens,
    #             return_dict=False,
    #         )[0][:, : z.size(1)]

    #         comb_pred = neg_pred + true_cfg_scale * (noise_pred - neg_pred)
    #         # 这里的 Norm 修正逻辑保持不变
    #         noise_pred = comb_pred * (torch.norm(noise_pred, dim=-1, keepdim=True) / torch.norm(comb_pred, dim=-1, keepdim=True))

    #     return noise_pred

    def v_pred_fn(z, sigma):
        # 此时 z 和 image_latents 的 dim=0 都是 batch_size，dim=1 拼接后长度为 8192
        latent_model_input = torch.cat([z, image_latents], dim=1) if image_latents is not None else z
        t_steps = torch.full([latent_model_input.shape[0]], sigma, device=z.device, dtype=torch.float32)
        
        noise_pred = self.transformer(
            hidden_states=latent_model_input,
            timestep=t_steps,
            guidance=guidance,
            encoder_hidden_states_mask=prompt_embeds_mask,
            encoder_hidden_states=prompt_embeds,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
        )[0][:, : z.size(1)]
        
        if true_cfg_scale > 1 and negative_prompt:
            neg_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=t_steps,
                guidance=guidance,
                encoder_hidden_states_mask=negative_prompt_embeds_mask,
                encoder_hidden_states=negative_prompt_embeds,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                return_dict=False,
            )[0][:, : z.size(1)]
            noise_pred = neg_pred + true_cfg_scale * (noise_pred - neg_pred)

        return noise_pred

    # 运行采样
    sigmas_tensor = self.scheduler.sigmas.float()
    latents, all_latents, all_log_probs = run_sampling(
        v_pred_fn, latents, sigmas_tensor, solver, deterministic, noise_level
    )

    # 8. 后处理
    latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
    latents = latents.to(self.vae.dtype)
    
    # 这里的 VAE Decode 逻辑保持原样
    l_mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    l_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents = latents / l_std + l_mean
    
    image_out = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
    image_out = self.image_processor.postprocess(image_out, output_type=output_type)

    self.maybe_free_model_hooks()

    return (image_out, all_latents, image_latents, img_shapes, txt_seq_lens, prompt_embeds, prompt_embeds_mask, all_log_probs)
