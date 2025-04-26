import os
import torch
# from diffusers import MochiPipeline
from pipeline_mochi_rgba import MochiPipeline
from pipeline_mochi_rgba import prepare_attention_mask, linear_quadratic_schedule, retrieve_timesteps
from diffusers.utils import export_to_video
import argparse
from rgba_utils import *
import numpy as np
import copy

def main(args):
    # 1. load pipeline  
    pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview", torch_dtype=torch.bfloat16).to("cuda")
    pipe.enable_vae_tiling()

    # 2. define prompt and arguments
    pipeline_args = {
        "prompt": args.prompt,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "max_sequence_length": 256,
        "output_type": "latent",
    }

    num_inference_steps = args.num_inference_steps
    # 3. prepare rgbx utils    
    prepare_for_rgba_inference(
        pipe.transformer,
        device="cuda",
        dtype=torch.bfloat16,
        lora_rank=32,
    )
    

    if args.lora_path is not None:
        checkpoint = torch.load(args.lora_path, map_location="cpu", weights_only=False)
        processor_state_dict = checkpoint["state_dict"]
        load_processor_state_dict(pipe.transformer, processor_state_dict)


    (
        prompt_embeds,
        prompt_attention_mask,
        negative_prompt_embeds,
        negative_prompt_attention_mask,
    ) = pipe.encode_prompt(
        prompt=args.prompt,
        num_videos_per_prompt=1,
        max_sequence_length=256,
        device="cuda",
    )

    num_channels_latents = pipe.transformer.config.in_channels
    latents = pipe.prepare_latents(
        1,
        num_channels_latents,
        args.height,
        args.width,
        args.num_frames,
        prompt_embeds.dtype,
        "cuda",
        generator=None,
    ).repeat(1,1,2,1,1)
    all_attention_mask = prepare_attention_mask(prompt_attention_mask, latents)

    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
        
    
    threshold_noise = 0.025
    sigmas = linear_quadratic_schedule(num_inference_steps, threshold_noise)
    sigmas = np.array(sigmas)

    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps,
        "cuda",
        sigmas,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * pipe.scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)
    
    
 
    with pipe.progress_bar(total=num_inference_steps//4) as progress_bar:
        for i, t in enumerate(timesteps[:len(timesteps) // 4]):
            latent_model_input = torch.cat([latents] * 2)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)

            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                encoder_attention_mask=all_attention_mask,
                return_dict=False,
            )[0]
            # Mochi CFG + Sampling runs in FP32
            noise_pred = noise_pred.to(torch.float32)
            
            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = pipe.scheduler.step(noise_pred, t, latents.to(torch.float32), return_dict=False)[0]
            latents = latents.to(latents_dtype)

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                progress_bar.update()

    latents1 = latents.clone()
    latents2 = latents.clone()
    
    for n, latents in enumerate([latents1, latents2]):
        timesteps_ = timesteps[:]
        pipe_ = copy.copy(pipe) 
        pipe_.scheduler = copy.copy(pipe.scheduler)
        with pipe_.progress_bar(total=num_inference_steps//4 * 3) as progress_bar:
            for i, t in enumerate(timesteps_[len(timesteps_) // 4:]):
                i = i + len(timesteps_) // 4
                latent_model_input = latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)

                noise_pred = pipe_.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    encoder_attention_mask=all_attention_mask,
                    return_dict=False,
                )[0]
                # Mochi CFG + Sampling runs in FP32
                noise_pred = noise_pred.to(torch.float32)
                
                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = pipe_.scheduler.step(noise_pred, t, latents.to(torch.float32), return_dict=False)[0]
                latents = latents.to(latents_dtype)

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                # call the callback, if provided
                if i == len(timesteps_) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe_.scheduler.order == 0):
                    progress_bar.update()    

        has_latents_mean = hasattr(pipe_.vae.config, "latents_mean") and pipe_.vae.config.latents_mean is not None
        has_latents_std = hasattr(pipe_.vae.config, "latents_std") and pipe_.vae.config.latents_std is not None
        if has_latents_mean and has_latents_std:
            latents_mean = (
                torch.tensor(pipe_.vae.config.latents_mean).view(1, 12, 1, 1, 1).to(latents.device, latents.dtype)
            )
            latents_std = (
                torch.tensor(pipe_.vae.config.latents_std).view(1, 12, 1, 1, 1).to(latents.device, latents.dtype)
            )
            latents = latents * latents_std / pipe_.vae.config.scaling_factor + latents_mean
        else:
            latents = latents / pipe_.vae.config.scaling_factor
        print(latents.shape)
        video = pipe_.vae.decode(latents, return_dict=False)[0]
        video = pipe_.video_processor.postprocess_video(video)
        print(video.shape) 
        export_to_video(video[0], os.path.join(args.output_path, f"video_{n}.mp4"), fps=args.fps)
        print(f"Exported to {os.path.join(args.output_path, f'video_{n}.mp4')}")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt")
    parser.add_argument("--prompt", type=str, required=True, help="The description of the video to be generated")
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    
    parser.add_argument(
        "--model_path", type=str, default="genmo/mochi-1-preview", help="Path of the pre-trained model use"
    )
    parser.add_argument("--output_path", type=str, default="./output", help="The path save generated video")
    parser.add_argument("--guidance_scale", type=float, default=6, help="The scale for classifier-free guidance")
    parser.add_argument("--num_inference_steps", type=int, default=64, help="Inference steps")
    parser.add_argument("--num_frames", type=int, default=79, help="Number of steps for the inference process")
    parser.add_argument("--width", type=int, default=576, help="Number of steps for the inference process")
    parser.add_argument("--height", type=int, default=320, help="Number of steps for the inference process")
    parser.add_argument("--fps", type=int, default=30, help="Number of steps for the inference process")
    parser.add_argument("--seed", type=int, default=None, help="The seed for reproducibility")
    args = parser.parse_args()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.inference_mode():
            with torch.no_grad():
                main(args)
