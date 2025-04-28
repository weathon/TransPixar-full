# Copyright 2024 The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import random
from glob import glob
import math
import os
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Any, Dict, Tuple, List

import torch
import wandb
from pipeline_mochi_rgba import *
from diffusers import FlowMatchEulerDiscreteScheduler, MochiTransformer3DModel
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.training_utils import cast_training_params
from diffusers.utils import export_to_video
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from huggingface_hub import create_repo, upload_folder
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


from args import get_args  # isort:skip
from dataset_simple import LatentEmbedDataset

from utils import print_memory, reset_memory  # isort:skip
from rgba_utils import *


# Taken from
# https://github.com/genmoai/mochi/blob/aba74c1b5e0755b1fa3343d9e4bd22e89de77ab1/demos/fine_tuner/train.py#L139
def get_cosine_annealing_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_model_card(
    repo_id: str,
    videos=None,
    base_model: str = None,
    validation_prompt=None,
    repo_folder=None,
    fps=30,
):
    widget_dict = []
    if videos is not None and len(videos) > 0:
        for i, video in enumerate(videos):
            export_to_video(video, os.path.join(repo_folder, f"final_video_{i}.mp4"), fps=fps)
            widget_dict.append(
                {
                    "text": validation_prompt if validation_prompt else " ",
                    "output": {"url": f"final_video_{i}.mp4"},
                }
            )

    model_description = f"""
# Mochi-1 Preview LoRA Finetune

<Gallery />

## Model description

This is a lora finetune of the Mochi-1 preview model `{base_model}`.

The model was trained using [CogVideoX Factory](https://github.com/a-r-r-o-w/cogvideox-factory) - a repository containing memory-optimized training scripts for the CogVideoX and Mochi family of models using [TorchAO](https://github.com/pytorch/ao) and [DeepSpeed](https://github.com/microsoft/DeepSpeed). The scripts were adopted from [CogVideoX Diffusers trainer](https://github.com/huggingface/diffusers/blob/main/examples/cogvideo/train_cogvideox_lora.py).

## Download model

[Download LoRA]({repo_id}/tree/main) in the Files & Versions tab.

## Usage

Requires the [🧨 Diffusers library](https://github.com/huggingface/diffusers) installed.

```py
from diffusers import MochiPipeline
from diffusers.utils import export_to_video
import torch 

pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview")
pipe.load_lora_weights("CHANGE_ME")
pipe.enable_model_cpu_offload()

with torch.autocast("cuda", torch.bfloat16):
    video = pipe(
        prompt="CHANGE_ME",
        guidance_scale=6.0,
        num_inference_steps=64,
        height=480,
        width=848,
        max_sequence_length=256,
        output_type="np"
    ).frames[0]
export_to_video(video)
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters) on loading LoRAs in diffusers.

"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="apache-2.0",
        base_model=base_model,
        prompt=validation_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-video",
        "diffusers-training",
        "diffusers",
        "lora",
        "mochi-1-preview",
        "mochi-1-preview-diffusers",
        "template:sd-lora",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(
    pipe: MochiPipeline,
    args: Dict[str, Any],
    pipeline_args: Dict[str, Any],
    step: int,
    wandb_run: str = None,
    is_final_validation: bool = False,
):
    print(
        f"Running validation... \n Generating {args.num_validation_videos} videos with prompt: {pipeline_args['prompt']}."
    )
    phase_name = "test" if is_final_validation else "validation"

    if not args.enable_model_cpu_offload:
        pipe = pipe.to("cuda")

    # run inference
    generator = torch.manual_seed(args.seed) if args.seed else None

    videos = []
    with torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
        for _ in range(args.num_validation_videos):
            video = pipe(**pipeline_args, generator=generator, output_type="np").frames[0]
            print(video.shape)
            videos.append(video)

    video_filenames = []
    for i, video in enumerate(videos):
        prompt = (
            pipeline_args["prompt"][:25]
            .replace(" ", "_")
            .replace(" ", "_")
            .replace("'", "_")
            .replace('"', "_")
            .replace("/", "_")
        )
        filename = os.path.join(args.output_dir, f"{phase_name}_{str(step)}_video_{i}_{prompt}.mp4")
        export_to_video(video, filename, fps=30)
        video_filenames.append(filename)

    if wandb_run:
        wandb.log(
            {
                phase_name: [
                    wandb.Video(filename, caption=f"{i}: {pipeline_args['prompt']}", fps=30)
                    for i, filename in enumerate(video_filenames)
                ]
            }
        )

    return videos


# Adapted from the original code:
# https://github.com/genmoai/mochi/blob/aba74c1b5e0755b1fa3343d9e4bd22e89de77ab1/src/genmo/mochi_preview/pipelines.py#L578
def cast_dit(model, dtype):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            assert any(
                n in name for n in ["time_embed", "proj_out", "blocks", "norm_out"]
            ), f"Unexpected linear layer: {name}"
            module.to(dtype=dtype)
        elif isinstance(module, torch.nn.Conv2d):
            module.to(dtype=dtype)
    return model


def save_checkpoint(model, optimizer, lr_scheduler, global_step, checkpoint_path):
    # lora_state_dict = get_peft_model_state_dict(model)
    processor_state_dict = get_processor_state_dict(model)
    torch.save(
        {
            "state_dict": processor_state_dict,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "global_step": global_step,
        },
        checkpoint_path,
    )

import ignite
def latent_mask_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
):
    pred = pred.float().abs().mean(1)
    target = target.float().abs().mean(1)
    pred = (pred - pred.min())/(pred.max() - pred.min())
    target = (target - target.min())/(target.max() - target.min())

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    dice_loss = 1 - (2 * intersection + eps) / (union + eps)
    return dice_loss, pred, target

def generate_custom_dist_tensor(size):
  """
  Generates a tensor from a custom distribution on [0, 1]
  where the density at x=1 is twice the density at x=0.
  Uses inverse transform sampling based on CDF F(x) = 2^x - 1.
  """
  # Uniform samples
  u = torch.rand(size)

  # Inverse CDF: x = log2(u + 1) = ln(u + 1) / ln(2)
  ln2 = torch.log(torch.tensor(2.0))
  x = torch.log(u + 1) / ln2

  return x

class CollateFunction:
    def __init__(self, caption_dropout: float = None) -> None:
        self.caption_dropout = caption_dropout

    def __call__(self, samples: List[Tuple[dict, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        ldists = torch.cat([data[0]["ldist"] for data in samples], dim=0)
        z = DiagonalGaussianDistribution(ldists).sample()
        assert torch.isfinite(z).all()

        # Sample noise which we will add to the samples.
        eps = torch.randn_like(z)
        distribution = Beta(concentration1=5.0, concentration0=1.0)
        # samples = torch.distributions.Exponential(0.3).sample(z.shape[:1]).to(torch.float32)
        # samples = samples / samples.max()
        # sigma = distribution.sample().to(torch.float32)
        sigma = torch.rand(z.shape[:1], device="cpu", dtype=torch.float32)
        # sigma = generate_custom_dist_tensor(z.shape[:1]).to(torch.float32)

        prompt_embeds = torch.cat([data[1]["prompt_embeds"] for data in samples], dim=0)
        prompt_attention_mask = torch.cat([data[1]["prompt_attention_mask"] for data in samples], dim=0)
        negative_prompt_embeds = torch.cat(
            [data[1]["negative_prompt_embeds"] for data in samples], dim=0
        ) if "negative_prompt_embeds" in samples[0][1] else None
        negative_prompt_attention_mask = torch.cat(
            [data[1]["negative_prompt_attention_mask"] for data in samples], dim=0
        ) if "negative_prompt_attention_mask" in samples[0][1] else None
        
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=1)
        prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=1)
        
        if self.caption_dropout and random.random() < self.caption_dropout:
            prompt_embeds.zero_()
            prompt_attention_mask = prompt_attention_mask.long()
            prompt_attention_mask.zero_()
            prompt_attention_mask = prompt_attention_mask.bool()

        return dict(
            z=z, eps=eps, sigma=sigma, prompt_embeds=prompt_embeds, prompt_attention_mask=prompt_attention_mask
        )

from torch.distributions.beta import Beta
def main(args):
    if not torch.cuda.is_available():
        raise ValueError("Not supported without CUDA.")

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    # Handle the repository creation
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Prepare models and scheduler
    transformer = MochiTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    # scheduler.set_timesteps(num_inference_steps=40)
    
    transformer.requires_grad_(False)
    transformer.to("cuda")
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    if args.cast_dit:
        transformer = cast_dit(transformer, torch.bfloat16)
    if args.compile_dit:
        transformer.compile()
 
    prepare_for_rgba_inference(
        model=transformer,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        # seq_length=seq_length,
        lora_rank=args.rank,
        lora_alpha=args.lora_alpha,
    )    
    processor_params = get_all_processor_params(transformer)
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = args.learning_rate * args.train_batch_size
    # only upcast trainable parameters (LoRA) into fp32

    if not isinstance(processor_params, list):
        processor_params = [processor_params]
    for m in processor_params:
        for param in m:
            # only upcast trainable parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)

    # Prepare optimizer
    transformer_lora_parameters = processor_params # list(filter(lambda p: p.requires_grad, transformer.parameters()))
    num_trainable_parameters = sum(param.numel() for param in transformer_lora_parameters)
    optimizer = torch.optim.AdamW(transformer_lora_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

    # Dataset and DataLoader
    train_vids = list(sorted(glob(f"{args.data_root}/*.mp4")))
    train_vids = [v for v in train_vids if not v.endswith(".recon.mp4")]
    print(f"Found {len(train_vids)} training videos in {args.data_root}")
    assert len(train_vids) > 0, f"No training data found in {args.data_root}"

    collate_fn = CollateFunction(caption_dropout=args.caption_dropout)
    train_dataset = LatentEmbedDataset(train_vids, repeat=1)
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.pin_memory,
        shuffle=True
    )

    # LR scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = len(train_dataloader)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_cosine_annealing_lr_scheduler(
        optimizer, warmup_steps=args.lr_warmup_steps, total_steps=args.max_train_steps
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = len(train_dataloader)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    wandb_run = None
    if args.report_to == "wandb":
        tracker_name = args.tracker_name or "mochi-1-rgba-lora"
        wandb_run = wandb.init(project=tracker_name, config=vars(args))

    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu", weights_only=False)
        if "global_step" in checkpoint:
            global_step = checkpoint["global_step"]
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "lr_scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        # set_peft_model_state_dict(transformer, checkpoint["state_dict"]) # Luozhou: modify this line

        processor_state_dict = checkpoint["state_dict"]
        load_processor_state_dict(transformer, processor_state_dict)

        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        print(f"Resuming from global step: {global_step}")
    else:
        global_step = 0

    print("===== Memory before training =====")
    reset_memory("cuda")
    print_memory("cuda")

    # Train!
    total_batch_size = args.train_batch_size
    print("***** Running training *****")
    print(f"  Num trainable parameters = {num_trainable_parameters}")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches each epoch = {len(train_dataloader)}")
    print(f"  Num epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Total optimization steps = {args.max_train_steps}")

    first_epoch = 0
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
    )
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            with torch.no_grad():
                z = batch["z"].to("cuda")
                eps = batch["eps"].to("cuda")
                sigma = batch["sigma"].to("cuda")
                prompt_embeds = batch["prompt_embeds"].to("cuda")
                prompt_attention_mask = batch["prompt_attention_mask"].to("cuda")

                all_attention_mask = prepare_attention_mask(
                    prompt_attention_mask=prompt_attention_mask, 
                    latents=z
                )
                
                sigma_bcthw = sigma[:, None, None, None, None]  # [B, 1, 1, 1, 1]
                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                z_sigma = (1 - sigma_bcthw) * z + sigma_bcthw * eps
                ut = z - eps

                # (1 - sigma) because of
                # https://github.com/genmoai/mochi/blob/aba74c1b5e0755b1fa3343d9e4bd22e89de77ab1/src/genmo/mochi_preview/dit/joint_model/asymm_models_joint.py#L656
                # Also, we operate on the scaled version of the `timesteps` directly in the `diffusers` implementation.
                timesteps = (1 - sigma) * scheduler.config.num_train_timesteps

                z_sigma = torch.cat([z_sigma] * 2)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timesteps = timesteps.expand(z_sigma.shape[0]).to(z_sigma.dtype)
                
            with torch.autocast("cuda", torch.bfloat16):
                model_pred = transformer(
                    hidden_states=z_sigma,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=all_attention_mask,
                    timestep=timesteps,
                    return_dict=False,
                )[0]
            assert model_pred.shape == z.shape
            print(model_pred.shape) 
            seq_len_ = model_pred.shape[2]
            loss_rgb = F.mse_loss(model_pred[:,:,:seq_len_//2].float(), ut[:,:,:seq_len_//2].float())
            loss_alpha = F.mse_loss(model_pred[:,:,seq_len_//2:].float(), ut[:,:,seq_len_//2:].float())
            print(model_pred[:,:,seq_len_//2:].shape)
            alpha_dice_loss, pred_img, target_img = latent_mask_loss(
                model_pred[:,:,seq_len_//2:].float(),
                ut[:,:,seq_len_//2:].float()
            )
            # could also try coundry loss
            alpha_dice_loss = 0
            loss = (loss_rgb + loss_alpha + alpha_dice_loss)/3
            loss.backward() 
            if global_step % 16 == 15:
                optimizer.step()
                optimizer.zero_grad()
            lr_scheduler.step()
            

            progress_bar.update(1)
            

            last_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else args.learning_rate
            logs = {"loss": loss.detach().item(), "lr": last_lr, "loss_alpha": loss_alpha, "loss_rgb": loss_rgb, "alpha_dice_loss": alpha_dice_loss}
            progress_bar.set_postfix(**logs)
            if wandb_run:
                wandb_run.log(logs, step=global_step)
                wandb_run.log({"pred_img": wandb.Image(pred_img[0,-1]), "target_img": wandb.Image(target_img[0,-1]), "sigma": sigma[0]}, step=global_step)

            if args.checkpointing_steps is not None and global_step % args.checkpointing_steps == 0:
                print(f"Saving checkpoint at step {global_step}")
                checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pt")
                save_checkpoint(
                    transformer,
                    optimizer,
                    lr_scheduler,
                    global_step,
                    checkpoint_path,
                )
            
            if global_step % args.validation_steps == 0:
                print("===== Memory before validation =====")
                print_memory("cuda")

                transformer.eval()
                pipe = MochiPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    transformer=transformer,
                    scheduler=scheduler,
                    revision=args.revision,
                    variant=args.variant,
                )

                if args.enable_slicing:
                    pipe.vae.enable_slicing()
                if args.enable_tiling:
                    pipe.vae.enable_tiling()
                if args.enable_model_cpu_offload:
                    pipe.enable_model_cpu_offload()

                # validation_prompts = args.validation_prompt.split(args.validation_prompt_separator)
                validation_prompts = [
                    "A devil scorpionfish moves slowly across the seafloor, crawling with small, deliberate motions. The fish's body is excellently camouflaged, blending into the rocky, algae-covered environment with its mottled texture, muted colors, and irregular outline, making it very hard to see against the background.",
                ]
                for validation_prompt in validation_prompts:
                    pipeline_args = {
                        "prompt": validation_prompt,
                        "negative_prompt": "distinct outlines, brightly colored, standing out, highly visible, unnatural colors, vibrant tones, sharp borders, pixelation, low resolution, visible text, overexposed, blurred, artificial body shapes",
                        "num_frames": 1 if args.single_frame else 37,
                        "num_inference_steps": 64,
                        "height": args.height,
                        "width": args.width,
                        "max_sequence_length": 512,
                    }
                    log_validation(
                        pipe=pipe,
                        args=args,
                        pipeline_args=pipeline_args,
                        step=global_step,
                        wandb_run=wandb_run,
                    )

                print("===== Memory after validation =====")
                print_memory("cuda")
                reset_memory("cuda")

                del pipe.text_encoder
                del pipe.vae
                del pipe
                gc.collect()
                torch.cuda.empty_cache()

                transformer.train()
            global_step += 1
            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    transformer.eval()

    # saving lora weights
    # transformer_lora_layers = get_peft_model_state_dict(transformer)
    # MochiPipeline.save_lora_weights(save_directory=args.output_dir, transformer_lora_layers=transformer_lora_layers)

    # Cleanup trained models to save memory
    del transformer

    gc.collect()
    torch.cuda.empty_cache()

    # Final test inference
    # validation_outputs = []
    # if args.validation_prompt and args.num_validation_videos > 0:
    #     print("===== Memory before testing =====")
    #     print_memory("cuda")
    #     reset_memory("cuda")

    #     pipe = MochiPipeline.from_pretrained(
    #         args.pretrained_model_name_or_path,
    #         revision=args.revision,
    #         variant=args.variant,
    #     )



    #     if args.enable_slicing:
    #         pipe.vae.enable_slicing()
    #     if args.enable_tiling:
    #         pipe.vae.enable_tiling()
    #     if args.enable_model_cpu_offload:
    #         pipe.enable_model_cpu_offload()

    #     # Load LoRA weights
    #     # lora_scaling = args.lora_alpha / args.rank
    #     # pipe.load_lora_weights(args.output_dir, adapter_name="mochi-lora")
    #     # pipe.set_adapters(["mochi-lora"], [lora_scaling])

    #     # Run inference
    #     validation_prompts = args.validation_prompt.split(args.validation_prompt_separator)
    #     for validation_prompt in validation_prompts:
    #         pipeline_args = {
    #             "prompt": validation_prompt,
    #             "guidance_scale": 6.0,
    #             "num_inference_steps": 64,
    #             "height": args.height,
    #             "width": args.width,
    #             "max_sequence_length": 256,
    #         }

    #         video = log_validation(
    #             pipe=pipe,
    #             args=args,
    #             pipeline_args=pipeline_args,
    #             epoch=epoch,
    #             wandb_run=wandb_run,
    #             is_final_validation=True,
    #         )
    #         validation_outputs.extend(video)

    #     print("===== Memory after testing =====")
    #     print_memory("cuda")
    #     reset_memory("cuda")
    #     torch.cuda.synchronize("cuda")



if __name__ == "__main__":
    args = get_args()
    main(args)
