import click
import torch
import numpy as np
import torchvision
from pathlib import Path
from diffusers import AutoencoderKLMochi, MochiPipeline
from transformers import T5EncoderModel, T5Tokenizer
from tqdm.auto import tqdm
from diffusers.utils import export_to_video
vae = AutoencoderKLMochi.from_pretrained("genmo/mochi-1-preview", subfolder="vae", torch_dtype=torch.float32).to("cuda")
latten = torch.load("final_dataset/cuttlefish_4_002.latent.pt")["ldist"].float()

mean = (
    torch.tensor(vae.config.latents_mean).view(1, 12, 1, 1, 1).to(latten.device, latten.dtype)
)
std = (
    torch.tensor(vae.config.latents_std).view(1, 12, 1, 1, 1).to(latten.device, latten.dtype)
)
print(latten.shape)
vae.enable_slicing()
vae.enable_tiling()
data = latten[:,:12] * std / vae.config.scaling_factor + mean
with torch.no_grad(): 
    video = vae.decode(data)["sample"].float().cpu().permute(0, 2, 3, 4, 1).numpy()
print(video.shape)
video = (video - video.min())/(video.max() - video.min()) 
print(video.max())
video = [np.array(frame) for frame in video[0]]
print(video[0].shape)
export_to_video(video, "video.mp4")