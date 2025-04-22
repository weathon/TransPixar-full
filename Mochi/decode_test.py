import click
import torch
import torchvision
from pathlib import Path
from diffusers import AutoencoderKLMochi, MochiPipeline
from transformers import T5EncoderModel, T5Tokenizer
from tqdm.auto import tqdm
from diffusers.utils import export_to_video
vae = AutoencoderKLMochi.from_pretrained("genmo/mochi-1-preview", subfolder="vae", torch_dtype=torch.float32).to("cuda")

latten = torch.load("final_dataset2/crab_1_002.latent.pt")["ldist"].float()
print(latten.shape)
vae.enable_slicing()
vae.enable_tiling()
with torch.no_grad():
    video = vae.decode(latten[:,:12])
export_to_video(video["sample"], "video.mp4")