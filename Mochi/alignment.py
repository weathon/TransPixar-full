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
latent_alpha = torch.load("/home/wg25r/make_it_move/TransPixar-full/Mochi/video_alpha/cuttlefish_4_002.latent.pt")["ldist"].float()
latent_rgb = torch.load("/home/wg25r/make_it_move/TransPixar-full/Mochi/video_rgb/cuttlefish_4_002.latent.pt")["ldist"].float()
