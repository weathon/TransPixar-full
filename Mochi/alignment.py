import click
import torch
import numpy as np
import torchvision
from pathlib import Path
from diffusers import AutoencoderKLMochi, MochiPipeline
from transformers import T5EncoderModel, T5Tokenizer
from tqdm.auto import tqdm
from diffusers.utils import export_to_video
import cv2

vae = AutoencoderKLMochi.from_pretrained("genmo/mochi-1-preview", subfolder="vae", torch_dtype=torch.float32).to("cuda")
latent_alpha = torch.load("./video_alpha/cuttlefish_4_002.latent.pt")["ldist"].float()
latent_rgb = torch.load("./video_rgb_/cuttlefish_4_002.latent.pt")["ldist"].float()

rgb_gray = latent_rgb[0].mean(0)[0].cpu().numpy()
rgb_gray = (rgb_gray - rgb_gray.min())/(rgb_gray.max() - rgb_gray.min()) * 255
cv2.imwrite("rgb.png", rgb_gray)

alpha_gray = latent_alpha[0].mean(0)[0].cpu().numpy()
alpha_gray = (alpha_gray - alpha_gray.min())/(alpha_gray.max() - alpha_gray.min()) * 255
cv2.imwrite("alpha.png", alpha_gray)