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
import os

vae = AutoencoderKLMochi.from_pretrained("genmo/mochi-1-preview", subfolder="vae", torch_dtype=torch.float32).to("cuda")
files = os.listdir("./video_alpha/")
files = [i for i in files if i.endswith("latent.pt")]
file = files[100]
latent_alpha = torch.load("./video_alpha/" + file, weights_only=False)["ldist"].float()
# latent_rgb = torch.load("./video_rgb_/cuttlefish_4_002.latent.pt")["ldist"].float()

# rgb_gray = latent_rgb[0].mean(0)[0]
# rgb_gray = (rgb_gray - rgb_gray.min())/(rgb_gray.max() - rgb_gray.min()) * 50 - 25
# rgb_gray = torch.sigmoid(rgb_gray).cpu().numpy() * 255
# cv2.imwrite("rgb.png", rgb_gray)

alpha_gray = latent_alpha[0].mean(0)[0]
alpha_gray = (alpha_gray - alpha_gray.min())/(alpha_gray.max() - alpha_gray.min()) * 5 - 2.5
alpha_gray = torch.sigmoid(alpha_gray).cpu().numpy() * 255
alpha_gray = cv2.resize(alpha_gray, (512, 512))
cv2.imwrite("alpha.png", alpha_gray)

# print(torch.nn.L1Loss()(latent_alpha, latent_rgb))
# print(latent_alpha.std())
# print(latent_rgb.shape)
# print(torch.nn.functional.cosine_similarity(latent_alpha, latent_rgb, 1).mean())
