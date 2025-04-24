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
latten1 = torch.load("final_dataset/cuttlefish_4_002.latent.pt")["ldist"].float()
# latten2 = torch.load("/home/wg25r/make_it_move/TransPixar.bak/Mochi/video_rgb/crab_1_000.latent.pt")["ldist"].float()
latten = latten1#torch.zeros_like(latten2)
# for i in range(latten2.shape[2]):
#     latten[:,:,i] = (i/latten2.shape[2]) * latten1[:,:,i] + (1 - i/latten2.shape[2]) * latten2[:,:,i]

# latten = torch.cat([latten1[:,:6], latten2[:,6:12]], dim=1)

sample_mean = latten1[:,:12]
sample_std = latten1[:,12:]

# sample latten
# latten = sample_mean + torch.exp(sample_std) * torch.randn_like(sample_mean[:,0])[None,:]
# from PIL import Image
# img = np.array(sample_mean[0].abs().mean(0)[0].cpu())
# # img = (img - img.min())/(img.max() - img.min())
# import scipy.special as scipy
# img = scipy.softmax(img)
# img = (img/img.max()) * 255
# img = img.astype(np.uint8) 
# img = Image.fromarray(img)
# img.save("sample_mean.png")


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