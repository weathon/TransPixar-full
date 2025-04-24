import torch
import os
files = os.listdir("video_rgb_")
os.makedirs("final_dataset", exist_ok=True)
for file in files:
    if not file.endswith("latent.pt"):
        continue
    tensor1 = torch.load("video_rgb_/"+file)["ldist"]
    tensor2 = torch.load("video_alpha/"+file)["ldist"]
    assert (torch.abs(tensor1 - tensor2) > 0.01).any()
    res = torch.cat([tensor1[:], tensor2[:]], dim=2)
    print(res.shape)
    torch.save({"ldist": res}, "final_dataset/"+file)