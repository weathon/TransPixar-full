import torch
import os
files = os.listdir("dataset/video_rgb")
os.makedirs("final_dataset2", exist_ok=True)
for file in files:
    if not file.endswith("latent.pt"):
        continue
    tensor1 = torch.load("dataset/video_rgb/"+file)["ldist"]
    tensor2 = torch.load("dataset/video_alpha/"+file)["ldist"]
    assert (torch.abs(tensor1 - tensor2) > 0.01).any()
    res = torch.cat([tensor1[:], tensor2[:]], dim=2)
    print(res.shape)
    torch.save({"ldist": res}, "final_dataset2/"+file)