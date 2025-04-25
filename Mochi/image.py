import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

def generate_opposite_image(img, max_iters=1000, lr=0.01):
    img = img.detach()
    target_mean = img.mean()
    C, H, W = img.shape
    img_flat = img.view(C, -1).t()

    other = torch.rand_like(img, requires_grad=True)
    optimizer = torch.optim.Adam([other], lr=lr)

    for _ in range(max_iters):
        optimizer.zero_grad()
        other_flat = other.view(C, -1).t()

        abs_diff = F.l1_loss(other_flat, img_flat, reduction='none').sum(dim=1).mean()
        loss_abs = -abs_diff

        cos_sim = F.cosine_similarity(other_flat, img_flat, dim=1).mean()
        loss_cos = -cos_sim * 2

        loss = loss_abs + loss_cos
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            other.clamp_(0, 1)

    return other.detach() 

# Load and transform image
img_path = "/home/wg25r/make_it_move/alex.jpg"
img = Image.open(img_path).convert("RGB")
transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
img_tensor = transform(img)

# Process
result = generate_opposite_image(img_tensor)

# Show original and result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_tensor.permute(1, 2, 0))
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result.permute(1, 2, 0))
plt.title("Opposite Image")
plt.axis("off")
plt.show()
plt.savefig("opposite_image.png")
