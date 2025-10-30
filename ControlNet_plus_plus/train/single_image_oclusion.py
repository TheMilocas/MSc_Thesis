"""
===========================================================
Single Image Occlusion Evaluation with ArcFace
===========================================================

Purpose:
    Evaluate the impact of occluding part of a face on identity 
    recognition using the ArcFace model.

Inputs:
    - Original face image: e.g., './SAME_ID/images/1.jpg'
    - Mask image: e.g., 'mask.png' (white areas indicate occlusion)
    - ArcFace model weights: e.g., '../../ARCFACE/models/R100_MS1MV3/backbone.pth'

Outputs:
    - Cosine similarity score between original and occluded embeddings
    - Optional visualization showing:
        - Original image
        - Occluded image
        - Cosine similarity score
"""


import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F
from backbones.iresnet import iresnet100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = "./SAME_ID/images"  
mask_path = "mask.png"  
arcface_model_path = "../../ARCFACE/models/R100_MS1MV3/backbone.pth"

class ARCFACE(torch.nn.Module):
    def __init__(self, model_path, device='cuda'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = iresnet100(pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()
    
    def forward(self, x):
        with torch.no_grad():
            x = self.model(x.to(self.device))
            x = F.normalize(x, p=2, dim=1)
        return x

arcface = ARCFACE(arcface_model_path, device=device)

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

def preprocess_image(image):
    """Resize + normalize an image to ArcFace input."""
    return transform(image).unsqueeze(0).to(device)

def compute_embedding(image):
    tensor = preprocess_image(image)
    return arcface(tensor).squeeze(0)

def cosine_similarity(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

orig_path = os.path.join(data_dir, "1.jpg")
if not os.path.exists(orig_path):
    raise FileNotFoundError(f"Original image not found at {orig_path}")

mask = Image.open(mask_path).convert("L").resize((512, 512))  
original = Image.open(orig_path).convert("RGB").resize((512, 512))

original_np = np.array(original)
mask_np = np.array(mask) / 255.0 
occluded_np = original_np * (1 - mask_np[..., None]) + 255 * mask_np[..., None]  
occluded_image = Image.fromarray(occluded_np.astype(np.uint8))

orig_emb = compute_embedding(original)
occl_emb = compute_embedding(occluded_image)

cosine = cosine_similarity(orig_emb, occl_emb)
print(f"Cosine similarity (original vs occluded): {cosine:.4f}")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(original)
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(occluded_image)
axes[1].set_title(f"Occluded\nCosine: {cosine:.4f}")
axes[1].axis("off")

plt.tight_layout()
plt.show()
