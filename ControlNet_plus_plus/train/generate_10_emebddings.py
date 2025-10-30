"""
===========================================================
Compute ArcFace Similarities for Generated Images
===========================================================

This script calculates the cosine similarity between generated 
images (produced by ControlNet-guided inpainting) and reference 
identity embeddings using the ArcFace model. 

Output:
    - A CSV file containing a similarity matrix of all generated 
      images against reference embeddings.

Role in Pipeline:
    - Provides a **quantitative evaluation** of identity preservation 
      in the generated images, used later for visualization and analysis.
"""

#!/usr/bin/env python3
import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from backbones.iresnet import iresnet100

GEN_DIR = "./COMPARE_IDS_CONTROLNET/grid_10x10"
MODEL_PATH = "../../ARCFACE/models/R100_MS1MV3/backbone.pth"
BASE_PATH = "../../datasets/celebahq/conditions_no_mask"
OUTPUT_CSV = "comparison_results.csv"

class ARCFACE(torch.nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        self.model = iresnet100(pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device).eval()
        self.device = device

    def forward(self, x):
        with torch.no_grad():
            x = self.model(x.to(self.device))
            x = F.normalize(x, p=2, dim=1)
        return x

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

def get_embedding_from_image(img: Image.Image, arcface: ARCFACE):
    img = img.convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(arcface.device)
    emb = arcface(img_tensor).squeeze(0).cpu().numpy()
    return emb

def compare_generated_images_with_refs():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arcface = ARCFACE(MODEL_PATH, device)

    npy_files = sorted([f for f in os.listdir(BASE_PATH) if f.endswith(".npy")])[:10]
    
    reference_embeddings = [
        np.load(os.path.join(BASE_PATH, f)).astype(np.float32) for f in npy_files
    ]
    results = []

    for row in range(10):
            row_sims = []
            ref_emb = reference_embeddings[row] 
            for col in range(10): 
                filename = f"gen_row{row}_col{col}.png"
                img_path = os.path.join(GEN_DIR, filename)
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"Missing file: {img_path}")
        
                img = Image.open(img_path).convert("RGB")
                emb = get_embedding_from_image(img, arcface)
        
                sim = np.dot(emb, ref_emb) / (np.linalg.norm(emb) * np.linalg.norm(ref_emb))
                row_sims.append(sim)
        
            results.append(row_sims)

    np.savetxt(
        OUTPUT_CSV,
        np.array(results).T,
        delimiter=",",
        fmt="%.6f",
        header=",".join([f"col{c}" for c in range(10)]),
        comments=""
    )
    print(f"\nSaved results to {OUTPUT_CSV}")

if __name__ == "__main__":
    compare_generated_images_with_refs()
