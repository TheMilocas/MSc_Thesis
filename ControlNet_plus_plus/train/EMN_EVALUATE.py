"""
===========================================================
Visual Comparison of Masked Face Regions vs Original Faces
===========================================================

Purpose:
    Compare the identity similarity of masked face regions (eyes, nose, mouth)
    against the original full face using ArcFace embeddings.

Inputs:
    - CelebA-HQ clean dataset: `DATASET_NAME`
    - Precomputed identity embeddings: `EMBEDDING_PREFIX`
    - Masked regions images: `MASK_DIR` (organized by region: eyes_big_vertical, nose, mouth)
    - ArcFace pretrained model: "../../ARCFACE/models/R100_MS1MV3/backbone.pth"

Process:
    1. Randomly select a few samples from the dataset (defined in `SAMPLES`).
    2. Load the original image and masked images for each facial region.
    3. Compute ArcFace embeddings for the original and each masked region.
    4. Compute cosine similarity between the original embedding and each masked embedding.
    5. Generate a visual strip for each sample:
        - Original image on the left
        - Masked images on the right with similarity scores annotated

Outputs:
    - Comparison strips showing masked regions and identity similarity.
    - Saved in `OUTPUT_DIR` as PNG files, e.g., `comparison_5.png`, `comparison_10.png`.

Use Case:
    - Visualize how occluding different facial regions affects identity recognition.
    - Useful for evaluating robustness of face recognition under occlusions.
"""

import os
import random
import torch
import numpy as np
from torchvision import transforms
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
from backbones.iresnet import iresnet100
import torch.nn as nn
from torch.nn import functional as F

DATASET_NAME = "Milocas/celebahq_clean"
MASK_DIR = "./DATASET_EVAL/masks" 
EMBEDDING_PREFIX = "../../"
OUTPUT_DIR = "./comparisons_random"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLES = [5, 10]  
MASK_NAMES = {
    "eyes_big_vertical": "Eyes",
    "nose": "Nose",
    "mouth": "Mouth"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ARCFACE(nn.Module):
    def __init__(self, model_path: str, device: str = 'cuda'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = iresnet100(pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.model(x.to(self.device))
            x = F.normalize(x, p=2, dim=1)
        return x

def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a.view(-1), b.view(-1), dim=0).item()

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def image_to_tensor(img):
    return transform(img).unsqueeze(0)

def create_strip(samples, save_path):
    font_size = 48  
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "arial.ttf"
    ]
    font = None
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except IOError:
            continue
    if font is None:
        print("Warning: Could not find a TTF font file, using default font.")
        font = ImageFont.load_default()

    img_w, img_h = samples[0]["original"].size

    x_spacing = 50 
    y_spacing = 100 

    total_w = (len(MASK_NAMES) + 1) * img_w + x_spacing * len(MASK_NAMES)
    total_h = len(samples) * img_h + y_spacing * len(samples)

    result = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(result)

    y_offset = 0
    for sample in samples:
        result.paste(sample["original"], (0, y_offset))

        for j, (mask_key, mask_label) in enumerate(MASK_NAMES.items()):
            x = (j + 1) * img_w + j * x_spacing
            result.paste(sample["masked"][mask_key], (x, y_offset))
            sim = sample["sims"][mask_key]

            text = f"{mask_label}: {sim:.3f}"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            draw.text(
                (x + (img_w - text_w)//2, y_offset + img_h + 10),
                text,
                fill="black",
                font=font
            )

        y_offset += img_h + y_spacing

    result.save(save_path)
    print(f"Saved: {save_path}")

def main():
    arcface_model_path = "../../ARCFACE/models/R100_MS1MV3/backbone.pth"
    arcface = ARCFACE(arcface_model_path)

    dataset = load_dataset(DATASET_NAME, split="train")

    for N in SAMPLES:
        chosen = random.sample(range(2947), N)
        samples = []

        for idx in chosen:
            sample = dataset[idx]
            
            image = sample["image"].resize((512, 512))
            embedding = torch.from_numpy(np.load(EMBEDDING_PREFIX + sample["condition"])).unsqueeze(0).to(device)

            masked_images = {}
            sims = {}
            for mask_key in MASK_NAMES.keys():
                mask_path = os.path.join(MASK_DIR, mask_key, f"sample_{idx}.png")

                masked_img = Image.open(mask_path).convert("RGB").resize((512, 512))
                masked_images[mask_key] = masked_img

                masked_emb = arcface(image_to_tensor(masked_img).to(device)).squeeze(0)
                sims[mask_key] = cosine_similarity(embedding.squeeze(0), masked_emb)

            samples.append({
                "original": image,
                "masked": masked_images,
                "sims": sims
            })

        save_path = os.path.join(OUTPUT_DIR, f"comparison_{N}.png")
        create_strip(samples, save_path)

if __name__ == "__main__":
    main()
