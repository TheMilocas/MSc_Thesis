"""
===========================================================
Generate Images with ControlNet for Identity Evaluation
===========================================================

This script uses ControlNet-guided inpainting to generate images 
from occluded inputs using different identity embeddings.

Outputs:
    - Generated images saved in a grid-friendly folder structure, 
      ready for similarity evaluation and visualization.

Role in Pipeline:
    - Produces the **images** that are evaluated numerically (script 1) 
      and visualized (script 2), forming the first step in the pipeline.
"""


import sys
sys.path.append("diffusers_new/src")

import os
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torchvision import transforms
from datasets import load_dataset

from diffusers import StableDiffusionControlNetInpaintPipeline, DDIMScheduler
from diffusers_new.src.diffusers.models.controlnets.controlnet1 import ControlNetModel
from backbones.iresnet import iresnet100

CONTROLNET_PATH = "../../identity_controlnet_final"
DATASET_NAME = "Milocas/celebahq_clean"
IMG_DIR = "./DATASET_EVAL/images_ocluded_table"
MASK_DIR = "./DATASET_EVAL/masks_table"
NUM_SAMPLES = 10
SAVE_DIR = "./COMPARE_IDS_CONTROLNET"

os.makedirs(SAVE_DIR, exist_ok=True)
grid_dir = os.path.join(SAVE_DIR, f"grid_{NUM_SAMPLES}x{NUM_SAMPLES}")
os.makedirs(grid_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading dataset...")
dataset = load_dataset(DATASET_NAME, split="train")
samples = [dataset[i] for i in range(NUM_SAMPLES)]

occluded_imgs, masks, embeddings, names = [], [], [], []
for i, sample in enumerate(samples):
    name_img = f"{i}.png"
    names.append(name_img)

    image_masked_path = os.path.join(IMG_DIR, name_img)
    mask_path = os.path.join(MASK_DIR, name_img)

    occluded_imgs.append(Image.open(image_masked_path).convert("RGB").resize((512, 512)))
    masks.append(Image.open(mask_path).convert("L").resize((512, 512)))
    embeddings.append(torch.from_numpy(np.load("../../" + sample["condition"])).squeeze(0))

embeddings = torch.stack(embeddings)

print("Loading ControlNet")
controlnet = ControlNetModel.from_pretrained(CONTROLNET_PATH, torch_dtype=torch.float32).to(device)

print("Loading pipeline with ControlNet")
pipe_controlnet = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    controlnet=controlnet,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    safety_checker=None, requires_safety_checker=False
).to(device)
pipe_controlnet.scheduler = DDIMScheduler.from_config(pipe_controlnet.scheduler.config)
pipe_controlnet.enable_model_cpu_offload()

similarities = np.zeros((NUM_SAMPLES, NUM_SAMPLES))

for i in range(NUM_SAMPLES):      
    for j in range(NUM_SAMPLES):   
        print(f"Generating [{i},{j}]...")

        with torch.no_grad(), torch.autocast(device.type):
            gen_img = pipe_controlnet(
                prompt="",
                image=occluded_imgs[i],
                mask_image=masks[i],
                control_image=embeddings[j].unsqueeze(0).to(device),
                num_inference_steps=25,
                generator=torch.Generator(device).manual_seed(1000 + i*NUM_SAMPLES + j),
            ).images[0]

        gen_path = os.path.join(grid_dir, f"gen_row{i}_col{j}.png")
        gen_img.save(gen_path)
