"""
===========================================================
Visualize Generated Images and Similarity Scores
===========================================================

This script generates grid images showing:
    - Original images (columns)
    - Masked images (rows)
    - Generated images (grid cells)
    - Cosine similarity scores below each generated image

Inputs:
    - Original and masked images
    - Generated images
    - CSV file of similarity scores from ArcFace evaluation

Role in Pipeline:
    - Provides a **visual summary** of identity preservation 
      and inpainting quality, complementing the numerical evaluation.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

GEN_DIR = "./COMPARE_IDS_CONTROLNET/grid_10x10"
ORIG_DIR = "./COMPARE_IDS_CONTROLNET/original"      
MASK_PATH = "mask.png"                             
CSV_PATH = "./COMPARE_IDS_CONTROLNET/comparison_results.csv"

OUTPUT_IMAGE_FULL = "./COMPARE_IDS_CONTROLNET/comparison_grid_10x10.png"
OUTPUT_IMAGE_SMALL = "./COMPARE_IDS_CONTROLNET/comparison_grid_6x6.png"

CELL_SIZE = (112, 112)
PADDING = 20      
FONT_SIZE = 30        

def apply_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Apply a binary mask to an image.
    White (255) in mask = occluded (set to black)
    Black (0) = keep original pixel.
    """
    mask = mask.convert("L").resize(image.size)
    inverted_mask = Image.fromarray(255 - np.array(mask))
    masked = Image.composite(image, Image.new("RGB", image.size, "black"), inverted_mask)
    return masked


def create_grid_image(scores, n, output_path, orig_dir, mask_path, gen_dir):
    """Create an n×n grid visualization with headers and similarity text below."""
    print(f"Creating {n}x{n} grid image...")

    orig_images = [
        Image.open(os.path.join(orig_dir, f"{i}.png")).convert("RGB").resize(CELL_SIZE)
        for i in range(n)
    ]

    mask = Image.open(mask_path).convert("L")

    masked_images = [apply_mask(img, mask).resize(CELL_SIZE) for img in orig_images]

    cols = n + 1
    rows = n + 1
    grid_w = cols * (CELL_SIZE[0] + PADDING)
    grid_h = rows * (CELL_SIZE[1] + PADDING + 10) 
    canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arialbd.ttf", FONT_SIZE) 
    except:
        font = ImageFont.load_default()

    for col in range(n):
        x = (col + 1) * (CELL_SIZE[0] + PADDING)
        y = 0
        canvas.paste(orig_images[col], (x, y))

    for row in range(n):
        x = 0
        y = (row + 1) * (CELL_SIZE[1] + PADDING + 10)
        canvas.paste(masked_images[row], (x, y))

    for row in range(n):
        for col in range(n):
            filename = f"gen_row{row}_col{col}.png"
            img_path = os.path.join(gen_dir, filename)
            if not os.path.exists(img_path):
                continue
            img = Image.open(img_path).convert("RGB").resize(CELL_SIZE)

            x = (col + 1) * (CELL_SIZE[0] + PADDING)
            y = (row + 1) * (CELL_SIZE[1] + PADDING + 10)
            canvas.paste(img, (x, y))

            score = scores[row, col]
            text = f"{score:.4f}" 
            
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            except AttributeError:
                text_w, text_h = draw.textsize(text, font=font)
            
            text_x = x + (CELL_SIZE[0] - text_w) / 2
            text_y = y + CELL_SIZE[1] + 5 
            draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)

    canvas.save(output_path)


if __name__ == "__main__":
    print("Loading similarity scores...")
    scores = np.loadtxt(CSV_PATH, delimiter=",", skiprows=1)

    create_grid_image(
        scores,
        n=10,
        output_path=OUTPUT_IMAGE_FULL,
        orig_dir=ORIG_DIR,
        mask_path=MASK_PATH,
        gen_dir=GEN_DIR,
    )

    create_grid_image(
        scores[:6, :6],
        n=6,
        output_path=OUTPUT_IMAGE_SMALL,
        orig_dir=ORIG_DIR,
        mask_path=MASK_PATH,
        gen_dir=GEN_DIR,
    )
