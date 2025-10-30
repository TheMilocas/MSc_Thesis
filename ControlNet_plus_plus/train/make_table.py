import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

DATASET_NAME = "Milocas/celebahq_clean"
OCCLUDED_DIR = "./DATASET_EVAL/images_ocluded_table"
GEN_DIR_10 = "./COMPARE_IDS_CONTROLNET/grid_10x10" #"./COMPARE_IDS_CONTROLNET/masked_eval"
SIMS_PATH = "./COMPARE_IDS_CONTROLNET/sims_10x10.csv" #"./COMPARE_IDS_CONTROLNET/masked_eval/sims_masked_10x10.csv"

print("Loading dataset...")
dataset = load_dataset(DATASET_NAME, split="train")

def load_unoccluded_imgs(dataset, N):
    return [dataset[i]["image"].convert("RGB") for i in range(N)]

def load_occluded_imgs(N, folder=OCCLUDED_DIR):
    return [Image.open(os.path.join(folder, f"{i}.png")).convert("RGB") for i in range(N)]

def build_grid(unoccluded_imgs, occluded_imgs, N, gen_dir, save_img, sims_csv=None):
    sims = None
    if sims_csv and os.path.exists(sims_csv):
        sims = pd.read_csv(sims_csv, index_col=0)

    fig, axes = plt.subplots(N+1, N+1, figsize=(2*N, 2*N))

    axes[0, 0].axis("off")

    for j in range(N):
        axes[0, j+1].imshow(unoccluded_imgs[j])
        axes[0, j+1].axis("off")

    for i in range(N):
        axes[i+1, 0].imshow(occluded_imgs[i])
        axes[i+1, 0].axis("off")

    for i in range(N):
        for j in range(N):
            fname = f"gen_row{i}_col{j}.png"
            path = os.path.join(gen_dir, fname)

            if os.path.exists(path):
                gen_img = Image.open(path).convert("RGB")
                axes[i+1, j+1].imshow(gen_img)
            else:
                axes[i+1, j+1].imshow(Image.new("RGB", (64, 64), (255, 255, 255)))

            axes[i+1, j+1].axis("off")

            if sims is not None:
                sim_val = sims.iloc[i, j]
                axes[i+1, j+1].text(
                    0.5, -0.05, f"{sim_val:.4f}",
                    fontsize=12, ha="center", va="top",
                    transform=axes[i+1, j+1].transAxes
                )

    plt.tight_layout()
    plt.savefig(save_img, dpi=200)
    plt.close()

unoccluded_imgs10 = load_unoccluded_imgs(dataset, 10)
occluded_imgs10 = load_occluded_imgs(10)

build_grid(unoccluded_imgs10, occluded_imgs10,
           N=10,
           gen_dir=GEN_DIR_10,
           save_img="grid10x10.png",
           sims_csv=SIMS_PATH)

build_grid(unoccluded_imgs10[:6], occluded_imgs10[:6],
           N=6,
           gen_dir=GEN_DIR_10,
           save_img="grid6x6.png",
           sims_csv=SIMS_PATH)
