#!/usr/bin/env python3

"""
Evaluation script for image generation experiments (JPG images).

Description:
    This script evaluates generated images produced by ControlNet and a baseline method.
    For each method:
      1. Loads generated images (.jpg) and matches them with test dataset samples.
      2. Computes per-image LPIPS (Learned Perceptual Image Patch Similarity) scores.
      3. Computes FID (Frechet Inception Distance) and mFID (masked FID) distributions using bootstrapping.
      4. Saves summary results CSV and per-image LPIPS CSV.
      5. Generates boxplots for FID, mFID, and LPIPS distributions.

Inputs:
    - DATASET_NAME: HuggingFace dataset name with images + masks
    - CONTROLNET_DIR: Directory with ControlNet generated images (.jpg)
    - BASE_DIR: Directory with baseline generated images (.jpg)

Outputs:
    - CSV files with overall and per-image metrics
    - Boxplots for FID, mFID, and LPIPS
"""

import os
import re
import glob
import shutil
import tempfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset

from cleanfid import fid  # use fid.get_files_features and fid.frechet_distance
from cleanfid.features import build_feature_extractor
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# ---------------- CONFIG ----------------
DATASET_NAME = "Milocas/celebahq_clean"
CONTROLNET_DIR = "./TEST_CELEBAHQ/TEST/controlnet1"
BASE_DIR = "./TEST_CELEBAHQ/TEST/base1"
NEW_CSV_PATH = "./comparison_outputs_random_seed/results_overall_clean.csv"
LPIPS_PER_IMAGE_CSV = "./comparison_outputs_random_seed/lpips_per_image.csv"
BOOTSTRAP_ITERS = 200  # increase for smoother FID/mFID distributions (slower)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(os.path.dirname(NEW_CSV_PATH), exist_ok=True)

# ---------------- TRANSFORMS / METRICS ----------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
lpips_metric = LearnedPerceptualImagePatchSimilarity(normalize=True).to(DEVICE)
cf_model = build_feature_extractor(mode="clean", device=DEVICE)

# ---------------- HELPERS ----------------
def parse_sample_index(fname):
    # Match any numeric part before .png, with or without 'sample' prefix or leading zeros
    m = re.search(r"(?:sample[_-]?)?0*([0-9]+)\.png$", fname, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None

def apply_binary_mask_pil(img_pil: Image.Image, mask_pil: Image.Image):
    img = img_pil.convert("RGB").resize((256, 256))
    mask = mask_pil.convert("L").resize((256, 256))
    img_arr = np.array(img, dtype=np.uint8)
    mask_arr = (np.array(mask) > 127).astype(np.uint8)
    mask_3 = np.stack([mask_arr]*3, axis=-1)
    return Image.fromarray((img_arr * mask_3).astype(np.uint8))

def compute_features_for_paths(paths):
    """Return numpy array (N,featdim) with features for the given list of image file paths."""
    if len(paths) == 0:
        return np.zeros((0, 0))
    feats = fid.get_files_features(paths, cf_model)
    return np.array(feats)

def frechet_distance_from_feats(a_feats, b_feats):
    mu_a, cov_a = np.mean(a_feats, axis=0), np.cov(a_feats, rowvar=False)
    mu_b, cov_b = np.mean(b_feats, axis=0), np.cov(b_feats, rowvar=False)
    return fid.frechet_distance(mu_a, cov_a, mu_b, cov_b)

def bootstrap_fid_from_feats(a_feats, b_feats, n_iter=200):
    """
    Compute bootstrap FID samples by sampling indices with replacement.
    Returns array of length n_iter.
    """
    n = a_feats.shape[0]
    if n == 0:
        return np.array([])
    boot = []
    for _ in range(n_iter):
        idxs = np.random.randint(0, n, size=n)
        try:
            val = frechet_distance_from_feats(a_feats[idxs], b_feats[idxs])
        except Exception:
            # if numerical issues occur, fallback to computing on all
            val = frechet_distance_from_feats(a_feats, b_feats)
        boot.append(val)
    return np.array(boot)

# ---------------- LOAD DATASET ----------------
print("[INFO] Loading dataset...")
dataset = load_dataset(DATASET_NAME, split="test")
n_dataset = len(dataset)
print(f"[INFO] Dataset size: {n_dataset}")

# ---------------- MAIN ----------------
methods = {
    "ControlNet": CONTROLNET_DIR,
    "Baseline": BASE_DIR
}

all_results = []
lpips_rows = []  # collect per-image LPIPS rows for boxplot / CSV

# We'll use a short-lived temp dir so cleanfid can read files (auto-removed)
with tempfile.TemporaryDirectory() as tmpdir:
    print(f"[INFO] Using temporary dir {tmpdir} for intermediate images (auto-cleaned)")

    for method_name, method_dir in methods.items():
        print(f"\n[INFO] Evaluating {method_name}")

        # gather generated samples and parse indices
        gen_paths_raw = sorted(glob.glob(os.path.join(method_dir, "*.jpg")))
        idx_to_gen = {}
        for p in gen_paths_raw:
            idx = parse_sample_index(os.path.basename(p))
            if idx is None:
                # fallback: try numeric part before extension
                name = os.path.splitext(os.path.basename(p))[0]
                digits = re.findall(r"(\d+)", name)
                if digits:
                    idx = int(digits[-1])
            if idx is not None:
                idx_to_gen[idx] = p


        if len(idx_to_gen) == 0:
            raise RuntimeError(f"No matching generated images found in {method_dir}")

        # Sort sampled indices to create matched lists
        matched_indices = sorted(idx_to_gen.keys())
        print(f"[INFO] Found {len(matched_indices)} matching generated samples (will use these)")

        # Prepare lists of file paths (saved to tmpdir) for FID/mFID and arrays for LPIPS
        gen_tmp_paths = []
        real_tmp_paths = []
        mgen_tmp_paths = []
        mreal_tmp_paths = []

        per_image_lpips = []  # per-image LPIPS for this method

        for i, idx in enumerate(tqdm(matched_indices, desc=f"Preparing {method_name}")):
            gen_path = idx_to_gen[idx]
            # load dataset originals and masks
            orig = dataset[idx]['image']
            mask = dataset[idx]['mask']
            if not isinstance(orig, Image.Image):
                orig = Image.fromarray(np.array(orig))
            if not isinstance(mask, Image.Image):
                mask = Image.fromarray(np.array(mask))

            # generated image
            gen_img = Image.open(gen_path).convert("RGB").resize((256,256))
            # masked versions
            masked_gen = apply_binary_mask_pil(gen_img, mask)
            masked_real = apply_binary_mask_pil(orig, mask)
            # resized real
            real_img = orig.convert("RGB").resize((256,256))

            # save all four to tmpdir with deterministic names
            gen_tmp = os.path.join(tmpdir, f"{method_name}_gen_{i:06d}.png")
            real_tmp = os.path.join(tmpdir, f"{method_name}_real_{i:06d}.png")
            mgen_tmp = os.path.join(tmpdir, f"{method_name}_mgen_{i:06d}.png")
            mreal_tmp = os.path.join(tmpdir, f"{method_name}_mreal_{i:06d}.png")

            gen_img.save(gen_tmp)
            real_img.save(real_tmp)
            masked_gen.save(mgen_tmp)
            masked_real.save(mreal_tmp)

            gen_tmp_paths.append(gen_tmp)
            real_tmp_paths.append(real_tmp)
            mgen_tmp_paths.append(mgen_tmp)
            mreal_tmp_paths.append(mreal_tmp)

            # LPIPS per-image (gen vs original)
            g_t = transform(gen_img).unsqueeze(0).to(DEVICE)
            o_t = transform(real_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                score = lpips_metric(g_t, o_t).item()
            per_image_lpips.append(score)
            lpips_rows.append({"method": method_name, "index": idx, "lpips": score})

        # Compute features once for all saved files for speed
        print("[INFO] Extracting features for FID computations (this may take a while)...")
        gen_feats = compute_features_for_paths(gen_tmp_paths)
        real_feats = compute_features_for_paths(real_tmp_paths)
        mgen_feats = compute_features_for_paths(mgen_tmp_paths)
        mreal_feats = compute_features_for_paths(mreal_tmp_paths)

        # Bootstrap FID and mFID distributions
        print(f"[INFO] Bootstrapping FID/mFID distributions ({BOOTSTRAP_ITERS} iters)...")
        fid_boot = bootstrap_fid_from_feats(gen_feats, real_feats, n_iter=BOOTSTRAP_ITERS)
        mfid_boot = bootstrap_fid_from_feats(mgen_feats, mreal_feats, n_iter=BOOTSTRAP_ITERS)

        # Fallback if bootstrap arrays empty
        fid_mean, fid_std = (float(np.mean(fid_boot)), float(np.std(fid_boot))) if fid_boot.size else (np.nan, np.nan)
        mfid_mean, mfid_std = (float(np.mean(mfid_boot)), float(np.std(mfid_boot))) if mfid_boot.size else (np.nan, np.nan)

        # LPIPS stats
        lpips_mean = float(np.mean(per_image_lpips)) if per_image_lpips else float("nan")
        lpips_std = float(np.std(per_image_lpips)) if per_image_lpips else float("nan")

        all_results.append({
            "method": method_name,
            "n_samples": len(per_image_lpips),
            "FID_mean": fid_mean,
            "FID_std": fid_std,
            "mFID_mean": mfid_mean,
            "mFID_std": mfid_std,
            "LPIPS_mean": lpips_mean,
            "LPIPS_std": lpips_std
        })

        # Save bootstrap arrays for plotting later (attach to this method)
        # We'll store them in memory dictionaries for plotting outside the loop
        if 'fid_boots' not in locals():
            fid_boots = {}
            mfid_boots = {}
        fid_boots[method_name] = fid_boot
        mfid_boots[method_name] = mfid_boot

    # end of methods loop; tmpdir is still available inside with-block

# ---------------- SAVE RESULTS ----------------
results_df = pd.DataFrame(all_results)
results_df.to_csv(NEW_CSV_PATH, index=False)
print(f"\n[INFO] Saved summary results CSV to {NEW_CSV_PATH}")

# Save per-image LPIPS
lpips_df = pd.DataFrame(lpips_rows)
lpips_df.to_csv(LPIPS_PER_IMAGE_CSV, index=False)
print(f"[INFO] Saved per-image LPIPS to {LPIPS_PER_IMAGE_CSV}")

# ---------------- PLOTS ----------------
os.makedirs(os.path.dirname(NEW_CSV_PATH), exist_ok=True)
sns.set(style="whitegrid")

# 1) FID boxplot (bootstrap distributions)
fid_df = []
for method, arr in fid_boots.items():
    for v in arr:
        fid_df.append({"method": method, "value": v})
fid_df = pd.DataFrame(fid_df)

plt.figure(figsize=(6,5))
sns.boxplot(data=fid_df, x="method", y="value")
plt.title("FID (bootstrap distribution)")
plt.ylabel("FID")
out_fid = os.path.join(os.path.dirname(NEW_CSV_PATH), "FID_boxplot.png")
plt.savefig(out_fid)
plt.close()
print(f"[INFO] Saved FID boxplot to {out_fid}")

# 2) mFID boxplot (bootstrap distributions)
mfid_df = []
for method, arr in mfid_boots.items():
    for v in arr:
        mfid_df.append({"method": method, "value": v})
mfid_df = pd.DataFrame(mfid_df)

plt.figure(figsize=(6,5))
sns.boxplot(data=mfid_df, x="method", y="value")
plt.title("mFID (bootstrap distribution)")
plt.ylabel("mFID")
out_mfid = os.path.join(os.path.dirname(NEW_CSV_PATH), "mFID_boxplot.png")
plt.savefig(out_mfid)
plt.close()
print(f"[INFO] Saved mFID boxplot to {out_mfid}")

# 3) LPIPS per-image boxplot
plt.figure(figsize=(6,5))
sns.boxplot(data=lpips_df, x="method", y="lpips")
plt.title("LPIPS (per-image distribution)")
plt.ylabel("LPIPS")
out_lpips = os.path.join(os.path.dirname(NEW_CSV_PATH), "LPIPS_boxplot.png")
plt.savefig(out_lpips)
plt.close()
print(f"[INFO] Saved LPIPS boxplot to {out_lpips}")

# final summary print
print("\nSUMMARY:")
print(results_df.to_string(index=False))
