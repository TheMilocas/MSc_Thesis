import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from io import StringIO

# -------- CONFIG --------
embedding_dir_test = "../../datasets/celebahq/test/original_embeddings"  # ground truth for PVA
embedding_dir_pva = "../../PVA-CelebAHQ-IDI/embeddings"
SAVE_PREFIX = "celeba-hq-dual-all"
CSV_PATH = "./TEST_CELEBAHQ/TEST/similarity_scores_celeba-hq.csv"  # previously saved CSV
SAVE_DIR = "TEST_CELEBAHQ/TEST"
# ------------------------

os.makedirs(SAVE_DIR, exist_ok=True)

# -------- LOAD PREVIOUS COSINE VALUES --------
# --- Load CSV safely, ignoring any extra summary/stat lines ---
with open(CSV_PATH, "r") as f:
    lines = []
    for line in f:
        if line.strip().startswith("#") or not line.strip():
            break  # stop reading at summary section or empty lines
        lines.append(line)

# Load clean portion into a DataFrame
df = pd.read_csv(StringIO("".join(lines)))
print(f"Loaded {len(df)} valid entries from {CSV_PATH}")

cosine_cn = df["cosine_controlnet"].to_numpy()  # reuse ControlNet values

# -------- RECOMPUTE PVA SIMILARITIES --------
pva_files = sorted([f for f in os.listdir(embedding_dir_pva) if f.endswith(".npy")])
test_files = sorted([f for f in os.listdir(embedding_dir_test) if f.endswith(".npy")])

# Ensure matching count
assert len(pva_files) == len(df), f"Mismatch: {len(pva_files)} PVA vs {len(df)} CSV entries"

cosine_pva = []

print(f"Computing PVA cosine similarities for {len(pva_files)} embeddings...")
for idx in tqdm(range(len(pva_files)), desc="Computing PVA sims"):
    pva_file = pva_files[idx]
    file_id = int(os.path.splitext(pva_file)[0])

    gt_path = os.path.join(embedding_dir_test, f"{file_id}.npy")
    if not os.path.exists(gt_path):
        cosine_pva.append(np.nan)
        continue

    gt_pva = torch.from_numpy(np.load(gt_path)).unsqueeze(0)
    emb_pva = torch.from_numpy(np.load(os.path.join(embedding_dir_pva, pva_file))).unsqueeze(0)

    gt_pva = torch.nn.functional.normalize(gt_pva, dim=1)
    emb_pva = torch.nn.functional.normalize(emb_pva, dim=1)

    sim_pva = torch.sum(gt_pva * emb_pva).item()
    cosine_pva.append(sim_pva)

cosine_pva = np.array(cosine_pva)
delta = cosine_cn - cosine_pva

# -------- SAVE UPDATED COMPARISON --------
out_csv = os.path.join(SAVE_DIR, f"similarity_scores_dual_{SAVE_PREFIX}.csv")
df_new = pd.DataFrame({
    "id": df["id"],
    "cosine_pva_test": cosine_pva,
    "cosine_controlnet": cosine_cn,
    "delta": delta
})
df_new.to_csv(out_csv, index=False)
print(f"\nSaved combined results to: {out_csv}")

# -------- PLOTS --------
plt.figure(figsize=(8,5))
plt.hist(cosine_pva, bins=50, alpha=0.6, label="PVA")
plt.hist(cosine_cn, bins=50, alpha=0.6, label="ControlNet")
plt.xlabel("Cosine Similarity", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
plt.title(f"Cosine Similarity Distribution", fontsize=19)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, f"similarity_hist_{SAVE_PREFIX}.png"), dpi=300)
plt.close()
