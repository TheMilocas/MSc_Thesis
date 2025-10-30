ID-ControlNet: Recovering Occluded Facial Features Using Identity-Aware Generative Models
Overview

ID-ControlNet is a framework for masked face image completion that preserves the identity of the target person while producing visually realistic outputs. Given a masked face and a frozen identity embedding (from a pretrained face recognition model), the system reconstructs the occluded regions in a way that faithfully matches the original person’s identity.

Unlike methods that require per-subject fine-tuning, ID-ControlNet leverages a pretrained latent diffusion model and trains only a small control branch.

How It Works

- The input face embedding is projected into a spatial representation and injected into a frozen diffusion model.

- The control branch gradually guides the model to reconstruct identity-specific details without modifying the pretrained network.

- Training uses a combination of image reconstruction, identity alignment, and cycle consistency to ensure outputs are both realistic and identity-preserving.

- During inference, you provide a masked face and the corresponding embedding, and the system reconstructs the missing facial content.

Command for training script:

CUDA_VISIBLE_DEVICES=1 accelerate launch ControlNet_plus_plus/train/my_reward_control.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --dataset_name="Milocas/celebahq_single_mask" \
  --image_column="image" \
  --conditioning_image_column="condition" \
  --caption_column="" \
  --task_name="identity" \
  --resolution=512 \
  --train_batch_size=4 \
  --num_train_epochs=100 \
  --grad_scale=0.5 \
  --validation_steps=500 \
  --checkpointing_steps=5000 \
  --output_dir="identity_controlnet_last" \
  --reward_model_name_or_path="ARCFACE/models/R100_MS1MV3/backbone.pth"
