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
