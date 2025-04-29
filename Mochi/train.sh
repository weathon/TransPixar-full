#!/bin/bash
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0

GPU_IDS="0"

DATA_ROOT="final_dataset"
MODEL="genmo/mochi-1-preview"
OUTPUT_PATH="mochi-rgba-lora-f37"

cmd="CUDA_VISIBLE_DEVICES=$GPU_IDS python train.py \
  --pretrained_model_name_or_path $MODEL \
  --cast_dit \
  --data_root $DATA_ROOT \
  --seed 42 \
  --output_dir $OUTPUT_PATH \
  --train_batch_size 2 \
  --dataloader_num_workers 20 \
  --pin_memory \
  --checkpointing_steps 100 \
  --report_to wandb \
  --rank 32 \
  --lr_warmup_steps 40 \
  --max_train_steps 6000 \
  --gradient_checkpointing \
  --push_to_hub \
  --validation_steps 200 \
  --enable_slicing \
  --enable_tiling \
  --optimizer adamw \
  --allow_tf32 \
  --weight_decay 0.05 \
  --width 576 \
  --height 320 \
  --num_validation_videos 2 \
  --learning_rate 7e-4"

  # --caption_dropout 0.1 \
  # --lora_alpha 1 \
  # --resume_from_checkpoint './mochi-rgba-lora-f37/checkpoint-3400.pt' \
  
echo "Running command: $cmd"
eval $cmd
echo -ne "-------------------- Finished executing script --------------------\n\n"

# --enable_model_cpu_offload \
