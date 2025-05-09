#!/bin/bash
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0

GPU_IDS="0"

DATA_ROOT="single_frame"
# DATA_ROOT="final_dataset"
MODEL="genmo/mochi-1-preview"
OUTPUT_PATH="mochi-rgba-lora-f37"

cmd="CUDA_VISIBLE_DEVICES=$GPU_IDS python train.py \
  --pretrained_model_name_or_path $MODEL \
  --cast_dit \
  --data_root $DATA_ROOT \
  --seed 42 \
  --output_dir $OUTPUT_PATH \
  --train_batch_size 1 \
  --enable_model_cpu_offload \
  --dataloader_num_workers 100 \
  --pin_memory \
  --checkpointing_steps 100 \
  --caption_dropout 0.1 \
  --report_to wandb \
  --rank 16 \
  --lr_warmup_steps 20 \
  --max_train_steps 2000 \
  --gradient_checkpointing \
  --push_to_hub \
  --validation_steps 20 \
  --enable_slicing \
  --enable_tiling \
  --optimizer adamw \
  --allow_tf32 \
  --learning_rate 7e-4" 

echo "Running command: $cmd"
eval $cmd
echo -ne "-------------------- Finished executing script --------------------\n\n"

# --enable_model_cpu_offload \
