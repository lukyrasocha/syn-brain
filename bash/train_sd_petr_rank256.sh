#!/bin/bash

#BSUB -J sd_finetuning     
#BSUB -q gpuv100
#BSUB -W 00:10                 
#BSUB -R "rusage[mem=32GB]"    
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -o bash/bash_outputs/sd_finetuning%J.out 
#BSUB -e bash/bash_outputs/sd_finetuning%J.err 

source ~/.bashrc
conda activate brain


SCRIPT_DIR=$(dirname "$0")                                  # Get the directory of the current script (train_sd.sh)
CACHE_DIR="$SCRIPT_DIR/cache"                               # Set the cache directory based on the script location
mkdir -p "$CACHE_DIR/huggingface"                           # Create the cache directory if it doesn't exist
export HF_HOME="$CACHE_DIR/huggingface"
echo "Hugging Face cache set to: $HF_HOME"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     # Set PyTorch CUDA allocation config (optional)

WANDB_CACHE_DIR="$CACHE_DIR/wandb"                          # Set WANDB cache/config directory (optional)
mkdir -p "$WANDB_CACHE_DIR"
export WANDB_DIR="$WANDB_CACHE_DIR"                         # Tells wandb where to write files
export WANDB_CONFIG_DIR="$WANDB_CACHE_DIR"                  # Tells wandb where to look for config
echo "W&B cache/config directory set to: $WANDB_DIR"

TARGET_STEPS=20000

# --pretrained_vae_model_name_or_path="madebyollin/ sdxl-vae-fp16-fix" \
accelerate launch --num_processes=1 --mixed_precision="bf16" src/train_lora_sd.py \
--pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
--train_data_dir="data/raw/Train_All_Images" \
--metadata_file="data/preprocessed_json_files/metadata_llava_med.jsonl" \
--output_dir="models/sd" \
--resolution=512 \
--train_batch_size=2 \
--gradient_accumulation_steps=8 \
--max_train_steps=$TARGET_STEPS \
--learning_rate=0.0001 \
--rank=256 \
--gradient_checkpointing \
--max_grad_norm=1.0 \
--lr_scheduler="cosine" \
--mixed_precision="bf16" \
--report_to="wandb" \
--validation_prompt="The brain MRI shows a tumor in the left frontal lobe. The tumor appears to be medium, with a size of 2.5 x 2.4 x 2.25 cm. It has a heterogeneous shape and intensity, which means that the tumor has an irregular or mixed appearance in terms of its shape and the signal intensity on the MRI file_name." \
--validation_epochs=1 \
--snr_gamma=5.0 \
--adam_weight_decay=0.01 \
--lr_warmup_steps=1000 \
--checkpointing_steps=1000 \
--use_8bit_adam \
--seed=42 \
--num_validation_images=10 \
--dataloader_num_workers=4



