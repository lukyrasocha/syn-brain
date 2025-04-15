#!/bin/bash

### -- set the job Name --
#BSUB -J sd_finetuning     

### -- specify queue --
#BSUB -q gpua100

### -- set walltime limit: hh:mm --
#BSUB -W 00:10                 

# request GB of system-memory per core
#BSUB -R "rusage[mem=15GB]"    

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- ask for number of cores (default: 1) --
#BSUB -n 4

### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"

### -- Specify the output and error file. %J is the job-id --
#BSUB -o bash/bash_outputs/sd_finetuning%J.out 
#BSUB -e bash/bash_outputs/sd_finetuning%J.err 

source ~/.bashrc
conda activate brain




# Get the directory of the current script (train_sd.sh)
SCRIPT_DIR=$(dirname "$0")

# Set the cache directory based on the script location
CACHE_DIR="$SCRIPT_DIR/cache"

# Create the cache directory if it doesn't exist
mkdir -p "$CACHE_DIR/huggingface"
export HF_HOME="$CACHE_DIR/huggingface"
echo "Hugging Face cache set to: $HF_HOME"

# Set PyTorch CUDA allocation config (optional)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set WANDB cache/config directory (optional)
WANDB_CACHE_DIR="$CACHE_DIR/wandb"
mkdir -p "$WANDB_CACHE_DIR"
export WANDB_DIR="$WANDB_CACHE_DIR"  # Tells wandb where to write files
export WANDB_CONFIG_DIR="$WANDB_CACHE_DIR"  # Tells wandb where to look for config
echo "W&B cache/config directory set to: $WANDB_DIR"

TARGET_STEPS=20000

# --pretrained_vae_model_name_or_path="madebyollin/ sdxl-vae-fp16-fix" \
accelerate launch --num_processes=1 --mixed_precision="bf16" src/train_lora_sd.py \
--pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
--train_data_dir="/data/raw/Train_All_Images" \
--output_dir="models/sd" \
--resolution=512 \
--train_batch_size=2 \
--gradient_accumulation_steps=8 \
--max_train_steps=$TARGET_STEPS \
--learning_rate=0.0001 \
--rank=64 \
--gradient_checkpointing \
--max_grad_norm=1.0 \
--lr_scheduler="cosine" \
--mixed_precision="bf16" \
--report_to="wandb" \
--validation_prompt="The brain MRI shows a tumor in the left frontal lobe. The tumor appears to be medium, with a size of 2.5 x 2.4 x 2.25 cm. It has a heterogeneous shape and intensity, which means that the tumor has an irregular or mixed appearance in terms of its shape and the signal intensity on the MRI file_name." \
--validation_epochs=10 \
--snr_gamma=5.0 \
--adam_weight_decay=0.01 \
--lr_warmup_steps=1000 \
--checkpointing_steps=1000 \
--use_8bit_adam \
--seed=42 \
--num_validation_images=10 \
--dataloader_num_workers=8



