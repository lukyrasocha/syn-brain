#!/bin/bash

### -- set the job Name --
#BSUB -J sdxl_finetuning     

### -- specify queue --
#BSUB -q gpua100

### -- set walltime limit: hh:mm --
#BSUB -W 24:00                 

# request GB of system-memory per core
#BSUB -R "rusage[mem=32GB]"    

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- ask for number of cores (default: 1) --
#BSUB -n 4

### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"

### -- Specify the output and error file. %J is the job-id --
#BSUB -o bash/bash_outputs/sdxl_finetuning%J.out 
#BSUB -e bash/bash_outputs/sdxl_finetuning%J.err 

# Send email when job begins
#BSUB -B

# Send email when job ends
#BSUB -N

# Email address to send notification to
#BSUB -u s240466@student.dtu.dk


source /dtu/blackhole/17/209207/miniconda3/etc/profile.d/conda.sh


conda activate brain




### -- Set temporary cache directory in scratch space --
CACHE_DIR="/dtu/blackhole/17/209207/$LSB_JOBID/cache"
mkdir -p "$CACHE_DIR/huggingface"
export HF_HOME="$CACHE_DIR/huggingface"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set WANDB cache/config directory (Optional but good practice in scratch)
WANDB_CACHE_DIR="$CACHE_DIR/wandb"
mkdir -p "$WANDB_CACHE_DIR"
export WANDB_DIR="$WANDB_CACHE_DIR" # Tells wandb where to write files
export WANDB_CONFIG_DIR="$WANDB_CACHE_DIR" # Tells wandb where to look for config


TARGET_STEPS=20000

accelerate launch --num_processes=1 --mixed_precision="bf16" \
/dtu/blackhole/17/209207/text-to-image-generation-in-the-medical-domain/diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
--pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
--train_data_dir="/dtu/blackhole/17/209207/text-to-image-generation-in-the-medical-domain/data/raw/Train_All_Images" \
--output_dir="/dtu/blackhole/17/209207/brain_big" \
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
--train_text_encoder \
--checkpointing_steps=1000 \
--use_8bit_adam \
--seed=42 \
--num_validation_images=10 \
--dataloader_num_workers=8



