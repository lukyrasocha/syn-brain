#!/bin/bash

### -- set the job Name --
#BSUB -J sdxl_lora_V100_24h

### -- specify queue --
#BSUB -q gpuv100

### -- set walltime limit: hh:mm --
#BSUB -W 24:00

# request GB of system-memory per core
#BSUB -R "rusage[mem=32GB]"

### -- Select the resources: 1 GPU in exclusive process mode on one node --
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"

### -- ask for number of cores --
#BSUB -n 4

### -- Specify the output and error file --
#BSUB -o bash/bash_outputs/sdxl_lora_V100_24h_%J.out
#BSUB -e bash/bash_outputs/sdxl_lora_V100_24h_%J.err

# Send email notifications at the start and end of the job
#BSUB -B
#BSUB -N
#BSUB -u s240466@student.dtu.dk

source /dtu/blackhole/17/209207/miniconda3/etc/profile.d/conda.sh
conda activate brain

# Set temporary cache directory in scratch space
CACHE_DIR="/dtu/blackhole/17/209207/$LSB_JOBID/cache"
mkdir -p "$CACHE_DIR/huggingface"
export HF_HOME="$CACHE_DIR/huggingface"
echo "Hugging Face cache set to: $HF_HOME"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set WANDB cache/config directory
WANDB_CACHE_DIR="$CACHE_DIR/wandb"
mkdir -p "$WANDB_CACHE_DIR"
export WANDB_DIR="$WANDB_CACHE_DIR"
export WANDB_CONFIG_DIR="$WANDB_CACHE_DIR"


TARGET_STEPS=9000

accelerate launch --num_processes=2 --mixed_precision="no" \
/dtu/blackhole/17/209207/text-to-image-generation-in-the-medical-domain/diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
--pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
--dataset_name="lambdalabs/naruto-blip-captions" \
--resolution=1024 \
--center_crop \
--random_flip \
--train_batch_size=4 \
--gradient_accumulation_steps=1 \
--max_train_steps=$TARGET_STEPS \
--learning_rate=1e-4 \
--rank=32 \
--gradient_checkpointing \
--max_grad_norm=1 \
--lr_scheduler="cosine" \
--lr_warmup_steps=0 \
--output_dir="/dtu/blackhole/17/209207/sdxl-naruto-lora-V100-output" \
--mixed_precision="no" \
--report_to="wandb" \
--validation_prompt="A ninja portrait of Naruto Uzumaki, facing camera, detailed illustration, anime style" \
--validation_epochs=10 \
--checkpointing_steps=500 \
--use_8bit_adam \
--seed=42 \
--enable_xformers_memory_efficient_attention \
--dataloader_num_workers=4
