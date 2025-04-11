#!/bin/bash

### -- set the job Name --
#BSUB -J sdxl_finetuning     

### -- specify queue --
#BSUB -q gpua100

### -- set walltime limit: hh:mm --
#BSUB -W 0:20                  

# request GB of system-memory per core
#BSUB -R "rusage[mem=16GB]"    

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


### -- Need to activate the python environment --
conda activate brain

## -- Load the environment variables from the .env file --
#set -a
#source /dtu/blackhole/17/209207/text-to-image-generation-in-the-medical-domain/.env
#set +a

# ADD HUGGING FACE TOKEN 
HUGGING_FACE_HUB_TOKEN=""

#WANDB TOKEN 
WANDB_API_KEY=""
WANDB_PROJECT=""


### -- Set temporary cache directory in scratch space --
CACHE_DIR="/dtu/blackhole/17/209207/$LSB_JOBID/cache"
mkdir -p "$CACHE_DIR/huggingface"
export HF_HOME="$CACHE_DIR/huggingface"
echo "Hugging Face cache set to: $HF_HOME"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set WANDB cache/config directory (Optional but good practice in scratch)
WANDB_CACHE_DIR="$CACHE_DIR/wandb"
mkdir -p "$WANDB_CACHE_DIR"
export WANDB_DIR="$WANDB_CACHE_DIR" # Tells wandb where to write files
export WANDB_CONFIG_DIR="$WANDB_CACHE_DIR" # Tells wandb where to look for config
echo "W&B cache/config directory set to: $WANDB_DIR"

python /dtu/blackhole/17/209207/text-to-image-generation-in-the-medical-domain/diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
  --dataset_name="lambdalabs/naruto-blip-captions" \
  --resolution=1024 --center_crop --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --rank=16 \
  --max_train_steps=5 \
  --learning_rate=1e-4 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir="/dtu/blackhole/17/209207/sdxl-pokemon-minimal-test-output" \
  --mixed_precision="no" \
  --report_to="wandb" \
  --validation_prompt="A photo of Pikachu pokemon" \
  --checkpointing_steps=10 \
  --use_8bit_adam \
  --seed=42 \
  --enable_xformers_memory_efficient_attention \
  --dataloader_num_workers=4