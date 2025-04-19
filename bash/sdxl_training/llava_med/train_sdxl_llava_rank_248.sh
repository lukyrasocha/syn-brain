#!/bin/bash
### ————————————————————————————————————————————————————————————— ###
###                       Job Configuration                       ###
### ————————————————————————————————————————————————————————————— ###
#BSUB -J train_sdxl_llava_rank_248                                     # job name
#BSUB -q gpua100                                                  # queue
#BSUB -W 24:00                                                    # walltime (hh:mm)
#BSUB -n 4                                                        # CPU cores
#BSUB -R "rusage[mem=32GB] span[hosts=1]"                         # memory and host
#BSUB -gpu "num=1:mode=exclusive_process"                         # memory and host
#BSUB -o bash/bash_outputs/train_sdxl_llava_rank_248.%J.out       # stdout
#BSUB -e bash/bash_outputs/train_sdxl_llava_rank_248.%J.err       # stdout
#BSUB -B                                                          # email at start
#BSUB -N                                                          # email at end
#BSUB -u s240466@student.dtu.dk                                   # your email
### ————————————————————————————————————————————————————————————— ###
###                   Environment / Cache Setup                      ###
### ————————————————————————————————————————————————————————————— ###

source /dtu/blackhole/17/209207/miniconda3/etc/profile.d/conda.sh
conda activate brain

# HF & W&B cache in scratch
CACHE_DIR="/dtu/blackhole/17/209207/$LSB_JOBID/cache"
mkdir -p "$CACHE_DIR/huggingface"
export HF_HOME="$CACHE_DIR/huggingface"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WANDB_CACHE_DIR="$CACHE_DIR/wandb"
mkdir -p "$WANDB_CACHE_DIR"
export WANDB_DIR="$WANDB_CACHE_DIR"
export WANDB_CONFIG_DIR="$WANDB_CACHE_DIR"

### ————————————————————————————————————————————————————————————— ###
###                    Training Parameters                        ###
### ————————————————————————————————————————————————————————————— ###
# model
PRETRAINED_MODEL="stabilityai/stable-diffusion-xl-base-1.0"
PRETRAINED_VAE="madebyollin/sdxl-vae-fp16-fix"

# data
TRAIN_DATA_DIR="data/raw/Train_All_Images"
METADATA_FILE="data/preprocessed_json_files/metadata_llava_med.jsonl"
OUTPUT_DIR="/dtu/blackhole/17/209207/llava/model_$LSB_JOBID"

# training
RESOLUTION=1024
BATCH_SIZE=2
ACCUM_STEPS=8
MAX_STEPS=20000  
LR=0.0001
RANK=248
SEED=42
VALID_EPOCHS=10
NUM_VAL_IMAGES=10
WORKERS=4

VALID_PROMPT="Brain MRI shows a large, irregularly shaped, hyperintense glioma in the right temporal lobe, with surrounding edema and mass effect. No other abnormalities are evident." \
### ————————————————————————————————————————————————————————————— ###
###                     Launch with Accelerate                    ###
### ————————————————————————————————————————————————————————————— ###
accelerate launch \
  --num_processes=1 \
  --mixed_precision="bf16" \
  src/train_lora_sdxl.py \
    --pretrained_model_name_or_path="$PRETRAINED_MODEL" \
    --pretrained_vae_model_name_or_path="$PRETRAINED_VAE" \
    --train_data_dir="$TRAIN_DATA_DIR" \
    --metadata_file="$METADATA_FILE" \
    --center_crop \
    --image_column="image" \
    --output_dir="$OUTPUT_DIR" \
    --resolution=$RESOLUTION \
    --train_batch_size=$BATCH_SIZE \
    --gradient_accumulation_steps=$ACCUM_STEPS \
    --max_train_steps=$MAX_STEPS \
    --learning_rate=$LR \
    --rank=$RANK \
    --gradient_checkpointing \
    --max_grad_norm=1.0 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=1000 \
    --snr_gamma=5.0 \
    --gradient_checkpointing \
    --adam_weight_decay=0.01 \
    --train_text_encoder \
    --use_8bit_adam \
    --checkpointing_steps=500 \
    --seed=$SEED \
    --validation_prompt="$VALID_PROMPT" \
    --validation_epochs=$VALID_EPOCHS \
    --num_validation_images=$NUM_VAL_IMAGES \
    --dataloader_num_workers=$WORKERS \
    --report_to="wandb" \