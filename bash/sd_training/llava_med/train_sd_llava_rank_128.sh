#!/bin/bash
### ————————————————————————————————————————————————————————————— ###
###                       Job Configuration                       ###
### ————————————————————————————————————————————————————————————— ###
#BSUB -J train_sd_llava_rank_128                                  # job name
#BSUB -q gpua100                                                  # queue
#BSUB -W 24:00                                                    # walltime (hh:mm)
#BSUB -n 4                                                        # CPU cores
#BSUB -R "rusage[mem=64GB] span[hosts=1]"                         # memory and host
#BSUB -gpu "num=1:mode=exclusive_process"                         # memory and host
#BSUB -o bash/bash_outputs/train_sd_llava_rank_128.%J.out         # stdout
#BSUB -e bash/bash_outputs/train_sd_llava_rank_128.%J.err         # stdout
#BSUB -B                                                          # email at start
#BSUB -N                                                          # email at end
### ————————————————————————————————————————————————————————————— ###
###                   Environment / Cache Setup                   ###
### ————————————————————————————————————————————————————————————— ###


source ~/.bashrc
conda activate brain

# HF & W&B cache in scratch

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


### ————————————————————————————————————————————————————————————— ###
###                    Training Parameters                        ###
### ————————————————————————————————————————————————————————————— ###
# training
RESOLUTION=1024
BATCH_SIZE=2
ACCUM_STEPS=8
MAX_STEPS=20000  
LR=0.0001
RANK=128
SEED=42
VALID_EPOCHS=1
NUM_VAL_IMAGES=10
WORKERS=4


# model
PRETRAINED_MODEL="stable-diffusion-v1-5/stable-diffusion-v1-5"

# data
TRAIN_DATA_DIR="data/raw/Train_All_Images"
METADATA_FILE="data/preprocessed_json_files/metadata_gemini.jsonl"
OUTPUT_DIR="models/gemini/model_${LSB_JOBID}_${RANK}_gpua100"


VALID_PROMPT="Brain MRI shows a large, irregularly shaped, hyperintense glioma in the right temporal lobe, with surrounding edema and mass effect. No other abnormalities are evident." \
### ————————————————————————————————————————————————————————————— ###
###                     Launch with Accelerate                    ###
### ————————————————————————————————————————————————————————————— ###
accelerate launch \
  --num_processes=1 \
  --mixed_precision="fp16" \
  src/train_lora_sd.py \
    --pretrained_model_name_or_path="$PRETRAINED_MODEL" \
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
    --lr_warmup_steps=500 \
    --snr_gamma=5.0 \
    --gradient_checkpointing \
    --adam_weight_decay=0.01 \
    --use_8bit_adam \
    --checkpointing_steps=500 \
    --seed=$SEED \
    --validation_prompt="$VALID_PROMPT" \
    --validation_epochs=$VALID_EPOCHS \
    --num_validation_images=$NUM_VAL_IMAGES \
    --dataloader_num_workers=$WORKERS \
    --report_to="wandb" \