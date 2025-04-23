#!/bin/bash
###############################################################################
#                         Job Configuration (LSF)                             #
###############################################################################
#BSUB -J train_ovis_text_encoder_rank_128
#BSUB -q gpuv100
#BSUB -W 01:00
#BSUB -n 4
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o bash/bash_outputs/train_sdxl_gemini_rank_128.%J.out
#BSUB -e bash/bash_outputs/train_sdxl_gemini_rank_128.%J.err
#BSUB -B
#BSUB -N
#BSUB -u s240466@student.dtu.dk

set -euo pipefail
echo "==========  Job started on $(hostname) at $(date)  =========="

###############################################################################
#                          Paths & user options                               
###############################################################################
PROJECT_DIR="$PWD"
BASE_DIR="${BASE_DIR:-$(dirname "$PWD")}"
echo "PROJECT_DIR: $PROJECT_DIR"
echo "BASE_DIR   : $BASE_DIR"

CONDA_ENV_NAME="brain_training_last_3"
PYTHON_VERSION=3.11
REQUIREMENTS_FILE="$PROJECT_DIR/requirements_training_env.txt"

ACCELERATE_CONFIG_DIR="${ACCELERATE_CONFIG_DIR:-$BASE_DIR/huggingface_cache/accelerate}"
ACCELERATE_CONFIG_FILE="$ACCELERATE_CONFIG_DIR/default_config.yaml"

TRAIN_TEXT_ENCODER="${TRAIN_TEXT_ENCODER:-true}"

###############################################################################
#                             Conda environment                               
###############################################################################
if [ -f "$BASE_DIR/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$BASE_DIR/miniconda3/etc/profile.d/conda.sh"
elif command -v conda &>/dev/null; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  module load conda/miniconda3 2>/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

conda env remove -n "$CONDA_ENV_NAME" -y || true
conda create -n "$CONDA_ENV_NAME" python=$PYTHON_VERSION -y
conda activate "$CONDA_ENV_NAME"

###############################################################################
#                    PyTorch 2.6 wheels (CUDA 12.4) via pip                   
###############################################################################
pip install --no-cache-dir \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --extra-index-url https://download.pytorch.org/whl/cu124

###############################################################################
#         Clean requirements: fix diffusers pin and yanked multidict          
###############################################################################
TMP_REQ=$(mktemp)
grep -vE '^diffusers==0\.33\.0\.dev0$' "$REQUIREMENTS_FILE" \
  | sed 's/^multidict==6\.3\.2$/multidict==6.0.5/' > "$TMP_REQ"

pip install --no-cache-dir -r "$TMP_REQ"
rm "$TMP_REQ"

# explicit healthy versions
pip install --no-cache-dir diffusers==0.33.1 multidict==6.0.5

###############################################################################
#                       Write Accelerate default_config                       
###############################################################################
mkdir -p "$ACCELERATE_CONFIG_DIR"
cat > "$ACCELERATE_CONFIG_FILE" <<'YAML'
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: 'NO'
downcast_bf16: 'no'
enable_cpu_affinity: false
ipex_config:
  ipex: false
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
YAML
echo ">>> Accelerate config written to $ACCELERATE_CONFIG_FILE"

###############################################################################
#                       Hugging Face / W&B cache paths                        
###############################################################################
CACHE_DIR="$PROJECT_DIR/cache/${LSB_JOBID}"
mkdir -p "$CACHE_DIR/huggingface" "$CACHE_DIR/wandb"
export HF_HOME="$CACHE_DIR/huggingface"
export WANDB_DIR="$CACHE_DIR/wandb"
export WANDB_CONFIG_DIR="$CACHE_DIR/wandb"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

###############################################################################
#                          Training parameters                                 
###############################################################################
PRETRAINED_MODEL="stabilityai/stable-diffusion-xl-base-1.0"
PRETRAINED_VAE="madebyollin/sdxl-vae-fp16-fix"
RANK=248

RESOLUTION=512
BATCH_SIZE=2
ACCUM_STEPS=8
MAX_STEPS=20000
LR=0.0001
LR_WARMUP_STEPS=1000
LR_SCHEDULER="cosine"
MAX_GRAD_NORM=1.0
ADAM_WEIGHT_DECAY=0.01
SNR_GAMMA=5.0

SEED=42
VALID_EPOCHS=1
NUM_VAL_IMAGES=10
CHECKPOINTING_STEPS=500
WORKERS=4
REPORT_TO="wandb"

TRAIN_DATA_DIR="$PROJECT_DIR/data/raw/Train_All_Images"
METADATA_FILE="$PROJECT_DIR/data/preprocessed_json_files/metadata_ovis_large.jsonl"
OUTPUT_DIR="$PROJECT_DIR/models/gemini_${LSB_JOBID}_${RANK}_gpua100_$( [ "$TRAIN_TEXT_ENCODER" = "true" ] && echo text_encoder || echo unet_only )"
VALID_PROMPT="tumor: yes; location: pituitary; size: large; shape: regular; intensity: bright; orientation: sagittal; general description: Brain MRI in sagittal view showing large pituitary tumor. Abnormal enhancement is seen involving the pituitary region and surrounding structures."
mkdir -p "$OUTPUT_DIR"

###############################################################################
#                                Training                                     
###############################################################################
echo "Launching training…"
ACCEL_CMD="accelerate launch --config_file $ACCELERATE_CONFIG_FILE \
  src/train_lora_sdxl.py \
  --pretrained_model_name_or_path=$PRETRAINED_MODEL \
  --pretrained_vae_model_name_or_path=$PRETRAINED_VAE \
  --train_data_dir=$TRAIN_DATA_DIR \
  --metadata_file=$METADATA_FILE \
  --image_column=image \
  --output_dir=$OUTPUT_DIR \
  --resolution=$RESOLUTION \
  --train_batch_size=$BATCH_SIZE \
  --gradient_accumulation_steps=$ACCUM_STEPS \
  --max_train_steps=$MAX_STEPS \
  --learning_rate=$LR \
  --rank=$RANK \
  --gradient_checkpointing \
  --max_grad_norm=$MAX_GRAD_NORM \
  --lr_scheduler=$LR_SCHEDULER \
  --lr_warmup_steps=$LR_WARMUP_STEPS \
  --snr_gamma=$SNR_GAMMA \
  --adam_weight_decay=$ADAM_WEIGHT_DECAY \
  --use_8bit_adam \
  --checkpointing_steps=$CHECKPOINTING_STEPS \
  --seed=$SEED \
  --validation_prompt=\"$VALID_PROMPT\" \
  --validation_epochs=$VALID_EPOCHS \
  --num_validation_images=$NUM_VAL_IMAGES \
  --dataloader_num_workers=$WORKERS \
  --report_to=$REPORT_TO"
[ "$TRAIN_TEXT_ENCODER" = "true" ] && ACCEL_CMD="$ACCEL_CMD --train_text_encoder"

echo "$ACCEL_CMD"
eval $ACCEL_CMD
EXIT_CODE=$?

###############################################################################
#                                   Cleanup                                   
###############################################################################
echo "Training exit code: $EXIT_CODE"
rm -rf "$CACHE_DIR"
echo "==========  Job finished at $(date)  =========="
exit $EXIT_CODE
