#!/bin/bash
###############################################################################
#                         Job configuration (LSF)                             #
###############################################################################
#BSUB -J train_ovis_text_encoder_rank_128_text_encoder_false
#BSUB -q gpua100
#BSUB -W 24:00
#BSUB -n 4
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o bash/bash_outputs/train_ovis_text_encoder_rank_128_text_encoder_false.%J.out
#BSUB -e bash/bash_outputs/train_ovis_text_encoder_rank_128_text_encoder_false.%J.err
#BSUB -B
#BSUB -N

set -euo pipefail
echo "==========  Job started on $(hostname) at $(date) =========="

###############################################################################
#                         Paths & user options                                #
###############################################################################
PROJECT_DIR="$PWD"
BASE_DIR="${BASE_DIR:-$(dirname "$PWD")}"
CONDA_ENV="last_env"
PY_VER=3.11
REQ_FILE="$PROJECT_DIR/requirements_training_env.txt"

ACCEL_DIR="${ACCELERATE_CONFIG_DIR:-$BASE_DIR/huggingface_cache/accelerate}"
ACCEL_CFG="$ACCEL_DIR/default_config.yaml"
TRAIN_TEXT_ENCODER="${TRAIN_TEXT_ENCODER:-false}" 

###############################################################################
#                       Conda: create & activate                              #
###############################################################################
if [ -f "$BASE_DIR/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$BASE_DIR/miniconda3/etc/profile.d/conda.sh"
elif command -v conda &>/dev/null; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  module load conda/miniconda3 2>/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

conda env remove -n "$CONDA_ENV" -y || true
conda create -n "$CONDA_ENV" python=$PY_VER -y
conda activate "$CONDA_ENV"

###############################################################################
#                   Install full Python stack (GPU wheels)                    #
###############################################################################
pip install --no-cache-dir -r "$REQ_FILE" \
  --extra-index-url https://download.pytorch.org/whl/cu124

###############################################################################
#                       Write Accelerate default_config                       #
###############################################################################
mkdir -p "$ACCEL_DIR"
cat > "$ACCEL_CFG" <<'YAML'
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
mixed_precision: 'no'
num_processes: 1
num_machines: 1
same_network: true
machine_rank: 0
use_cpu: false
YAML
echo ">>> Accelerate config written to $ACCEL_CFG"

###############################################################################
#                        Hugging Face / W&B caches                            #
###############################################################################
CACHE_DIR="$PROJECT_DIR/cache/${LSB_JOBID}"
mkdir -p "$CACHE_DIR"/{huggingface,wandb}
export HF_HOME="$CACHE_DIR/huggingface"
export WANDB_DIR="$CACHE_DIR/wandb"
export WANDB_CONFIG_DIR="$CACHE_DIR/wandb"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

###############################################################################
#                           Training parameters                               #
###############################################################################
PRETRAINED_MODEL="stabilityai/stable-diffusion-xl-base-1.0"
PRETRAINED_VAE="madebyollin/sdxl-vae-fp16-fix"
RANK=128

RESOLUTION=512
BATCH_SIZE=2
ACCUM_STEPS=8
MAX_STEPS=20000
LR=1e-4
LR_WARMUP=1000
CHECK_STEPS=500

OUTPUT_DIR="$PROJECT_DIR/models/ovis_${LSB_JOBID}_${RANK}_gpua100_$( [ $TRAIN_TEXT_ENCODER = true ] && echo text_encoder || echo unet_only )"
mkdir -p "$OUTPUT_DIR"

VALID_PROMPT="Tumor: yes; location: left hemisphere; size: large; shape: irregular; intensity: hyperintense; orientation: axial; general description: brain MRI shows a hyperintense glioma in the left hemisphere, with surrounding edema and midline shift. No other abnormalities are visible." \

###############################################################################
#                               Launch                                        #
###############################################################################
echo ">>> Launching training …"
accelerate launch --config_file "$ACCEL_CFG" src/train_lora_sdxl.py \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
  --pretrained_vae_model_name_or_path "$PRETRAINED_VAE" \
  --train_data_dir "$PROJECT_DIR/data/raw/Train_All_Images" \
  --metadata_file "$PROJECT_DIR/data/preprocessed_json_files/metadata_ovis_large.jsonl" \
  --image_column image \
  --output_dir "$OUTPUT_DIR" \
  --resolution $RESOLUTION \
  --train_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $ACCUM_STEPS \
  --max_train_steps $MAX_STEPS \
  --learning_rate $LR \
  --rank $RANK \
  --gradient_checkpointing \
  --lr_scheduler cosine \
  --lr_warmup_steps $LR_WARMUP \
  --snr_gamma 5.0 \
  --adam_weight_decay 0.01 \
  --use_8bit_adam \
  --checkpointing_steps $CHECK_STEPS \
  --seed 42 \
  --validation_prompt "$VALID_PROMPT" \
  --validation_epochs 1 \
  --num_validation_images 10 \
  --dataloader_num_workers 4 \
  --report_to wandb \
  $( [ $TRAIN_TEXT_ENCODER = true ] && echo --train_text_encoder )

EXIT=$?
echo ">>> Training exit code: $EXIT"

###############################################################################
#                                 Cleanup                                     #
###############################################################################
rm -rf "$CACHE_DIR"
echo "==========  Job finished at $(date) =========="
exit $EXIT
