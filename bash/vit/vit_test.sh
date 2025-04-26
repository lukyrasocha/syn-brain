#!/bin/bash
###############################################################################
#                         Job configuration (LSF)                             #
###############################################################################
#BSUB -J vit_training_test
#BSUB -q gpuv100
#BSUB -W 00:30
#BSUB -n 4
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o bash/bash_outputs/vit_training_test.%J.out
#BSUB -e bash/bash_outputs/vit_training_test.%J.err
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
REQ_FILE="$PROJECT_DIR/requirements_vit.txt"

ACCEL_DIR="${ACCELERATE_CONFIG_DIR:-$BASE_DIR/huggingface_cache/accelerate}"
ACCEL_CFG="$ACCEL_DIR/default_config.yaml"
TRAIN_TEXT_ENCODER="${TRAIN_TEXT_ENCODER:-true}" 

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
#                           Training parameters                               #
###############################################################################
MODEL_DIR='models/vit/model.pth'
TRAIN_DATA_DIR='data/raw/Train_All_Images'
BATCH_SIZE=16
MIN_TUMOR_RATIO=0.25
IMAGE_SIZE=512
PATCH_SIZE=16
EMBED_DIM=128
NUM_HEADS=4
NUM_LAYERS=4
DROPOUT=0.3
EPOCHS=20
LEARNING_RATE=1e-4
WARMUP_STEPS=625
WEIGHT_DECAY=1e-3
FC_DIM=512

###############################################################################
#                               Launch                                        #
###############################################################################
echo ">>> Launching training …"
python src/vit/vit_training.py \
    --train_data_dir "$TRAIN_DATA_DIR"  \
    --split_file 'data/vit_training/validation.json' \
    --batch_size $BATCH_SIZE \
    --min_tumor_ratio $MIN_TUMOR_RATIO \
    --image_size $IMAGE_SIZE \
    --patch_size $PATCH_SIZE \
    --channels 3 \
    --embed_dim $EMBED_DIM \
    --num_heads $NUM_HEADS \
    --num_layers $NUM_LAYERS \
    --num_classes 2 \
    --pos_enc 'learnable' \
    --pool 'cls' \
    --dropout $DROPOUT \
    --fc_dim $FC_DIM \
    --num_epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --warmup_steps $WARMUP_STEPS \
    --weight_decay $WEIGHT_DECAY \
    --gradient_clipping 1.0 \
    --seed 1 \
    --model_dir "$MODEL_DIR"

###############################################################################
#                                 Cleanup                                     #
###############################################################################
rm -rf "$CACHE_DIR"
echo "==========  Job finished at $(date) =========="
exit $EXIT