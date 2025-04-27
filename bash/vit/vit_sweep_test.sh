
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
#                           Hyper parameters                                  #
###############################################################################
embed_dims=(128 256 512)
num_heads=(8 16)
num_layers=(4 6)

###############################################################################
#                           Training parameters                               #
###############################################################################
TRAIN_DATA_DIR='data/raw/Train_All_Images'
BATCH_SIZE=16
MIN_TUMOR_RATIO=0.25
IMAGE_SIZE=512
PATCH_SIZE=16
DROPOUT=0.3
EPOCHS=40
LEARNING_RATE=1e-4
WARMUP_STEPS=625
WEIGHT_DECAY=1e-3
FC_DIM=512

# Loop over all combinations of hyperparameters
for embed_dim in "${embed_dims[@]}"; do
  for num_head in "${num_heads[@]}"; do
    for num_layer in "${num_layers[@]}"; do

      # Create a unique job name based on the hyperparameters
      JOB_NAME="vit_${embed_dim}_${num_head}_${num_layer}"
      
      # Create a unique model directory path
      MODEL_DIR="models/vit/${JOB_NAME}.pth"

      # Submit the job to the job scheduler
      bsub -J "$JOB_NAME" \
           -q gpuv100 \
           -W 00:45 \
           -n 4 \
           -R "rusage[mem=32GB] span[hosts=1]" \
           -gpu "num=1:mode=exclusive_process" \
           -o "bash/bash_outputs/${JOB_NAME}.%J.out" \
           -e "bash/bash_outputs/${JOB_NAME}.%J.err" \
           python src/vit/vit_training.py \
           --train_data_dir "$TRAIN_DATA_DIR" \
           --split_file 'data/vit_training/validation.json' \
           --batch_size $BATCH_SIZE \
           --min_tumor_ratio $MIN_TUMOR_RATIO \
           --image_size $IMAGE_SIZE \
           --patch_size $PATCH_SIZE \
           --channels 3 \
           --embed_dim $embed_dim \
           --num_heads $num_head \
           --num_layers $num_layer \
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

    done
  done
done

