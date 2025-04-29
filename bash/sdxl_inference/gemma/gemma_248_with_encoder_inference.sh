#!/bin/bash

# Usage:
# bsub < run_inference.sh MODEL_NAME
#
# Example:
# bsub < run_inference.sh ovis_24746239_248_gpua100_text_encoder_fixed

### ————————————————————————————————————————————————————————————— ###
###                       Job Configuration                       ###
### ————————————————————————————————————————————————————————————— ###
#BSUB -J sdxl_lora_inference              # job name
#BSUB -q gpuv100                          # queue
#BSUB -W 04:00                            # walltime (hh:mm)
#BSUB -n 4                                # CPU cores
#BSUB -R "rusage[mem=32GB] span[hosts=1]" # memory per core
#BSUB -gpu "num=1:mode=exclusive_process" # request 1 GPU, exclusive
#BSUB -o bash/bash_outputs/infer_%J.out   # stdout file
#BSUB -e bash/bash_outputs/infer_%J.err   # stderr file
#BSUB -B                                  # email at start
#BSUB -N                                  # email at end
#BSUB -u s240466@student.dtu.dk           # your email

set -euo pipefail
echo "==========  Job started on $(hostname) at $(date) =========="

### ————————————————————————————————————————————————————————————— ###
###                  Environment / Cache Setup                     ###
### ————————————————————————————————————————————————————————————— ###
# Directory structure following the training script
PROJECT_DIR="$PWD"
BASE_DIR="${BASE_DIR:-$(dirname "$PWD")}"

source "$BASE_DIR/miniconda3/etc/profile.d/conda.sh"
conda activate last_env

# Set up a clean Hugging‑Face cache
CACHE_DIR="$PROJECT_DIR/cache/${LSB_JOBID}"
mkdir -p "$CACHE_DIR/huggingface"
export HF_HOME="$CACHE_DIR/huggingface"
echo "HF cache: $HF_HOME"

# Make CUDA allocator more flexible
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

### ————————————————————————————————————————————————————————————— ###
###                      Inference Parameters                       ###
### ————————————————————————————————————————————————————————————— ###
# Model name/folder - can be passed as first argument to the script
MODEL_NAME="${1-gemma_24777024_248_gpua100_text_encoder}"  # REPLACE XXX with the model name

# Derived paths - following the structure from the training script
LORA_ROOT="$PROJECT_DIR/models/${MODEL_NAME}"
OUT_ROOT="$PROJECT_DIR/generated_images/${MODEL_NAME}"

# Create output directory
mkdir -p "$OUT_ROOT"

# Path to metadata JSON file containing prompts and filenames
METADATA_JSON="$PROJECT_DIR/data/preprocessed_json_files/metadata_gemma_better_test.jsonl"



# Inference parameters
BATCH_SIZE=4
NUM_STEPS=50
SEED=42

### ————————————————————————————————————————————————————————————— ###
###                           Run Inference                         ###
### ————————————————————————————————————————————————————————————— ###
echo "Starting inference for model: $MODEL_NAME"
echo "Job ID: $LSB_JOBID on $(hostname) at $(date)"
echo "Model path: $LORA_ROOT"
echo "Output path: $OUT_ROOT"

python src/testing_inference_pipeline.py \
    --lora_root "$LORA_ROOT" \
    --out_root "$OUT_ROOT" \
    --metadata_json "$METADATA_JSON" \
    --device "cuda" \
    --batch_size "$BATCH_SIZE" \
    --num_inference_steps "$NUM_STEPS" \
    --seed "$SEED"

EXIT=$?
echo ">>> Inference exit code: $EXIT"

### ————————————————————————————————————————————————————————————— ###
###                           Cleanup                               ###
### ————————————————————————————————————————————————————————————— ###
rm -rf "$CACHE_DIR"
echo "==========  Job finished at $(date) =========="
exit $EXIT