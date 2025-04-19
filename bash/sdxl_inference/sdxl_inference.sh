#!/bin/bash

### ————————————————————————————————————————————————————————————— ###
###                       Job Configuration                       ###
### ————————————————————————————————————————————————————————————— ###
#BSUB -J sdxl_lora_inference              # job name
#BSUB -q gpua100                          # queue
#BSUB -W 02:00                            # walltime (hh:mm)
#BSUB -n 4                                # CPU cores
#BSUB -R "rusage[mem=32GB] span[hosts=1]" # memory per core
#BSUB -gpu "num=1:mode=exclusive_process" # request 1 GPU, exclusive
#BSUB -o bash/bash_outputs/infer_%J.out   # stdout file
#BSUB -e bash/bash_outputs/infer_%J.err   # stderr file
#BSUB -B                                  # email at start
#BSUB -N                                  # email at end
#BSUB -u s240466@student.dtu.dk           # your email

### ————————————————————————————————————————————————————————————— ###
###                  Environment / Cache Setup                     ###
### ————————————————————————————————————————————————————————————— ###
source /dtu/blackhole/17/209207/miniconda3/etc/profile.d/conda.sh
conda activate brain

# Set up a clean Hugging‑Face cache
CACHE_DIR="/dtu/blackhole/17/209207/${LSB_JOBID}/cache"
mkdir -p "$CACHE_DIR/huggingface"
export HF_HOME="$CACHE_DIR/huggingface"
echo "HF cache: $HF_HOME"

# Make CUDA allocator more flexible
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

### ————————————————————————————————————————————————————————————— ###
###                      Inference Parameters                       ###
### ————————————————————————————————————————————————————————————— ###
# Where your LoRA checkpoints live:
LORA_ROOT="/dtu/blackhole/17/209207/brain_big_new_2_2"
# Where to write out the generated images
OUT_ROOT="/dtu/blackhole/17/209207/brain_big_new_2_2/figures_inference"
mkdir -p "$OUT_ROOT"

### ————————————————————————————————————————————————————————————— ###
###                           Run Inference                         ###
### ————————————————————————————————————————————————————————————— ###
echo "Starting inference job $LSB_JOBID on $(hostname) at $(date)"

python src/testing_inference.py \
    --lora_root "$LORA_ROOT" \
    --out_root  "$OUT_ROOT" \
    --device    "cuda" \
    --seed      42

echo "Inference finished at $(date). Outputs in $OUT_ROOT"
