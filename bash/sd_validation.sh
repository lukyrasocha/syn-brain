#!/bin/bash

#BSUB -J pretrained
#BSUB -q gpuv100
#BSUB -W 00:05
#BSUB -R "rusage[mem=2GB]"
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -o bash/bash_outputs/pretrained_%J.out
#BSUB -e bash/bash_outputs/pretrained_%J.err

# Properly initialize conda in the script
source activate base
conda activate brain

export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"

# The following shows how to use the accelerate denpending on the hardware  https://github.com/huggingface/accelerate/tree/main/examples

python test_pretrained.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --output_dir="sd-naruto-model"

    

