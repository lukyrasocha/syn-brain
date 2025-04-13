#!/bin/bash

#BSUB -J image_captioner_gemma
#BSUB -q gpua100
#BSUB -W 24:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -o bash/bash_outputs/image_captioner_gemma%J.out
#BSUB -e bash/bash_outputs/image_captioner_gemma%J.err
#BSUB -u s243867@student.dtu.dk
#BSUB -B
#BSUB -N

source ~/.bashrc
conda activate adlcv

python src/image_captioner_gemma.py \
    --image_folder "data/raw/Train_All_Images" \
    --max_new_tokens 77 \
    --hf_token "hf_token" \
    --model_id "google/gemma-3-12b-it" \
    --tumor \
    --prompt "Analyze brain MRI. Output format: tumor (yes/no); if yes—location (brain region), size (small/medium/large), shape, intensity. Also describe brain features, MRI orientation (axial/sagittal/coronal), and any other abnormalities. Max 77 tokens."
