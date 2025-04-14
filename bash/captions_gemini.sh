#!/bin/bash

#BSUB -J image_captioner_gemini
#BSUB -q gpuv100
#BSUB -W 12:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -o bash/bash_outputs/image_captioner_gemini%J.out
#BSUB -e bash/bash_outputs/image_captioner_gemini%J.err

source ~/.bashrc
conda activate brain

python src/image_captioner_gemini.py