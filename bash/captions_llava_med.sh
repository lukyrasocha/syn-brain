#!/bin/bash

#BSUB -J image_captioner_llava_med 
#BSUB -q gpuv100
#BSUB -W 12:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
##BSUB -u s233498@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -R "span[hosts=1]"
#BSUB -o bash/bash_outputs/image_captioner_llava_med%J.out
#BSUB -e bash/bash_outputs/image_captioner_llava_med%J.err

source ~/.bashrc
conda activate llava-med

python src/image_captioner_llava_med.py --all 