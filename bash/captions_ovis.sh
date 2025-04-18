#!/bin/bash

#BSUB -J image_captioner_ovis
#BSUB -q gpua100
#BSUB -W 24:00
#BSUB -R "rusage[mem=10GB]"
#BSUB -R "select[gpu80gb]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
##BSUB -u s233498@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -R "span[hosts=1]"
#BSUB -o bash/bash_outputs/image_captioner_ovis%J.out
#BSUB -e bash/bash_outputs/image_captioner_ovis%J.err

source ~/.bashrc
conda activate brain3

python src/image_captioner_ovis.py --all