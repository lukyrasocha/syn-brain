#!/bin/bash

#BSUB -J image_captioner_ovis
#BSUB -q gpua100
#BSUB -W 12:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "select[gpu80gb]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -o bash/bash_outputs/image_captioner_ovis%J.out
#BSUB -e bash/bash_outputs/image_captioner_ovis%J.err

source ~/.bashrc
conda activate brain

#python src/describe_image.py --prompt "Describe what you see in this MRI image. Is there a tumor? If yes, describe the tumor's size, location, intensity and shape. Also describe the image orientation (axial, sagittal, or coronal) and any important visual features of the MRI." --image data/raw/Train_All_Images/meningioma_0549.jpg
python src/image_captioner_ovis.py --all 