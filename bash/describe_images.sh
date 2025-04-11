#!/bin/bash

### -- set the job Name -- 
#BSUB -J train_test

### -- specify queue --
#BSUB -q gpuv100

### -- set walltime limit: hh:mm -- 
#BSUB -W 00:10

# request GB of system-memory per core
#BSUB -R "rusage[mem=8GB]"

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- ask for number of cores (default: 1) --
#BSUB -n 4

### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"

### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o train_test_%J.out
#BSUB -e train_test_%J.err

source ~/.bashrc
conda activate brain3

#python src/describe_image.py --prompt "Describe what you see in this MRI image. Is there a tumor? If yes, describe the tumor's size, location, intensity and shape. Also describe the image orientation (axial, sagittal, or coronal) and any important visual features of the MRI." --image data/raw/Train_All_Images/meningioma_0549.jpg
python src/describe_image.py --all 