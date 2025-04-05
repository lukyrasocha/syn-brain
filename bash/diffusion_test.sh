#!/bin/bash

### -- set the job Name -- 
#BSUB -J diffusion_test

### -- specify queue --
#BSUB -q gpuv100

### -- set walltime limit: hh:mm -- 
#BSUB -W 10

# request GB of system-memory per core
#BSUB -R "rusage[mem=10GB]"

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- ask for number of cores (default: 1) --
#BSUB -n 4

### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"

### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o bash/bash_outputs/diffusion_test__%J.out
#BSUB -e bash/bash_outputs/diffusion_test__%J.err

### -- Need to activate the python environment --
conda activate brain

### -- run in the job --
python src/models.py
