#!/bin/bash

### -- set the job Name --
#BSUB -J inference     # Updated job name for minimal test

### -- specify queue --
#BSUB -q gpua100

### -- set walltime limit: hh:mm --
#BSUB -W 00:05                   # Reduced walltime to 20 minutes for minimal test

# request GB of system-memory per core
#BSUB -R "rusage[mem=8GB]"     # Keeping memory request (adjust if needed)

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- ask for number of cores (default: 1) --
#BSUB -n 4

### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"

### -- Specify the output and error file. %J is the job-id --
#BSUB -o bash/bash_outputs/inference%J.out # Ensure 'bash/bash_outputs' exists
#BSUB -e bash/bash_outputs/inference%J.err  # Ensure 'bash/bash_outputs' exists

source /dtu/blackhole/17/209207/miniconda3/etc/profile.d/conda.sh


### -- Need to activate the python environment --
conda activate brain

### -- Set temporary cache directory in scratch space --
CACHE_DIR="/dtu/blackhole/17/209207/$LSB_JOBID/cache"
mkdir -p "$CACHE_DIR/huggingface"
export HF_HOME="$CACHE_DIR/huggingface"
echo "Hugging Face cache set to: $HF_HOME"

python /dtu/blackhole/17/209207/text-to-image-generation-in-the-medical-domain/src/inference.py