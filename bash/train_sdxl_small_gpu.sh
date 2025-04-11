#!/bin/bash

### -- set the job Name --
#BSUB -J sdxl_minimal_test 

### -- specify queue --
#BSUB -q gpuv100

### -- set walltime limit: hh:mm --
#BSUB -W 0:20                  

# request GB of system-memory per core
#BSUB -R "rusage[mem=16GB]"  

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- ask for number of cores (default: 1) --
#BSUB -n 4

### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"

### -- Specify the output and error file. %J is the job-id --
#BSUB -o bash/bash_outputs/sdxl_minimal_test_%J.out 
#BSUB -e bash/bash_outputs/sdxl_minimal_test_%J.err  

source /dtu/blackhole/17/209207/miniconda3/etc/profile.d/conda.sh


### -- Need to activate the python environment --
conda activate brain

## -- Load the environment variables from the .env file --
set -a
source /dtu/blackhole/17/209207/text-to-image-generation-in-the-medical-domain/.env
set +a


### -- Set temporary cache directory in scratch space --
CACHE_DIR="/dtu/blackhole/17/209207/$LSB_JOBID/cache"
mkdir -p "$CACHE_DIR/huggingface"
export HF_HOME="$CACHE_DIR/huggingface"
echo "Hugging Face cache set to: $HF_HOME"

python /dtu/blackhole/17/209207/text-to-image-generation-in-the-medical-domain/diffusers/examples/text_to_image/train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
  --dataset_name="lambdalabs/pokemon-blip-captions" \
  --resolution=1024 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=8 \
  --gradient_checkpointing \
  --max_train_steps=5 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="/dtu/blackhole/17/209207/sdxl-pokemon-minimal-test-output" \
  --mixed_precision="fp16" \
  --report_to="tensorboard" \
  --validation_prompt="A photo of Pikachu pokemon" \
  --checkpointing_steps=10 \
  --use_8bit_adam \
  --use_ema \
  --seed=42 \
  --enable_xformers_memory_efficient_attention



  