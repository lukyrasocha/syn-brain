#!/bin/bash

#BSUB -J sd_lora
#BSUB -q gpuv100
#BSUB -W 00:05
#BSUB -R "rusage[mem=10GB]"
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -o bash/bash_outputs/sd_lora_%J.out
#BSUB -e bash/bash_outputs/sd_lora_%J.err

# initialize conda in the script
source activate base
conda activate brain
pip install bitsandbytes

# The following shows how to use the accelerate denpending on the hardware  https://github.com/huggingface/accelerate/tree/main/examples
python src/train_text_to_image_lora.py \
    --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
    --dataset_name 'lambdalabs/naruto-blip-captions' \
    --output_dir "models/sd-naruto-model-lora" \
    --num_train_epochs 1 \
    --train_batch_size 1 \
    # --use_8bit_adam \
    # --enable_xformers_memory_efficient_attention
    # --mixed_precision="fp16" \
    # --variant 'fp16' \
    # --resolution 512 \
    # --learning_rate 1e-4 \
    # --lr_scheduler 'constant' \
    # --lr_warmup_steps 500 \
    # --dataloader_num_workers 4 \
    # --adam_beta1 0.9 \
    # --adam_beta2 0.999 \
    # --adam_weight_decay 1e-2 \
    # --adam_epsilon 1e-08 \
    # --max_grad_norm 1 \
    # --prediction_type 'epsilon' \ 
    # # --report_to wandb \