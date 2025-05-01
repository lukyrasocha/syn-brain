#!/bin/bash

#BSUB -J metrics
#BSUB -q gpuv100
#BSUB -W 00:10
#BSUB -R "rusage[mem=10GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -o bash/bash_outputs/metrics%J.out
#BSUB -e bash/bash_outputs/metrics%J.err
#BSUB -u s243867@student.dtu.dk
#BSUB -B
#BSUB -N

source /work3/s243867/miniconda3/etc/profile.d/conda.sh
conda activate adlcv
pip install git+https://github.com/openai/CLIP.git

#python src/metrics.py --extract_real
#python src/metrics.py --score_type "no_clip" --gen_folder "data/synthetic_bad"
python src/metrics.py   --score_type "all" \
                        --gen_folder "generated_images/llava_24816638_248_gpua100_text_encoder_fixed/checkpoint-10500" \
                        --captions_path "data/preprocessed_json_files/metadata_llava_med_test.jsonl"
