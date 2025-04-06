
import torch
from diffusers import DiffusionPipeline,  DDPMScheduler
from peft import LoraConfig
from utils import check_parameter_dtypes, check_learnable_parameters, cast_training_params, freez_learnable_parameters
from transformers import CLIPTokenizerFast

def stable_diffusion_model(device):
    pipe = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", # model name
                torch_dtype=torch.float16,                  # load as 16-bit floating point for faster inference
                use_safetensors=True,                       # a safe and fast way of loading the model
                variant="fp16"                              # stands for "floating point 16-bit"
            ).to(device)
    freez_learnable_parameters(pipe)    # Freez all the parameters    
    unet_lora_config = LoraConfig(
        r=1,
        lora_alpha=3,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )

    pipe.unet.add_adapter(unet_lora_config)
    cast_training_params(pipe.unet, dtype=torch.float32)

    noise_scheduler = DDPMScheduler.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler"
    )

    return pipe, noise_scheduler

    

if __name__ == '__main__':
    model, noise_scheduler = stable_diffusion_model('cuda')
    print(type(model.tokenizer))      # should be transformers.CLIPTokenizer
    print(type(model.tokenizer_2))    # should be transformers.CLIPTokenizerFast
    print(model)
    check_learnable_parameters(model)
    check_parameter_dtypes(model)

