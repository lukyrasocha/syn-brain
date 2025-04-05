import torch
from diffusers import DiffusionPipeline,  UNet2DConditionModel
from matplotlib import pyplot as plt
from peft import get_peft_model, LoraConfig

def check_learnable_parameters(pipe):
    print('\nCHECK FOR LEARNABLE PARAMETERS\n')
    # List of component names to check
    components_to_check = [
        "text_encoder",
        "text_encoder_2",
        "tokenizer",
        "tokenizer_2",
        "unet",
        "vae",
        "image_encoder",
        "scheduler",
        "feature_extractor"
    ]

    print("Learnable parameters per component:\n")

    for name in components_to_check:
        component = getattr(pipe, name, None)
        if component is None:
            continue

        params = list(component.parameters()) if hasattr(component, "parameters") else []
        learnable = [p for p in params if p.requires_grad]

        print(f"{name}: {len(learnable)} learnable parameters")

        # Optional: print total count
        total_params = sum(p.numel() for p in learnable)
        if total_params > 0:
            print(f"    Total learnable parameters: {total_params:,}")


    print(" check if some parameters is set to true_grad")
    has_grad = False

    for name, module in pipe.components.items():
        if hasattr(module, "parameters"):
            for param in module.parameters():
                if param.requires_grad:
                    print(f"✅ `{name}` has parameters with `requires_grad=True`")
                    has_grad = True
                    break  # no need to keep checking this module

    if not has_grad:
        print("❌ No modules have trainable (requires_grad=True) parameters.")

def freez_learnable_parameters(pipe):
    print('\nFREEZ LEARNABLE PARAMETERS\n')
    # Freeze parameters for specific components
    components_to_freeze = ["vae", "text_encoder", "text_encoder_2", "unet"]

    for name in components_to_freeze:
        component = getattr(pipe, name, None)
        if component is not None:
            for param in component.parameters():
                param.requires_grad = False

    # Verify that parameters are frozen
    print("Learnable parameters per component:\n")
    for name in components_to_freeze:
        component = getattr(pipe, name, None)
        if component is None:
            continue

        params = list(component.parameters()) if hasattr(component, "parameters") else []
        learnable = [p for p in params if p.requires_grad]

        print(f"{name}: {len(learnable)} learnable parameters")



if __name__ == '__main__':
    # https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py

    pipe = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", # model name
                torch_dtype=torch.float16,                  # load as 16-bit floating point for faster inference
                use_safetensors=True,                       # a safe and fast way of loading the model
                variant="fp16"                              # stands for "floating point 16-bit"
            ).to("cuda")
    print(f'The pipeline for the model is:\n{pipe}\n')
    check_learnable_parameters(pipe)    # check which parameters is set to require_grad = true
    freez_learnable_parameters(pipe)    # Freez all the parameters    
    check_learnable_parameters(pipe)    # check that the function above worked    

    # Apply LoRA to UNet 
    unet_lora_config = LoraConfig(
        r=1,
        lora_alpha=3,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    pipe.unet.add_adapter(unet_lora_config)
    check_learnable_parameters(pipe)    # check that the function above worked    

    print('train')
    pipe.unet.train()
    check_learnable_parameters(pipe)

    print('eval')
    with torch.no_grad():
        pipe.unet.eval()
        check_learnable_parameters(pipe)
    # prompt = "A high-resolution black and white MRI scan of a human brain, axial view, showing detailed brain structures with realistic textures and contrast. Medical imaging style, realistic lighting, and high anatomical accuracy."
    # images = pipe(prompt=prompt).images[0]
    

    # plt.imshow(images)
    # plt.axis('off')  
    # plt.savefig('test_displayed.png', bbox_inches='tight', pad_inches=0)  #












    # pipe = AutoPipelineForText2Image.from_pretrained('runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16)
    # pipe.load_lora_weights('DoctorDiffusion/doctor-diffusion-s-xray-xl-lora', weight_name='DD-xray-v1.safetensors')

    # pipe = AutoPipelineForText2Image.from_pretrained('runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16)
    # pipe.load_lora_weights('DoctorDiffusion/doctor-diffusion-s-xray-xl-lora', weight_name='DD-xray-v1.safetensors')