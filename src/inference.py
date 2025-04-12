from diffusers import DiffusionPipeline
import torch

model_path = "/dtu/blackhole/17/209207/sdxl-pokemon-minimal-test-output" 

pipe = DiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe = pipe.to("cuda")

prompt = "Corgi dog in a field of flowers, cinematic lighting, pokemon style" 

try:
    image = pipe(prompt=prompt).images[0]
    
    import os
    os.makedirs("figures", exist_ok=True)
    
    image.save("figures/finetuned_pokemon_test.png")
    print(f"Image successfully generated and saved to figures/finetuned_pokemon_test.png")
except Exception as e:
    print(f"Error generating image: {e}")
    
    # Additional debug info
    print(f"\nModel path exists: {os.path.exists(model_path)}")
    if os.path.exists(model_path):
        print(f"Files in model directory: {os.listdir(model_path)}")