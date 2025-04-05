from diffusers import DiffusionPipeline
import torch
from matplotlib import pyplot as plt

if __name__ == '__main__':


    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe.to("cuda")

    # if using torch < 2.0
    # pipe.enable_xformers_memory_efficient_attention()

    prompt = "A high-resolution black and white MRI scan of a human brain, axial view, showing detailed brain structures with realistic textures and contrast. Medical imaging style, realistic lighting, and high anatomical accuracy."
    images = pipe(prompt=prompt).images[0]
    
    # If you want to display the image using Matplotlib
    plt.imshow(images)
    plt.axis('off')  # Hide axes
    plt.savefig('test_displayed.png', bbox_inches='tight', pad_inches=0)  # Save the displayed figure without extra padding
