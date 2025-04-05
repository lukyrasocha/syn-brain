import torch
from datasets import load_dataset
from diffusers import StableDiffusionXLPipeline
from peft import get_peft_model, LoraConfig
from PIL import Image
from torchvision import transforms

# Load dataset from Hugging Face
dataset = load_dataset("lambdalabs/pokemon-blip-captions", split="train")

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# Load Stable Diffusion XL with LoRA-ready UNet
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")

# Apply LoRA to UNet
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v", "to_out"],
    lora_dropout=0.1,
    bias="none",
    task_type="text-to-image"
)
pipe.unet = get_peft_model(pipe.unet, lora_config)

# Optimizer
optimizer = torch.optim.Adam(pipe.unet.parameters(), lr=1e-5)

# Tiny training loop (1 epoch, 10 samples)
for i, example in enumerate(dataset.select(range(10))):  # Only first 10 samples
    caption = example["text"]
    image = Image.open(example["image"]).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to("cuda").half()

    # Encode image → latents
    latents = pipe.vae.encode(image_tensor).latent_dist.sample()
    noise = torch.randn_like(latents)
    noisy_latents = latents + noise

    # Encode caption → embeddings
    text_inputs = pipe.tokenizer(caption, return_tensors="pt").input_ids.to("cuda")
    prompt_embeds = pipe.text_encoder(text_inputs)[0]

    # UNet prediction
    noise_pred = pipe.unet(noisy_latents, torch.tensor([0.0]).to("cuda"), encoder_hidden_states=prompt_embeds).sample

    # Loss
    loss = torch.nn.functional.mse_loss(noise_pred, noise)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"[{i}] Prompt: {caption} | Loss: {loss.item():.4f}")

# import torch
# from diffusers import DiffusionPipeline
# from matplotlib import pyplot as plt
# from accelerate.utils import write_basic_config

# if __name__ == '__main__':


#     pipe = DiffusionPipeline.from_pretrained(
#         "stabilityai/stable-diffusion-xl-base-1.0", # model name
#         torch_dtype=torch.float16,                  # load as 16-bit floating point for faster inference
#         use_safetensors=True,                       # a safe and fast way of loading the model
#         variant="fp16"                              # stands for "floating point 16-bit"
#         ).to("cuda")

#     prompt = "A high-resolution black and white MRI scan of a human brain, axial view, showing detailed brain structures with realistic textures and contrast. Medical imaging style, realistic lighting, and high anatomical accuracy."
#     images = pipe(prompt=prompt).images[0]
    
#     plt.imshow(images)
#     plt.axis('off')  
#     plt.savefig('test_displayed.png', bbox_inches='tight', pad_inches=0)  #












    # pipe = AutoPipelineForText2Image.from_pretrained('runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16)
    # pipe.load_lora_weights('DoctorDiffusion/doctor-diffusion-s-xray-xl-lora', weight_name='DD-xray-v1.safetensors')

    # pipe = AutoPipelineForText2Image.from_pretrained('runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16)
    # pipe.load_lora_weights('DoctorDiffusion/doctor-diffusion-s-xray-xl-lora', weight_name='DD-xray-v1.safetensors')