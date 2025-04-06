import torch
from models import stable_diffusion_model
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
import random
import string
import sys

############################### DUMMY DATASET ########################################
class TextAndImageDataset(Dataset):
    def __init__(self, num_samples, image_size,prompt):
        self.num_samples = num_samples
        self.image_size = image_size
        self.prompt = prompt

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        prompt = self.prompt[idx]
        random_image = torch.randn(3, self.image_size, self.image_size)

        original_size = torch.tensor([self.image_size, self.image_size])
        crop_top_left = torch.tensor([0, 0])  

        return {
            'original_sizes': original_size,  
            'crop_top_lefts': crop_top_left,  
            'prompts': prompt,  # Raw text for prompt
            "images": random_image  # Random image tensor of shape [3, image_size, image_size]
        }

def create_dataloader(batch_size, image_size, num_samples=1000, prompt=None):
    dataset = TextAndImageDataset(
        num_samples=num_samples, 
        image_size=image_size, 
        prompt=prompt
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

########################################################################################



# Compute time IDs for spatial and temporal information in the image.
def compute_time_ids(original_size, crops_coords_top_left, resolution, device):
    target_size = torch.tensor([resolution, resolution])  
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids]).to(device, dtype=torch.float32)
    # print(f'crops_coords_top_left {crops_coords_top_left}')
    # print(f'original_size {original_size}')
    # print(f'target_size {target_size}')
    return add_time_ids


def train(model, train_dataloader, noise_scheduler, epochs, resolution, device='cuda'):
    
    for epoch in range(epochs):

            model.unet.train()
            train_loss = 0.0

            for step, batch in enumerate(train_dataloader):
                    
                    images = batch['images'].to(device)
                    print(f'images input shape: {images.shape}')

                    with autocast(device_type='cuda'):

                        # get the latent variable z
                        model_input = model.vae.encode(images).latent_dist.sample()
                        print(f'model output shape:{model_input.shape}')

                        # they use a scaling factor in the original code
                        print(model.vae.config.scaling_factor)
                        model_input = model_input * model.vae.config.scaling_factor

                        # Sample noise that we'll add to the latents
                        noise = torch.randn_like(model_input)
                        batch_size = model_input.shape[0]
                        print(f'noise shape: {noise.shape}')
                        print(f'bsz: {batch_size}')

                        # Sample a random timestep for each image
                        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
                        print(f'timesteps:', timesteps)

                        # apply this to our noise scheduler (use the one from the pretrained model)
                        noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

                        # add the spatial and temporal information in the image
                        add_time_ids = torch.cat([compute_time_ids(s, c, resolution, device) for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])])
                        print(f'add time ids cat: {add_time_ids}')
                        unet_added_conditions = {"time_ids": add_time_ids}


                        # https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/stable_diffusion_xl
                        # get the prompt embedings and the pooled proompt embeddings
                        prompt_embeds, _, pooled_prompt_embeds, _ = model.encode_prompt(prompt = batch['prompts'], device=device)
                        print(f"Prompt Embeds shape before reshape: {prompt_embeds.shape}")
                        print(f"Pooled Prompt Embeds shape before reshape: {pooled_prompt_embeds.shape}")

                        # will update what we will condition on in the unet
                        unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})

                        # Make the prediction from the UNet (denoising process)       
                        # https://huggingface.co/docs/diffusers/v0.32.2/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel                                         
                        model_pred = model.unet(
                            sample = noisy_model_input,                 # the noise inputs
                            timestep = timesteps,                       # sampled timesteps
                            encoder_hidden_states  = prompt_embeds,     # as I understand it, this is the input to the model
                            added_cond_kwargs=unet_added_conditions,    # As I understand it, this is what is conditioned on in the unet layers 
                            return_dict=False,
                        )[0]

if __name__ == '__main__':
    model, noise_scheduler = stable_diffusion_model('cuda')


    prompt = ['A random sentence for input 1'] * 1000
    image_size = 512
    train_dataloader = create_dataloader(batch_size=10, image_size=image_size, prompt=prompt)

    train(model, train_dataloader, noise_scheduler, 1, resolution=1024)






    # optimizer_class = torch.optim.AdamW
    # params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
    # optimizer = optimizer_class(
    #     params_to_optimize,
    #     lr=args.learning_rate,
    #     betas=(args.adam_beta1, args.adam_beta2),
    #     weight_decay=args.adam_weight_decay,
    #     eps=args.adam_epsilon,
    # )