import torch
from models import stable_diffusion_model
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
import random
import string
import sys

# Dataset class that returns the raw text (without tokenizing)
class TextAndImageDataset(Dataset):
    def __init__(self, num_samples, image_size, texts_1, texts_2):
        self.num_samples = num_samples
        self.image_size = image_size
        self.texts_1 = texts_1  # List of texts for input 1
        self.texts_2 = texts_2  # List of texts for input 2

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Get raw text and random image
        text_one = self.texts_1[idx]
        text_two = self.texts_2[idx]
        random_image = torch.randn(3, self.image_size, self.image_size)

        # Assuming that the original size is the same as the image size for each sample
        original_size = torch.tensor([self.image_size, self.image_size])
        crop_top_left = torch.tensor([0, 0])  # Assuming no crop, just set it to (0, 0)

        return {
            'original_sizes': original_size,  # Tensor of size [2] for original image size
            'crop_top_lefts': crop_top_left,  # Tensor of size [2] for crop coordinates
            'prompts': text_one,  # Raw text for prompt
            "images": random_image  # Random image tensor of shape [3, image_size, image_size]
        }
    
        # return {
        #     'original_sizes': (self.image_size, self.image_size),  # Assuming original size is the image size
        #     'crop_top_lefts': (0, 0), 
        #     'prompt': text_one,
        #     "images": random_image
        # }

# Example of how to use the Dataset and DataLoader
def create_dataloader(batch_size, image_size, num_samples=1000, texts_1=None, texts_2=None):
    dataset = TextAndImageDataset(
        num_samples=num_samples, 
        image_size=image_size, 
        texts_1=texts_1, 
        texts_2=texts_2
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)





# Compute the time_ids (spatial and temporal information)
def compute_time_ids(original_size, crops_coords_top_left, resolution, device):
    target_size = torch.tensor([resolution, resolution])  # Assuming no crop, just set it to (0, 0)
    # print(f'crops_coords_top_left {crops_coords_top_left}')
    # print(f'original_size {original_size}')
    # print(f'target_size {target_size}')
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids]).to(device, dtype=torch.float32)
    print('here', add_time_ids)
    return add_time_ids

def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

# # Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
# def encode_prompt(text_encoders, tokenizers, prompt):
#     prompt_embeds_list = []

#     for i, text_encoder in enumerate(text_encoders):
#         if tokenizers is not None:
#             tokenizer = tokenizers[i]
#             print('tokenizer', tokenizer)
#             print('promt', prompt)
#             text_input_ids = tokenize_prompt(tokenizer, prompt)
#             print('text_input_ids', text_input_ids.shape)


#         prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False)
#         print('prompt_embeds', type(prompt_embeds))
#         print('prompt_embeds len', len(prompt_embeds))


#         # We are only ALWAYS interested in the pooled output of the final text encoder
#         pooled_prompt_embeds = prompt_embeds[0]
#         print('prompt_embeds [0]', len(prompt_embeds))
#         prompt_embeds = prompt_embeds[-1][-2]
#         print('prompt_embeds [-1][-2]',prompt_embeds.shape)
#         bs_embed, seq_len, _ = prompt_embeds.shape
#         print('shape', prompt_embeds.shape)
#         prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
#         print('view prompt_embeds', prompt_embeds.shape)
#         prompt_embeds_list.append(prompt_embeds)

#     prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
#     pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
#     return prompt_embeds, pooled_prompt_embeds



def train(model, train_dataloader, noise_scheduler, epochs, resolution, device='cuda'):
    
    for epoch in range(epochs):

            model.unet.train()
            train_loss = 0.0

            for step, batch in enumerate(train_dataloader):
                    images = batch['images'].to(device)
                    print(f'images input shape: {images.shape}')

                    with autocast(device_type='cuda'):
                        model_input = model.vae.encode(images).latent_dist.sample()
                            
                        print(f'model output shape:{model_input.shape}')
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

                        noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

                        add_time_ids = torch.cat([compute_time_ids(s, c, resolution, device) for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])])
                        print(f'add time ids cat: {add_time_ids}')

                        # Encode the text prompts
                        unet_added_conditions = {"time_ids": add_time_ids}
                        # prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        #                                                     text_encoders=[model.text_encoder, model.text_encoder_2], # model.text_encoder_2
                        #                                                     tokenizers=[model.tokenizer, model.tokenizer_2], # model.tokenizer_2
                        #                                                     prompt=batch['prompt'],
                        #                                                 )
                        print('here')
                        # https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/stable_diffusion_xl
                        prompt_embeds, _, pooled_prompt_embeds, _ = model.encode_prompt(prompt = batch['prompts'], device=device)

                        print(f"Prompt Embeds shape before reshape: {prompt_embeds.shape}")
                        print(f"Pooled Prompt Embeds shape before reshape: {pooled_prompt_embeds.shape}")

                        unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})

                        # Make the prediction from the UNet (denoising process)       
                        # https://huggingface.co/docs/diffusers/v0.32.2/en/api/models/unet2d-cond#diffusers.UNet2DConditionModel                                         
                        model_pred = model.unet(
                            sample = noisy_model_input,
                            timestep = timesteps,
                            encoder_hidden_states  = prompt_embeds,
                            added_cond_kwargs=unet_added_conditions,
                            return_dict=False,
                        )[0]

                        break
                        if noise_scheduler.config.prediction_type == "epsilon":
                            print('YESSSSSSS')
                    # Compute the loss (e.g., L2 loss or other)
                    loss = ((model_pred - model_input) ** 2).mean()

                    # Backpropagate the loss
                    loss.backward()

                    # Update the optimizer
                    optimizer.step()
                    optimizer.zero_grad()

                    # Log the loss (optional)
                    train_loss += loss.item()
                    break

if __name__ == '__main__':
    model, noise_scheduler = stable_diffusion_model('cuda')


    texts_1 = ['A random sentence for input 1'] * 1000
    texts_2 = ['Another random sentence for input 2'] * 1000
    image_size = 512
    train_dataloader = create_dataloader(batch_size=10, image_size=image_size, texts_1=texts_1, texts_2=texts_2)

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