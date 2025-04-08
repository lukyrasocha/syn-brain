import os
import json 
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# root directory is data/raw/training
ROOT_DIR_TRAINING = "data/raw/Train_All_Images"
# root directory is data/raw/testing
ROOT_DIR_TESTING = "data/raw/Test_All_Images"

JSON_PATH = "data/json_files/test.json"



class BrainTumorDataset(Dataset):
    """
    A dataset that reads image filenames & prompts from a JSON file.
    All images are assumed to be in ONE folder (root_dir).
    Optional filtering by substring (class_filter).
    """
    def __init__(self, root_dir, json_path, transform=None, class_filter=None):
        """
        Args:
            root_dir (str): Folder containing images.
            json_path (str): Path to a JSON file with { "image": str, "text": str } entries.
            transform (callable, optional): Torch transform (e.g. Resize, Normalize).
            class_filter (str or list, optional): If given, only filenames containing these substrings
                                                 will be included (e.g. "glioma").
        """
        self.root_dir = root_dir
        self.transform = transform
        
        
        with open(json_path, 'r') as f: 
            all_samples = json.load(f)
            
            
        if isinstance(class_filter, str):
            class_filter = [class_filter]
            
        self.samples = []
        for entry in all_samples:
            image_name = entry["image"]
            text = entry["text"]
            
            # if class filter, skip files that do not match 
            if class_filter:
                if not any(cf.lower in image_name.lower() for cf in class_filter):
                    continue
            
            # path to the images 
            image_path = os.path.join(self.root_dir, image_name)
            
            # store in the dataset 
            self.samples.append({"image": image_path, "text": text}) 

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        image_path = data["image"]
        text = data["text"]
        
        image = Image.open(image_path).convert("L").convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        return {"image": image, "text": text}
    
    
#################################################################################################################
#################################################################################################################
#################################################################################################################
    
#################################################
# 3) Dummy Training Loop (Skeleton)
#################################################
def train_model(dataloader):
    """
    This function just demonstrates how you'd iterate over your dataset.
    Real fine-tuning for SD is more involved (DreamBooth, LoRA, etc.).
    """
    # Imagine we have a "model" and an "optimizer"
    for epoch in range(1):  # example: 1 epoch
        for i, batch in enumerate(dataloader):
            images = batch["image"]  # shape: (B, 3, 512, 512)
            texts = batch["text"]    # list of strings
            print(f"[Epoch {epoch} - Batch {i}] => images shape: {images.shape}, text: {texts[:2]}")
            # Here you'd tokenize 'texts', feed (images, text_embeds) into a diffusion model, etc.
            # We'll break early for demonstration
            break


#################################################
# 4) Inference with Stable Diffusion v1.5
#################################################
def do_inference(prompt, negative_prompt=None, output_filename="sd15_output.png"):
    """
    Generates an image from a text prompt using Stable Diffusion v1.5.
    """
    from diffusers import StableDiffusionPipeline

    # Load the SD v1.5 pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,  # half precision
        safety_checker=None,        # disable safety for quick testing
        requires_safety_checker=False
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Generating image for prompt: '{prompt}'")
    # You can pass negative_prompt if you want
    results = pipeline(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=10)
    image = results.images[0]
    image.save(output_filename)
    print(f"Image saved to {output_filename}")

#################################################################################################################
#################################################################################################################
#################################################################################################################
if __name__ == "__main__":
    sd_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_dataset = BrainTumorDataset(
        root_dir=ROOT_DIR_TRAINING,
        json_path=JSON_PATH,
        transform=sd_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # Run dummy training
    train_model(train_loader)

    # Run a sample inference with v1.5
    prompt = "An astronaut riding a horse in outer space, cinematic lighting, ultra realistic"
    negative_prompt = "blurry, distorted, text, watermark, low quality"
    do_inference(prompt, negative_prompt, output_filename="astronaut_sd15.png")
