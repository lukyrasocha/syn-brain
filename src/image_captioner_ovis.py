import argparse
import os
import json
import torch
import requests
from io import BytesIO
from PIL import Image
from transformers import AutoModelForCausalLM

def generate_prompt(class_name):
    if class_name == "notumor":
        return (
            "This brain MRI shows no tumor. Analyze this brain MRI. Output format in one line: tumor: yes/no; general_description: describe the brain MRI image in general. Max 77 tokens."
        )
    else:
        return (
            f"This image has {class_name}. Analyze this brain MRI. Output format in one line: tumor: yes/no; if yes—location: brain region; size: small/medium/large; shape; intensity; orientation: axial/saggital/coronal; general description: describe the brain MRI in general and also mention any other abnormalities. Max 77 tokens"
        )

def load_image(image_path: str) -> Image.Image:
    """Load an image from a local file or URL and convert it to RGB."""
    if image_path.startswith("http://") or image_path.startswith("https://"):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    return image

def caption_image(image_path: str, prompt: str, model, text_tokenizer, visual_tokenizer, max_partition: int = 9) -> str:
    """
    Generate a caption for a single image using the Ovis model.
    """
    image = load_image(image_path)
    images = [image]
    
    query = f"<image>\n{prompt}"
    
    _, input_ids, pixel_values = model.preprocess_inputs(query, images, max_partition=max_partition)
    
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
    if pixel_values is not None:
        pixel_values = pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)
    pixel_values = [pixel_values]
    
    gen_kwargs = dict(
        max_new_tokens=1024,
        do_sample=False,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=text_tokenizer.pad_token_id,
        use_cache=True
    )
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids, 
            pixel_values=pixel_values, 
            attention_mask=attention_mask, 
            **gen_kwargs
        )[0]
    output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
    return output

def process_all_images(directory: str, model, text_tokenizer, visual_tokenizer, max_partition: int = 9):
    """
    process all images in a directory and generate captions.
    """

    directories = ["data/raw/Train_All_Images", "data/raw/Test_All_Images"]

    results = []

    for dir_path in directories:
        if not os.path.exists(dir_path):
            print(f"Directory '{dir_path}' does not exist. Skipping.")
            continue
        for root, _, files in os.walk(dir_path):
            for file in files:
                if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                full_path = os.path.join(root, file)
                try:

                    base_name = os.path.basename(file)
                    class_name = base_name.split("_")[0]

                    prompt = generate_prompt(class_name)

                    caption = caption_image(full_path, prompt, model, text_tokenizer, visual_tokenizer, max_partition)
                    print("=" * 50)
                    print(f"Image: {full_path}")
                    print("Caption:")
                    print(caption)
                    print("=" * 50)
                    results.append({
                        "file_name": base_name,
                        "text": caption,
                        "class": class_name,
                        "path": full_path,
                    })
                except Exception as e:
                    print(f"Error processing {full_path}: {e}")
        return results

def main():

    prompt = "Analyze this brain MRI. Output format: tumor (yes/no); if yes—location (brain region), size (small/medium/large), shape, intensity. Also describe brain features, MRI orientation (axial/sagittal/coronal), and any other abnormalities."

    parser = argparse.ArgumentParser(description="Generate image captions using the Ovis model.")
    parser.add_argument("--image", type=str, help="Path or URL to the input image (ignored if --all is used)")
    parser.add_argument("--all", action="store_true", help="Process all images in the specified directory")
    parser.add_argument("--dir", type=str, default="images", help="Directory containing images (used with --all)")
    parser.add_argument("--prompt", type=str, default=prompt, help="Caption prompt text")
    parser.add_argument("--model_path", type=str, default="AIDC-AI/Ovis2-34B", help="Ovis model identifier or local path")
    parser.add_argument("--max_partition", type=int, default=9, help="Max partition parameter for multimodal processing")
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        multimodal_max_length=32768,
        trust_remote_code=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()
    
    if args.all:
        results = process_all_images(args.dir, model, text_tokenizer, visual_tokenizer, args.max_partition)
        output_file = "captions_ovis_large.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Captions for all images have been saved to {output_file}")
    else:
        if not args.image:
            print("Please provide an image path using the --image flag, or use --all to process a directory.")
            return
        caption = caption_image(args.image, args.prompt, model, text_tokenizer, visual_tokenizer, args.max_partition)
        print("Generated caption:")
        print(caption)

if __name__ == "__main__":
    main()