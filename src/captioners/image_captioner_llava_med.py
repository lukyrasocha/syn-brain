import argparse
import os
import json
import torch
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm  

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria


def get_class_from_filename(filename):
    return filename.split("_")[0].lower()

def generate_prompt(class_name):
    if class_name == "notumor":
        return (
            "This brain MRI shows no tumor. Analyze this brain MRI. Output format in one line: tumor: yes/no; general_description: describe the brain MRI image in general. Max 77 tokens."
        )
    else:
        return (
            f"This image has {class_name}. Analyze this brain MRI. Output format in one line: tumor: yes/no; if yes—location: brain region; size: small/medium/large; shape; intensity; orientation: axial/saggital/coronal; general description: describe the brain MRI in general and also mention any other abnormalities. Max 77 tokens"
        )

def load_image(image_file: str) -> Image.Image:
    """Load an image from a local file or URL."""
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def describe_image(
    image_file: str,
    prompt_text: str,
    model_path: str = "microsoft/llava-med-v1.5-mistral-7b",
    device: str = "cuda",
    temperature: float = 0.2,
    max_new_tokens: int = 512,
    load_8bit: bool = False,
    load_4bit: bool = False,
) -> str:
    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, model_path, load_8bit, load_4bit, device=device
    )

    conv_mode = "mistral_instruct"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    image = load_image(image_file)
    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    if model.config.mm_use_im_start_end:
        final_prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt_text
    else:
        final_prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text

    conv.append_message(roles[0], final_prompt)
    conv.append_message(roles[1], None)
    full_prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if conv.sep_style == SeparatorStyle.TWO:
        stop_str = conv.sep2  
    else:
        stop_str = conv.sep   

    if stop_str:
        response = full_text.split(stop_str)[-1].strip()
    else:
        response = full_text.split(roles[1])[-1].strip()

    return response

def describe_all_images(
    model_path: str = "microsoft/llava-med-v1.5-mistral-7b",
    device: str = "cuda",
    temperature: float = 0.2,
    max_new_tokens: int = 512,
    load_8bit: bool = False,
    load_4bit: bool = False,
):
    """
    Process all images in the directories
    and save the generated descriptions along with image paths, tumor flags, and class info,
    in a JSON structured file.
    """
    directories = ["data/raw/Train_All_Images", "data/raw/Test_All_Images"]
    
    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, model_path, load_8bit, load_4bit, device=device
    )
    conv_mode = "mistral_instruct"
    base_conv = conv_templates[conv_mode].copy()
    roles = base_conv.roles

    results = []
    all_image_paths = []
    for dir_path in directories:
        if not os.path.exists(dir_path):
            print(f"Directory '{dir_path}' does not exist. Skipping.")
            continue

        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, file)
                    all_image_paths.append(full_path)


    for full_path in tqdm(all_image_paths, desc="Processing images", unit="image"):
        try:

            print(f"Processing {full_path}")
            base_name = os.path.basename(full_path)
            class_name = get_class_from_filename(base_name)
            prompt = generate_prompt(class_name)
            
            conv = base_conv.copy()
            if model.config.mm_use_im_start_end:
                final_prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt
            else:
                final_prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt

            conv.append_message(roles[0], final_prompt)
            conv.append_message(roles[1], None)
            full_prompt = conv.get_prompt()

            image = load_image(full_path)
            image_tensor = process_images([image], image_processor, model.config)
            if isinstance(image_tensor, list):
                image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)

            input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )
            full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            if conv.sep_style == SeparatorStyle.TWO:
                stop_str = conv.sep2  
            else:
                stop_str = conv.sep   
            if stop_str:
                caption = full_text.split(stop_str)[-1].strip()
            else:
                caption = full_text.split(roles[1])[-1].strip()

            print("="*50)
            print(class_name)
            print(caption)
            print("="*50)

            results.append({
                "image": base_name,
                "text": caption,
                "class": class_name,
                "path": full_path,
            })
            #c += 1

            #if c == 10:
            #    break

        except Exception as e:
            print(f"Error processing {full_path}: {e}")

    output_file = "captions_llava_med.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Generated descriptions saved in {output_file}")

def main():
    """
    Usage:
    For a single image:
      python describe_image.py --image path_or_URL --prompt "Describe the image as a radiologist"

    For processing all images in the datasets:
      python describe_image.py --all --prompt "Describe the image as a radiologist"
    """

    prompt = "Analyze brain MRI. Output format: tumor (yes/no); if yes—location (brain region), size (small/medium/large), shape, intensity. Also describe brain features, MRI orientation (axial/sagittal/coronal), and any other abnormalities."

    parser = argparse.ArgumentParser(description="Generate descriptions for images using LLaVA-Med")
    parser.add_argument("--image", type=str, help="Path or URL to the input image (ignored if --all is used)")
    parser.add_argument("--prompt", type=str, default=prompt, help="Instruction prompt")
    parser.add_argument("--model_path", type=str, default="microsoft/llava-med-v1.5-mistral-7b", help="Model identifier or local path")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate")
    parser.add_argument("--load_8bit", action="store_true", help="Load model in 8-bit mode")
    parser.add_argument("--load_4bit", action="store_true", help="Load model in 4-bit mode")
    parser.add_argument("--all", action="store_true", help="Process all images in data/Train_All_Images and data/Test_All_Images")
    args = parser.parse_args()


    if args.all:
        describe_all_images(
            model_path=args.model_path,
            device=args.device,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            load_8bit=args.load_8bit,
            load_4bit=args.load_4bit,
        )
    else:
        if not args.image:
            print("Provide an image path with --image flag if --all is not specified.")
            return
        description = describe_image(
            image_file=args.image,
            prompt_text=args.prompt,
            model_path=args.model_path,
            device=args.device,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            load_8bit=args.load_8bit,
            load_4bit=args.load_4bit,
        )
        print("Generated Description:")
        print(description)

if __name__ == "__main__":
    main()