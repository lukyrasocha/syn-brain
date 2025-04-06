import argparse
import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
from PIL import Image
import requests
from io import BytesIO


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

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    #stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)


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

def main():
    """
    call with
    python src/describe_image.py --image src/llava/chest.jpeg --prompt "Describe the image as a radiologist" 
    """
    parser = argparse.ArgumentParser(description="Generate a description for an image using LLaVA-Med")
    parser.add_argument("--image", type=str, required=True, help="Path or URL to the input image")
    parser.add_argument("--prompt", type=str, default="Describe the image as a radiologist.", help="Instruction prompt")
    parser.add_argument("--model_path", type=str, default="microsoft/llava-med-v1.5-mistral-7b", help="Model identifier or local path")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate")
    parser.add_argument("--load_8bit", action="store_true", help="Load model in 8-bit mode")
    parser.add_argument("--load_4bit", action="store_true", help="Load model in 4-bit mode")
    args = parser.parse_args()

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