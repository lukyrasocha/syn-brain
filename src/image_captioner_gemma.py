import argparse
import os
import json
import torch
import gc
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig

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
    return Image.open(image_file).convert("RGB")

def setup_model(model_id, hf_token=None, use_quantization=False):
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        token_param = {"token": hf_token}
    else:
        token_param = {}

    quantization_config = BitsAndBytesConfig(load_in_8bit=True) if use_quantization else None
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quantization_config,
        **token_param
    ).eval()

    processor = AutoProcessor.from_pretrained(model_id, **token_param)
    return model, processor

def describe_image(image_file, prompt_text, model, processor, max_new_tokens):
    image = load_image(image_file)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image", "image": image}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    generated_tokens = outputs[0][input_len:]
    caption = processor.decode(generated_tokens, skip_special_tokens=True)
    return caption.replace('\n', ' ').strip()

def describe_all_images(
    model_id="google/gemma-3-12b-it",
    hf_token=None,
    use_quantization=False,
    max_new_tokens=77
):
    directories = ["data/raw/Train_All_Images", "data/raw/Test_All_Images"]
    supported_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    model, processor = setup_model(model_id, hf_token, use_quantization)

    results = []
    for dir_path in directories:
        if not os.path.exists(dir_path):
            print(f"Directory '{dir_path}' does not exist. Skipping.")
            continue

        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith(supported_exts):
                    full_path = os.path.join(root, file)
                    class_name = file.split("_")[0]
                    prompt = generate_prompt(class_name)

                    try:
                        caption = describe_image(
                            full_path, prompt, model, processor, max_new_tokens
                        )

                        print("="*50)
                        print(class_name)
                        print(caption)
                        print("="*50)

                        results.append({
                            "image": file,
                            "text": caption,
                            "class": class_name,
                            "path": full_path
                        })

                        torch.cuda.empty_cache()
                        gc.collect()

                    except Exception as e:
                        print(f"Error processing {full_path}: {e}")

    with open("captions_gemma.jsonl", "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    print("✅ Generated descriptions saved to captions_gemma.jsonl")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="google/gemma-3-12b-it")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=77)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--image", type=str, help="Path to a single image")
    args = parser.parse_args()

    if args.all:
        describe_all_images(
            model_id=args.model_id,
            hf_token=args.hf_token,
            use_quantization=args.quantize,
            max_new_tokens=args.max_new_tokens
        )
    elif args.image:
        model, processor = setup_model(args.model_id, args.hf_token, args.quantize)
        class_name = os.path.basename(args.image).split("_")[0]
        prompt = generate_prompt(class_name)
        caption = describe_image(args.image, prompt, model, processor, args.max_new_tokens)
        print("Generated Caption:\n", caption)
    else:
        print("Use --all to process all images or --image to run on one image")

if __name__ == "__main__":
    main()
